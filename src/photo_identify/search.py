"""自然语言搜索模块：支持普通 FTS 检索和基于 Embedding 向量的语义检索。

提供两种搜索策略：
1. 普通搜索（默认）：优先 AND 多词 FTS 匹配，降级 OR 匹配，完全本地运行无网络请求。
2. 智能搜索（--smart 模式）：使用 Embedding 模型将查询转化为向量，并与图片库中的描述向量计算余弦相似度，实现彻底的语义匹配。
"""

import json
import logging
import re
import sys
import urllib.error
import urllib.request

from photo_identify.config import DEFAULT_BASE_URL, DEFAULT_SEARCH_LIMIT, DEFAULT_TEXT_MODEL
from photo_identify.embedding_runtime import get_text_embedding_sync
from photo_identify.storage import Storage

logger = logging.getLogger(__name__)


def _get_query_embedding(
    query: str,
    api_key: str,
    base_url: str,
    model: str,
    backend: str = "",
    workers: int = 1,
) -> list[float]:
    """根据指定后端获取用户查询文本的 Embedding 向量。"""

    try:
        return get_text_embedding_sync(
            text=query,
            model=model,
            backend=backend,
            api_key=api_key,
            base_url=base_url,
            workers=workers,
        )
    except Exception as exc:
        logger.warning("获取 Query Embedding 失败: %s", exc)
        print(f"  ⚠️ 获取语义向量失败: {exc}", file=sys.stderr)
    return []


_RERANK_SYSTEM_PROMPT = """\
你是一个图片检索重排序助手。请立即、直接输出 JSON 数组结果，禁止输出任何思维过程(Reasoning/Thinking)或解释说明。
用户提供了一个搜索描述，以及若干候选图片的序号和内容信息。
你的任务是：深入理解用户的语言意图，从候选图片中选出相对最符合要求的图片，并按匹配程度从高到低排序返回。

返回规则严格遵守：
直接输出选中的候选图片序号的 JSON 数组，例如：
[3, 1, 5]
如果没有符合要求的，返回 []。不要输出任何除了方括号和数字之外的内容（不要加 ```json 标记）。"""

def _llm_rerank_results(query: str, results: list[dict], api_key: str, base_url: str, model: str) -> tuple[list[dict], str]:
    if not results:
        return [], ""

    max_rerank = 20
    to_rerank = results[:max_rerank]
    remainder = results[max_rerank:]

    candidates_text = []
    id_map = {}
    for i, r in enumerate(to_rerank, 1):
        id_map[i] = r
        scene = r.get("scene", "")
        objects = r.get("objects", [])
        if isinstance(objects, list):
            objects_str = ", ".join(objects)
        else:
            objects_str = str(objects)
        text = f"[{i}] 场景: {scene} | 包含物体: {objects_str}"
        candidates_text.append(text)
    
    user_content = f"真实搜索意图：{query}\n\n候选图片列表：\n" + "\n".join(candidates_text)
    
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 10240,
        "messages": [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    
    url = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        msg = data.get("choices", [{}])[0].get("message", {})
        ans = msg.get("content", "").strip()
        
        ans = re.sub(r'<\|[^|]*\|>', '', ans).strip()
        if not ans:
            raise ValueError("模型返回了空内容")
        
        array_match = re.search(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', ans)
        if array_match:
            ans = array_match.group()
        else:
            ans = ans.replace("```json", "").replace("```", "").strip()
        
        best_indices = json.loads(ans)
        if not isinstance(best_indices, list):
            raise ValueError("结果不是数组")
        
        reranked = []
        seen = set()
        for idx in best_indices:
            try:
                idx_int = int(idx)
                if idx_int in id_map and idx_int not in seen:
                    reranked.append(id_map[idx_int])
                    seen.add(idx_int)
            except ValueError:
                continue
                
        print(f"  🧠 LLM 排序原始返回: {ans}, 匹配出 {len(reranked)} 项")
        
        for i, r in enumerate(to_rerank, 1):
            if i not in seen:
                reranked.append(r)
                
        return reranked + remainder, ""

    except Exception as exc:
        logger.warning("LLM 重排序失败: %s", exc)
        print(f"  ⚠️ LLM 排序失败，回退默认顺序: {exc}", file=sys.stderr)
        return results, f"LLM 排序失败，已回退默认顺序: {exc}"


def search(
    query: str,
    db_paths: str | list[str],
    limit: int = DEFAULT_SEARCH_LIMIT,
    smart: bool = False,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_TEXT_MODEL,
    rerank: bool = False,
    embedding_model: str = "",
    embedding_base_url: str = "",
    embedding_api_key: str = "",
    embedding_backend: str = "",
    embedding_workers: int = 1,
) -> tuple[list[dict], list[str]]:
    """执行图片搜索并返回匹配结果。

    Args:
        query: 搜索文本。
        db_paths: SQLite 数据库文件路径。
        limit: 最大返回条数。
        smart: 是否启用基于 Embedding 的纯语义向量搜索。
        api_key: 文本处理模型 API Key（用于 rerank）。
        base_url: 文本处理模型 API Base URL。
        model: 文本处理模型名称（用于 rerank）。
        rerank: 是否使用 LLM 对召回结果进行二次排序。
        embedding_model: 向量模型名称。
        embedding_base_url: 向量模型 API Base URL。
        embedding_api_key: 向量模型 API Key。
        embedding_backend: 向量模型运行方式，支持 ``api`` 或 ``local``。
        embedding_workers: 本地向量模型编码时使用的批大小提示。

    Returns:
        (results, warnings) 二元组：匹配的图片记录列表 和 警告信息列表。
    """
    query_emb = None
    if smart:
        if not embedding_model:
            print("  ⚠️ 智能模式(Semantic Search)需要配置向量模型，回退为普通本地搜索", file=sys.stderr)
            smart = False
        else:
            emb_key = embedding_api_key or api_key
            emb_url = embedding_base_url or base_url
            print(f"  🔍 正在获取检索词的语义向量...")
            query_emb_list = _get_query_embedding(
                query,
                emb_key,
                emb_url,
                embedding_model,
                backend=embedding_backend,
                workers=embedding_workers,
            )
            if query_emb_list:
                import numpy as np
                query_emb = np.array(query_emb_list, dtype=np.float32)
            else:
                smart = False

    if isinstance(db_paths, str):
        db_paths = [db_paths]
        
    results = []
    warnings = []
    
    for db in db_paths:
        try:
            storage = Storage(db)
            if query_emb is not None:
                import numpy as np
                all_embs = storage.get_all_embeddings()
                if all_embs:
                    ids = []
                    vecs = []
                    for img_id, emb_bytes in all_embs:
                        try:
                            vec = np.frombuffer(emb_bytes, dtype=np.float32)
                            vecs.append(vec)
                            ids.append(img_id)
                        except Exception:
                            continue
                    
                    if vecs:
                        matrix = np.stack(vecs)
                        norm_matrix = np.linalg.norm(matrix, axis=1)
                        norm_q = np.linalg.norm(query_emb)
                        
                        # 避免除以 0
                        norm_matrix[norm_matrix == 0] = 1e-9
                        norm_q = norm_q if norm_q != 0 else 1e-9
                        
                        sims = np.dot(matrix, query_emb) / (norm_matrix * norm_q)
                        
                        # 挑选 top K，加入小阈值过滤完全无关的内容
                        top_indices = np.argsort(sims)[::-1][:limit * 2]
                        top_ids = [ids[i] for i in top_indices if sims[i] > 0.05]
                        
                        if top_ids:
                            db_results = storage.get_images_by_ids(top_ids)
                            # 根据 ID 重建相似度字典并赋值 score，负数越小越靠前
                            sim_dict = {ids[i]: float(sims[i]) for i in top_indices}
                            for r in db_results:
                                r["score"] = -sim_dict.get(r["id"], 0.0)
                                r["db_path"] = db
                            db_results.sort(key=lambda x: x["score"])
                            results.extend(db_results)
            else:
                # 纯本地多级 FTS 检索
                fetch_limit = limit * 2 if rerank else limit
                db_results = storage.search_fts(query, fetch_limit)
                for r in db_results:
                    r["db_path"] = db
                results.extend(db_results)
            storage.close()
        except Exception as e:
            logger.warning("在数据库 %s 中搜索失败: %s", db, e)
            print(f"  ⚠️ 读取数据库出错 {db}: {e}", file=sys.stderr)
            
    # 如果有多个库，全局排序
    results.sort(key=lambda x: x.get("score", 0.0))
    fetch_limit = limit * 2 if rerank else limit
    results = results[:fetch_limit]
    
    if rerank and results:
        results, rerank_warn = _llm_rerank_results(query, results, api_key, base_url, model)
        if rerank_warn:
            warnings.append(rerank_warn)
            
    return results[:limit], warnings


def format_results(results: list[dict]) -> str:
    if not results:
        return "未找到匹配的图片。"

    lines = [f"找到 {len(results)} 张匹配图片：\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. {r.get('file_name', '?')}")
        lines.append(f"     路径: {r.get('path', '?')}")
        scene = r.get("scene", "")
        if scene:
            lines.append(f"     场景: {scene[:100]}{'…' if len(scene) > 100 else ''}")
        objects = r.get("objects", "")
        if objects:
            lines.append(f"     物体: {objects[:100]}{'…' if len(objects) > 100 else ''}")
        lines.append("")
    return "\n".join(lines)
