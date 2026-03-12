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
        print(f"  [WARN] 获取语义向量失败: {exc}", file=sys.stderr)
    return []


def _build_query_variants(query: str) -> list[tuple[str, float]]:
    """构造本地查询变体，用于提升向量检索对中文近义表达的鲁棒性。"""

    normalized_query = query.strip()
    if not normalized_query:
        return []

    variants: list[tuple[str, float]] = [(normalized_query, 1.0)]
    seen = {normalized_query}

    def add_variant(text: str, weight: float) -> None:
        candidate = text.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        variants.append((candidate, weight))

    replacement_groups = {
        "小孩": ["小男孩", "小女孩", "儿童", "孩子", "宝宝", "小朋友"],
        "孩子": ["小孩", "儿童", "宝宝", "小朋友", "小男孩", "小女孩"],
        "儿童": ["小孩", "孩子", "宝宝", "小朋友", "小男孩", "小女孩"],
        "宝宝": ["小孩", "孩子", "儿童", "小朋友"],
        "小朋友": ["小孩", "孩子", "儿童", "小男孩", "小女孩"],
        "男孩": ["小男孩", "小孩", "儿童", "孩子"],
        "女孩": ["小女孩", "小孩", "儿童", "孩子"],
    }
    phrase_replacements = {
        "骑马": ["骑木马", "坐木马", "骑在木马上"],
        "吃苹果": ["啃苹果", "咬苹果", "拿着苹果", "吃着苹果"],
    }

    subject_variants = [normalized_query]
    action_variants = [normalized_query]

    for source, targets in replacement_groups.items():
        if source in normalized_query:
            for target in targets:
                replaced = normalized_query.replace(source, target)
                add_variant(replaced, 0.92)
                subject_variants.append(replaced)

    for source, targets in phrase_replacements.items():
        if source in normalized_query:
            for target in targets:
                replaced = normalized_query.replace(source, target)
                add_variant(replaced, 0.88)
                action_variants.append(replaced)

    for subject_text in subject_variants[:8]:
        for source, targets in phrase_replacements.items():
            if source in subject_text:
                for target in targets:
                    add_variant(subject_text.replace(source, target), 0.84)

    for action_text in action_variants[:8]:
        for source, targets in replacement_groups.items():
            if source in action_text:
                for target in targets:
                    add_variant(action_text.replace(source, target), 0.84)

    return variants[:16]


def _compute_text_match_bonus(
    query: str,
    row: dict,
    lexical_score: float | None = None,
    query_variants: list[tuple[str, float]] | None = None,
) -> float:
    """为候选图片计算基于整句文本命中的温和加权。"""

    variants = query_variants or _build_query_variants(query)
    if not variants:
        return 0.0

    objects_text = str(row.get("objects") or "")
    scene_text = str(row.get("scene") or "")
    style_text = str(row.get("style") or "")
    location_time_text = str(row.get("location_time") or "")
    wallpaper_hint_text = str(row.get("wallpaper_hint") or "")
    file_name_text = str(row.get("file_name") or "")

    bonus = 0.0
    for variant_text, variant_weight in variants:
        field_weight = variant_weight if variant_text == query.strip() else variant_weight * 0.9
        if variant_text in objects_text:
            bonus -= 0.90 * field_weight
        if variant_text in scene_text:
            bonus -= 0.45 * field_weight
        if variant_text in style_text:
            bonus -= 0.08 * field_weight
        if variant_text in location_time_text:
            bonus -= 0.06 * field_weight
        if variant_text in wallpaper_hint_text:
            bonus -= 0.06 * field_weight
        if variant_text in file_name_text:
            bonus -= 0.04 * field_weight

    if lexical_score is not None:
        if lexical_score <= -1000.0:
            bonus -= 0.12
        elif lexical_score <= -100.0:
            bonus -= 0.06
        elif lexical_score < -1.0:
            bonus -= 0.03

    return bonus


def _has_strong_text_match(query: str, row: dict, query_variants: list[tuple[str, float]] | None = None) -> bool:
    """判断候选是否具备足够强的整句文本命中，可作为语义召回的兜底候选。"""

    variants = query_variants or _build_query_variants(query)
    if not variants:
        return False

    fields = (
        str(row.get("objects") or ""),
        str(row.get("scene") or ""),
        str(row.get("style") or ""),
        str(row.get("location_time") or ""),
        str(row.get("wallpaper_hint") or ""),
        str(row.get("file_name") or ""),
    )
    for variant_text, variant_weight in variants:
        if variant_weight < 0.85:
            continue
        if any(variant_text in field for field in fields):
            return True
    return False


def _build_query_concept_groups(query: str) -> list[tuple[tuple[str, ...], float, float]]:
    """根据查询提取关键概念组，用于约束主体/动作/物体一致性。"""

    normalized_query = query.strip()
    if not normalized_query:
        return []

    groups: list[tuple[tuple[str, ...], float, float]] = []
    child_terms = ("小孩", "孩子", "儿童", "宝宝", "小朋友", "男孩", "女孩", "小男孩", "小女孩")
    ride_terms = ("骑马", "骑木马", "坐木马", "骑在木马上", "木马玩具", "木马", "马")
    apple_terms = ("苹果",)
    eat_terms = ("吃苹果", "吃着苹果", "啃苹果", "咬苹果", "拿着苹果", "拿苹果", "吃", "啃", "咬")

    if any(term in normalized_query for term in child_terms):
        groups.append((child_terms, 0.42, 0.28))
    if any(term in normalized_query for term in ("骑马", "骑木马", "坐木马", "木马", "马")):
        groups.append((ride_terms, 0.34, 0.18))
    if "苹果" in normalized_query:
        groups.append((apple_terms, 0.32, 0.18))
    if any(term in normalized_query for term in ("吃", "啃", "咬")):
        groups.append((eat_terms, 0.14, 0.06))

    return groups


def _compute_concept_coverage_bonus(query: str, row: dict) -> float:
    """按查询概念覆盖情况给予奖励/惩罚，避免主体错误的语义近邻排到前面。"""

    concept_groups = _build_query_concept_groups(query)
    if not concept_groups:
        return 0.0

    text = " | ".join(
        [
            str(row.get("scene") or ""),
            str(row.get("objects") or ""),
            str(row.get("style") or ""),
            str(row.get("location_time") or ""),
            str(row.get("wallpaper_hint") or ""),
            str(row.get("file_name") or ""),
        ]
    )

    bonus = 0.0
    for terms, reward, penalty in concept_groups:
        if any(term in text for term in terms):
            bonus -= reward
        else:
            bonus += penalty
    return bonus


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
                
        print(f"  [INFO] LLM 排序原始返回: {ans}, 匹配出 {len(reranked)} 项")
        
        for i, r in enumerate(to_rerank, 1):
            if i not in seen:
                reranked.append(r)
                
        return reranked + remainder, ""

    except Exception as exc:
        logger.warning("LLM 重排序失败: %s", exc)
        print(f"  [WARN] LLM 排序失败，回退默认顺序: {exc}", file=sys.stderr)
        return results, f"LLM 排序失败，已回退默认顺序: {exc}"


_EXPAND_SYSTEM_PROMPT = """\
你是一个查询词拓展助手。用户会提供一段简短的搜索意图，你的任务是提炼和拓展相关词汇，以帮助搜索引擎检索到更多可能的相关图片。
请直接输出拓展后的同义词或相关词汇，词与词之间使用空格隔开。
严格遵守：不要输出任何思维过程，不要带任何标点符号。比如：
用户输入：小孩在吃苹果
你的输出：儿童 宝宝 男孩 女孩 水果 啃 咬"""

def _llm_expand_query(query: str, api_key: str, base_url: str, model: str) -> str:
    """使用 LLM 对用户的意图进行分词拓展。"""
    if not query.strip():
        return ""
        
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": _EXPAND_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    
    url = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        msg = data.get("choices", [{}])[0].get("message", {})
        ans = msg.get("content", "").strip()
        ans = re.sub(r'<\|[^|]*\|>', '', ans).strip()
        
        # 移除非空格的部分标点
        import string
        ans = ans.translate(str.maketrans("", "", string.punctuation + "，。！？；：“”‘’（）【】《》"))
        print(f"  [INFO] LLM 拓展分词结果: {ans[:50]}...")
        return query + " " + ans
        
    except Exception as exc:
        logger.warning("LLM 拓展查询词失败: %s", exc)
        print(f"  [WARN] LLM 拓展查询词失败，回退原始搜索: {exc}", file=sys.stderr)
        return query


def search(
    query: str,
    db_paths: str | list[str],
    limit: int = DEFAULT_SEARCH_LIMIT,
    smart: bool = False,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_TEXT_MODEL,
    rerank: bool = False,
    expand_query: bool = False,
    local_expand: bool = True,
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
        api_key: 文本处理模型 API Key（用于 rerank/expand_query）。
        base_url: 文本处理模型 API Base URL。
        model: 文本处理模型名称（用于 rerank/expand_query）。
        rerank: 是否使用 LLM 对召回结果进行二次排序。
        expand_query: 是否启用 LLM 兜底拓展（仅当本地拓展结果不足时触发）。
        local_expand: 是否启用本地分词拓展。
        embedding_model: 向量模型名称。
        embedding_base_url: 向量模型 API Base URL。
        embedding_api_key: 向量模型 API Key。
        embedding_backend: 向量模型运行方式，支持 ``api`` 或 ``local``。
        embedding_workers: 本地向量模型编码时使用的批大小提示。

    Returns:
        (results, warnings) 二元组：匹配的图片记录列表 和 警告信息列表。
    """
    should_expand_query = expand_query and bool(model)
    base_query = query
    expanded_query = ""

    query_emb = None
    query_variants = _build_query_variants(query) if local_expand else [(query.strip(), 1.0)]
    query_variant_weights: list[float] = []
    if smart:
        if not embedding_model:
            print("  [WARN] 智能模式(Semantic Search)需要配置向量模型，回退为普通本地搜索", file=sys.stderr)
            smart = False
        else:
            emb_key = embedding_api_key or api_key
            emb_url = embedding_base_url or base_url
            print("  [INFO] 正在获取检索词的语义向量...")
            query_embeddings = []
            active_variants: list[tuple[str, float]] = []
            for variant_text, variant_weight in query_variants:
                query_emb_list = _get_query_embedding(
                    variant_text,
                    emb_key,
                    emb_url,
                    embedding_model,
                    backend=embedding_backend,
                    workers=embedding_workers,
                )
                if query_emb_list:
                    import numpy as np

                    query_embeddings.append(np.array(query_emb_list, dtype=np.float32))
                    active_variants.append((variant_text, variant_weight))
            if query_embeddings:
                import numpy as np

                query_emb = np.stack(query_embeddings)
                query_variant_weights = [weight for _text, weight in active_variants]
                query_variants = active_variants
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
                        norm_q = np.linalg.norm(query_emb, axis=1)

                        # 避免除以 0
                        norm_matrix[norm_matrix == 0] = 1e-9
                        norm_q[norm_q == 0] = 1e-9

                        sims_matrix = np.dot(matrix, query_emb.T) / (norm_matrix[:, None] * norm_q[None, :])
                        variant_weights = np.asarray(query_variant_weights, dtype=np.float32)
                        weighted_sims = sims_matrix * variant_weights[None, :]
                        sims = np.max(weighted_sims, axis=1)

                        # 语义优先：扩大语义召回，再仅用“强整句命中”的文本候选做兜底。
                        semantic_top_k = max(limit * 8, 100)
                        top_indices = np.argsort(sims)[::-1][:semantic_top_k]
                        top_ids = [ids[i] for i in top_indices if sims[i] > 0.03]
                        sim_dict = {ids[i]: float(sims[i]) for i in top_indices}

                        lexical_results_map: dict[int, dict] = {}
                        for variant_text, _variant_weight in query_variants:
                            variant_results = storage.search_fts(variant_text, max(limit * 2, 20))
                            like_results = storage._search_like([variant_text], max(limit * 2, 20))
                            for item in list(variant_results) + list(like_results):
                                item_id = item.get("id")
                                if item_id is None:
                                    continue
                                if "score" not in item:
                                    item["score"] = -0.4
                                existing = lexical_results_map.get(item_id)
                                if existing is None or float(item.get("score", 0.0)) < float(existing.get("score", 0.0)):
                                    lexical_results_map[item_id] = item
                        lexical_results = list(lexical_results_map.values())
                        lexical_score_map = {
                            item["id"]: float(item.get("score", 0.0))
                            for item in lexical_results
                            if item.get("id") is not None
                        }
                        strong_lexical_ids = [
                            item["id"]
                            for item in lexical_results
                            if item.get("id") is not None and _has_strong_text_match(query, item, query_variants)
                        ]

                        merged_ids = []
                        seen_ids = set()
                        for image_id in top_ids + strong_lexical_ids:
                            if image_id not in seen_ids:
                                seen_ids.add(image_id)
                                merged_ids.append(image_id)

                        if merged_ids:
                            db_results = storage.get_images_by_ids(merged_ids)
                            for r in db_results:
                                strong_text_match = _has_strong_text_match(query, r, query_variants)
                                if r["id"] in sim_dict:
                                    semantic_score = -sim_dict[r["id"]]
                                else:
                                    semantic_score = -0.05 if strong_text_match else 0.25
                                lexical_score = lexical_score_map.get(r["id"])
                                text_bonus = _compute_text_match_bonus(query, r, lexical_score, query_variants)
                                concept_bonus = _compute_concept_coverage_bonus(query, r)
                                strong_bonus = -0.35 if strong_text_match else 0.0
                                r["score"] = semantic_score + text_bonus + concept_bonus + strong_bonus
                                r["semantic_score"] = semantic_score
                                r["text_bonus"] = text_bonus
                                r["concept_bonus"] = concept_bonus
                                r["strong_bonus"] = strong_bonus
                                r["db_path"] = db
                            db_results.sort(key=lambda x: x["score"])
                            results.extend(db_results)
            else:
                # 纯本地多级 FTS 检索
                fetch_limit = limit * 2 if rerank else limit
                if local_expand and query_variants:
                    merged_results: dict[int, dict] = {}
                    for variant_text, _variant_weight in query_variants:
                        variant_results = storage.search_fts(variant_text, fetch_limit)
                        for r in variant_results:
                            img_id = r.get("id")
                            if img_id is None:
                                continue
                            existing = merged_results.get(img_id)
                            if existing is None or float(r.get("score", 0.0)) < float(existing.get("score", 0.0)):
                                merged_results[img_id] = r
                    db_results = list(merged_results.values())
                    db_results.sort(key=lambda x: x.get("score", 0.0))
                else:
                    db_results = storage.search_fts(query, fetch_limit)
                for r in db_results:
                    r["db_path"] = db
                results.extend(db_results)
            storage.close()
        except Exception as e:
            logger.warning("在数据库 %s 中搜索失败: %s", db, e)
            print(f"  [WARN] 读取数据库出错 {db}: {e}", file=sys.stderr)
            
    def _dedupe_by_path(items: list[dict]) -> list[dict]:
        import os

        unique = []
        seen = set()
        for item in items:
            path = item.get("path")
            if path:
                norm_path = os.path.normpath(path).lower()
                if norm_path in seen:
                    continue
                seen.add(norm_path)
            unique.append(item)
        return unique

    # 如果有多个库，全局排序
    results.sort(key=lambda x: x.get("score", 0.0))
    results = _dedupe_by_path(results)

    def _count_strong_hits(items: list[dict], variants: list[tuple[str, float]]) -> int:
        return sum(1 for item in items if _has_strong_text_match(base_query, item, variants))

    threshold = min(5, max(1, int(limit / 3)))
    strong_hits = _count_strong_hits(results, query_variants)
    if should_expand_query and len(results) < threshold and strong_hits == 0:
        print("  [INFO] 本地拓展结果不足，触发 LLM 兜底拓展...")
        expanded_query = _llm_expand_query(base_query, api_key, base_url, model)
        query = expanded_query
        expanded_variants = _build_query_variants(expanded_query) if local_expand else [(expanded_query.strip(), 1.0)]

        fallback_results: list[dict] = []
        for db in db_paths:
            try:
                storage = Storage(db)
                if smart:
                    emb_key = embedding_api_key or api_key
                    emb_url = embedding_base_url or base_url
                    query_embeddings = []
                    active_variants: list[tuple[str, float]] = []
                    for variant_text, variant_weight in expanded_variants:
                        query_emb_list = _get_query_embedding(
                            variant_text,
                            emb_key,
                            emb_url,
                            embedding_model,
                            backend=embedding_backend,
                            workers=embedding_workers,
                        )
                        if query_emb_list:
                            import numpy as np

                            query_embeddings.append(np.array(query_emb_list, dtype=np.float32))
                            active_variants.append((variant_text, variant_weight))
                    if query_embeddings:
                        import numpy as np

                        query_emb = np.stack(query_embeddings)
                        query_variant_weights = [weight for _text, weight in active_variants]
                        expanded_variants = active_variants

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
                                norm_q = np.linalg.norm(query_emb, axis=1)
                                norm_matrix[norm_matrix == 0] = 1e-9
                                norm_q[norm_q == 0] = 1e-9
                                sims_matrix = np.dot(matrix, query_emb.T) / (norm_matrix[:, None] * norm_q[None, :])
                                variant_weights = np.asarray(query_variant_weights, dtype=np.float32)
                                weighted_sims = sims_matrix * variant_weights[None, :]
                                sims = np.max(weighted_sims, axis=1)

                                semantic_top_k = max(limit * 8, 100)
                                top_indices = np.argsort(sims)[::-1][:semantic_top_k]
                                top_ids = [ids[i] for i in top_indices if sims[i] > 0.03]
                                sim_dict = {ids[i]: float(sims[i]) for i in top_indices}

                                lexical_results_map: dict[int, dict] = {}
                                for variant_text, _variant_weight in expanded_variants:
                                    variant_results = storage.search_fts(variant_text, max(limit * 2, 20))
                                    like_results = storage._search_like([variant_text], max(limit * 2, 20))
                                    for item in list(variant_results) + list(like_results):
                                        item_id = item.get("id")
                                        if item_id is None:
                                            continue
                                        if "score" not in item:
                                            item["score"] = -0.4
                                        existing = lexical_results_map.get(item_id)
                                        if existing is None or float(item.get("score", 0.0)) < float(existing.get("score", 0.0)):
                                            lexical_results_map[item_id] = item
                                lexical_results = list(lexical_results_map.values())
                                lexical_score_map = {
                                    item["id"]: float(item.get("score", 0.0))
                                    for item in lexical_results
                                    if item.get("id") is not None
                                }
                                strong_lexical_ids = [
                                    item["id"]
                                    for item in lexical_results
                                    if item.get("id") is not None and _has_strong_text_match(expanded_query, item, expanded_variants)
                                ]

                                merged_ids = []
                                seen_ids = set()
                                for image_id in top_ids + strong_lexical_ids:
                                    if image_id not in seen_ids:
                                        seen_ids.add(image_id)
                                        merged_ids.append(image_id)

                                if merged_ids:
                                    db_results = storage.get_images_by_ids(merged_ids)
                                    for r in db_results:
                                        strong_text_match = _has_strong_text_match(expanded_query, r, expanded_variants)
                                        if r["id"] in sim_dict:
                                            semantic_score = -sim_dict[r["id"]]
                                        else:
                                            semantic_score = -0.05 if strong_text_match else 0.25
                                        lexical_score = lexical_score_map.get(r["id"])
                                        text_bonus = _compute_text_match_bonus(expanded_query, r, lexical_score, expanded_variants)
                                        concept_bonus = _compute_concept_coverage_bonus(expanded_query, r)
                                        strong_bonus = -0.35 if strong_text_match else 0.0
                                        r["score"] = semantic_score + text_bonus + concept_bonus + strong_bonus
                                        r["semantic_score"] = semantic_score
                                        r["text_bonus"] = text_bonus
                                        r["concept_bonus"] = concept_bonus
                                        r["strong_bonus"] = strong_bonus
                                        r["db_path"] = db
                                    db_results.sort(key=lambda x: x["score"])
                                    fallback_results.extend(db_results)
                    else:
                        fetch_limit = limit * 2 if rerank else limit
                        merged_results: dict[int, dict] = {}
                        for variant_text, _variant_weight in expanded_variants:
                            variant_results = storage.search_fts(variant_text, fetch_limit)
                            for r in variant_results:
                                img_id = r.get("id")
                                if img_id is None:
                                    continue
                                existing = merged_results.get(img_id)
                                if existing is None or float(r.get("score", 0.0)) < float(existing.get("score", 0.0)):
                                    merged_results[img_id] = r
                        db_results = list(merged_results.values())
                        db_results.sort(key=lambda x: x.get("score", 0.0))
                        for r in db_results:
                            r["db_path"] = db
                        fallback_results.extend(db_results)
                else:
                    fetch_limit = limit * 2 if rerank else limit
                    merged_results: dict[int, dict] = {}
                    for variant_text, _variant_weight in expanded_variants:
                        variant_results = storage.search_fts(variant_text, fetch_limit)
                        for r in variant_results:
                            img_id = r.get("id")
                            if img_id is None:
                                continue
                            existing = merged_results.get(img_id)
                            if existing is None or float(r.get("score", 0.0)) < float(existing.get("score", 0.0)):
                                merged_results[img_id] = r
                    db_results = list(merged_results.values())
                    db_results.sort(key=lambda x: x.get("score", 0.0))
                    for r in db_results:
                        r["db_path"] = db
                    fallback_results.extend(db_results)
                storage.close()
            except Exception as e:
                logger.warning("在数据库 %s 中执行 LLM 兜底拓展失败: %s", db, e)
                print(f"  [WARN] LLM 兜底拓展失败 {db}: {e}", file=sys.stderr)

        fallback_results = _dedupe_by_path(fallback_results)
        existing_paths = {item.get("path") for item in results if item.get("path")}
        for item in fallback_results:
            if item.get("path") in existing_paths:
                continue
            results.append(item)

    fetch_limit = limit * 2 if rerank else limit
    results = results[:fetch_limit]

    if rerank and results:
        results, rerank_warn = _llm_rerank_results(base_query, results, api_key, base_url, model)
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
