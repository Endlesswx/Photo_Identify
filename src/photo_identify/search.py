"""自然语言搜索模块：支持 LIKE 模糊搜索和可选的 LLM 辅助语义扩展。

提供两种搜索策略：
1. 普通搜索（默认）：将用户输入拆分为关键词，在 scene/objects/style 等字段中 LIKE 匹配
2. 智能搜索（--smart 模式）：先用 LLM 将口语化查询转换为同义词/近义词关键词组，再搜索
"""

import json
import logging
import sys
import urllib.error
import urllib.request

from photo_identify.config import DEFAULT_BASE_URL, DEFAULT_SEARCH_LIMIT, DEFAULT_TEXT_MODEL
from photo_identify.storage import Storage

logger = logging.getLogger(__name__)

# 智能搜索的 LLM 提示词模板，要求生成同义词/近义词扩展
_EXPAND_SYSTEM_PROMPT = """\
你是一个图片搜索关键词扩展助手。用户会用口语描述想找的图片，你的任务是：

1. 提取核心搜索意图中的关键词
2. 为每个关键词生成同义词、近义词、相关词（中文）
3. 考虑图片描述中可能使用的专业/书面表达方式

示例：
  输入: "美女在唱歌"
  输出: 美女 女性 女孩 女人 唱歌 歌手 演唱 录音 麦克风 音乐

  输入: "有红灯笼的那张过年图片"
  输出: 红灯笼 灯笼 红色 过年 春节 新年 节日 喜庆

  输入: "海边日落"
  输出: 海边 海滨 海岸 沙滩 日落 夕阳 黄昏 傍晚

规则：
- 只输出关键词，用空格分隔，不要其他内容
- 关键词数量 5-15 个
- 优先生成与图片视觉内容相关的词汇"""


def _llm_expand_query(query: str, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_TEXT_MODEL) -> str:
    """使用 LLM 将口语化查询扩展为包含同义词/近义词的关键词组。

    例如 "美女在唱歌" → "美女 女性 女孩 唱歌 歌手 演唱 录音 麦克风 音乐"

    Args:
        query: 用户的口语化搜索输入。
        api_key: API Key。
        base_url: API Base URL。
        model: 选择的大语言模型

    Returns:
        扩展后的关键词字符串，失败时返回原始查询。
    """
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 150,
        "messages": [
            {"role": "system", "content": _EXPAND_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        expanded = content.strip()
        if expanded:
            print(f"  🔍 关键词扩展: \"{query}\" → \"{expanded}\"")
            return expanded
    except Exception as exc:
        logger.warning("LLM 关键词扩展失败，回退为原始查询: %s", exc)
        print(f"  ⚠️ LLM 扩展失败，使用原始查询: {exc}", file=sys.stderr)
    return query


_RERANK_SYSTEM_PROMPT = """\
你是一个图片检索重排序助手。用户提供了一个搜索描述，以及若干候选图片的序号和内容信息。
你的任务是：深入理解用户的语言意图，从候选图片中选出相对最符合要求的图片，并按匹配程度从高到低排序返回。

返回规则严格遵守：
直接输出选中的候选图片序号的 JSON 数组，例如：
[3, 1, 5]
如果没有符合要求的，返回 []。不要输出任何除了方括号和数字之外的内容（不要加 ```json 标记）。"""

def _llm_rerank_results(query: str, results: list[dict], api_key: str, base_url: str, model: str) -> list[dict]:
    if not results:
        return []

    # 限制 LLM 处理的数量，防止上下文超长导致响应超时
    max_rerank = 20
    to_rerank = results[:max_rerank]
    remainder = results[max_rerank:]

    # 构造 candidates 字符串
    candidates_text = []
    # 建立 id 映射
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
        "max_tokens": 200,
        "messages": [
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions", data=body, headers=headers, method="POST"
    )
    
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        ans = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        ans = ans.replace("```json", "").replace("```", "").strip()
        
        # 尝试解析 JSON 数组
        best_indices = json.loads(ans)
        if not isinstance(best_indices, list):
            raise ValueError("结果不是数组")
        
        reranked = []
        seen = set()
        for idx in best_indices:
            try:
                # 可能是字符串数字
                idx_int = int(idx)
                if idx_int in id_map and idx_int not in seen:
                    reranked.append(id_map[idx_int])
                    seen.add(idx_int) # 保证不重复
            except ValueError:
                continue
                
        print(f"  🧠 LLM 排序原始返回: {ans}, 匹配出 {len(reranked)} 项")
        
        # 将在 max_rerank 范围内但未被 LLM 显式返回的其他图片追加到后面
        for i, r in enumerate(to_rerank, 1):
            if i not in seen:
                reranked.append(r)
                
        # 拼接剩余未参与重排的项
        return reranked + remainder

    except Exception as exc:
        logger.warning("LLM 重排序失败: %s", exc)
        print(f"  ⚠️ LLM 排序失败，回退默认顺序: {exc}", file=sys.stderr)
        return results

def search(
    query: str,
    db_paths: str | list[str],
    limit: int = DEFAULT_SEARCH_LIMIT,
    smart: bool = False,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_TEXT_MODEL,
    rerank: bool = False,
) -> list[dict]:
    """执行图片搜索并返回匹配结果。

    Args:
        query: 搜索文本（口语化查询或关键词）。
        db_paths: SQLite 数据库文件路径（单个字符串或字符串列表）。
        limit: 最大返回条数。
        smart: 是否启用 LLM 辅助关键词扩展。
        api_key: API Key（smart 或 rerank 模式需要）。
        base_url: API Base URL。
        model: 模型名称。
        rerank: 是否使用 LLM 对召回结果进行二次排序。

    Returns:
        匹配的图片记录列表。
    """
    expanded_query = query
    if smart or rerank:
        if not api_key:
            print("  ⚠️ 智能模式/重排序需要 API Key，回退为普通搜索模式", file=sys.stderr)
            smart = False
            rerank = False
        elif smart:
            expanded_query = _llm_expand_query(query, api_key, base_url)

    if isinstance(db_paths, str):
        db_paths = [db_paths]
        
    results = []
    # 若开启 rerank，扩大召回数量供大模型选择
    fetch_limit = limit * 2 if rerank else limit
    
    for db in db_paths:
        try:
            storage = Storage(db)
            db_results = storage.search_fts(expanded_query, fetch_limit)
            results.extend(db_results)
            storage.close()
        except Exception as e:
            logger.warning("在数据库 %s 中搜索失败: %s", db, e)
            print(f"  ⚠️ 读取数据库出错 {db}: {e}", file=sys.stderr)
            
    # 如果跨库汇总结果超出 fetch_limit，按需对结果的分数等进行全局重排，这里简单截断前 fetch_limit 个
    results = results[:fetch_limit]
    
    if rerank and results:
        results = _llm_rerank_results(query, results, api_key, base_url, model)
        # 截断结果到要求数量
        results = results[:limit]

    return results


def format_results(results: list[dict]) -> str:
    """将搜索结果格式化为可读文本。

    Args:
        results: 图片记录字典列表。

    Returns:
        格式化后的多行文本。
    """
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
