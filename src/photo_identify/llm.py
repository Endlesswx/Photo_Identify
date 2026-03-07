"""LLM API 调用层：负责图片/视频分析模型调用、速率控制和自动重试。

封装与 SiliconFlow API 的通信逻辑，包括：
- base64 编码图片构建请求载荷
- RPM/TPM 双维度节流
- 失败自动重试（指数退避）
- JSON 返回值解析与校验
"""

import asyncio
import base64
import json
import logging
import re
import time
import aiohttp

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)


def _build_prompt() -> str:
    """生成图片/视频分析的中文提示词。

    Returns:
        用于图片/视频分析模型的中文提示词。
    """
    return (
        "请用中文分析提供的媒体（图片或视频），直接输出纯 JSON（不要用 markdown 代码块包裹），包含字段："
        "scene（场景描述）、objects（主要物体列表，JSON 数组）、style（风格与色调）、"
        "location_time（可能的地点/时间）、wallpaper_hint（壁纸建议）。"
    )


def _build_payload(model: str, temperature: float, max_tokens: int, prompt: str, image_b64: str, image_format: str) -> dict:
    """构建 chat-completions 请求载荷。"""
    
    if image_format == "mp4":
        media_url = f"data:video/mp4;base64,{image_b64}"
        media_item = {"type": "video_url", "video_url": {"url": media_url}}
    else:
        media_url = f"data:image/{image_format};base64,{image_b64}"
        media_item = {"type": "image_url", "image_url": {"url": media_url}}

    messages = []
    messages.append({"role": "system", "content": "你是媒体分析助手。"})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            media_item,
        ],
    })

    return {
        "model": model,
        "temperature": 0.0 if image_format == "mp4" else temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }


async def _request_json_async(url: str, payload: dict, headers: dict, timeout: int, session: aiohttp.ClientSession) -> tuple[int, dict]:
    """发送 JSON POST 异步请求并返回状态码与响应内容。

    Args:
        url: 请求地址。
        payload: JSON 请求载荷。
        headers: 请求头字典。
        timeout: 超时时间（秒）。
        session: aiohttp 会话实例。

    Returns:
        (HTTP 状态码, 解析后的响应字典)。
    """
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            status = response.status
            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                raw = await response.text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}
            return status, data
    except Exception as exc:
        return 500, {"raw": str(exc), "error": str(exc)}


def _estimate_tokens(prompt: str, max_tokens: int) -> int:
    """粗略估算本次请求的 token 成本（用于 TPM 节流）。

    Args:
        prompt: 提示词内容。
        max_tokens: 允许的最大输出 token。

    Returns:
        估算的总 token 数。
    """
    prompt_tokens = max(1, int(len(prompt) / 2))
    return prompt_tokens + max_tokens


class RateLimiter:
    """RPM/TPM 双维度速率限制器（线程安全）。"""

    def __init__(self, rpm_limit: int, tpm_limit: int):
        """初始化速率限制器。

        Args:
            rpm_limit: 每分钟最大请求数。
            tpm_limit: 每分钟最大 token 数。
        """
        import threading
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._requests: list[float] = []
        self._tokens: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def wait(self, token_cost: int):
        """根据速率限制等待适当时间后放行（协程安全）。

        Args:
            token_cost: 本次请求估算的 token 成本。
        """
        sleep_time = 0.0
        async with self._lock:
            now = time.monotonic()
            self._requests = [t for t in self._requests if now - t < 60]
            self._tokens = [t for t in self._tokens if now - t[0] < 60]

            if self._rpm_limit > 0 and len(self._requests) >= self._rpm_limit:
                wait_seconds = 60 - (now - self._requests[0])
                if wait_seconds > 0:
                    sleep_time = max(sleep_time, wait_seconds)

            if self._tpm_limit > 0:
                total_tokens = sum(item[1] for item in self._tokens)
                if total_tokens + token_cost > self._tpm_limit and self._tokens:
                    wait_seconds = 60 - (now - self._tokens[0][0])
                    if wait_seconds > 0:
                        sleep_time = max(sleep_time, wait_seconds)

        if sleep_time > 0:
            logger.debug("速率节流，等待 %.1f 秒", sleep_time)
            await asyncio.sleep(sleep_time)

        async with self._lock:
            now = time.monotonic()
            self._requests.append(now)
            self._tokens.append((now, token_cost))


def _extract_json_from_text(text: str) -> dict | None:
    """尝试从可能包含 markdown 代码块的文本中提取 JSON 对象。

    Args:
        text: 模型返回的原始文本。

    Returns:
        解析成功的字典，失败返回 None。
    """
    # 先尝试直接解析
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # 尝试从 ```json ... ``` 代码块中提取
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 尝试找第一个 { ... } 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


class CircuitBreaker:
    """全局断路器，用于探测并熔断长时间无响应的 API 请求假死。"""
    def __init__(self, max_consecutive_failures: int = 5):
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self):
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.is_open = True
                
    def record_success(self):
        self.consecutive_failures = 0
        self.is_open = False

# 全局共享断路器实例
global_circuit_breaker = CircuitBreaker()


async def async_call_image_model(
    image_bytes: bytes,
    session: aiohttp.ClientSession,
    image_format: str = "jpeg",
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_IMAGE_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    """调用图片/视频分析模型分析媒体，包含自动重试和 JSON 解析（异步版本）。

    Args:
        image_bytes: 压缩后的图片字节内容。
        session: aiohttp ClientSession
        image_format: 图片格式（如 jpeg）。
        api_key: API Key。
        base_url: API Base URL。
        model: 模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        timeout: HTTP 超时秒数。
        rate_limiter: 速率限制器实例。

    Returns:
        包含 scene/objects/style/location_time/wallpaper_hint/llm_raw 的字典；
        失败时包含 error 字段。
    """
    if global_circuit_breaker.is_open:
        return {"error": "断路器已触发，API 拒绝服务或持续超时，请检查网络或稍后再试。", "__circuit_breaker_open__": True}

    prompt = _build_prompt()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = _build_payload(model, temperature, max_tokens, prompt, image_b64, image_format)
    headers = {"Content-Type": "application/json"}
    if api_key:  # 本地模型（如 Ollama）无需 Authorization 头
        headers["Authorization"] = f"Bearer {api_key}"
    token_cost = _estimate_tokens(prompt, max_tokens)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if rate_limiter:
            await rate_limiter.wait(token_cost)
            
        # 第一次请求且是视频时，加入随机抖动错开大文件传输和解码峰值
        if attempt == 1 and image_format == "mp4":
            import random
            await asyncio.sleep(random.uniform(1.0, 4.0))
            
        current_timeout = timeout if image_format != "mp4" else max(timeout, 300)
            
        try:
            status, data = await _request_json_async(f"{base_url}/chat/completions", payload, headers, current_timeout, session)
            # 成功请求则重置断路器
            if status < 500:
                global_circuit_breaker.record_success()
                
        except Exception as exc:
            last_error = str(exc)
            logger.warning("API 请求异常 (第 %d 次): %s", attempt, exc)
            
            # 对"完全读不出"或超时这类网络连通性异常触发快速失效并记录断路器
            if "timed out" in str(exc).lower() or "connection" in str(exc).lower() or "timeout" in str(exc).lower():
                global_circuit_breaker.record_failure()
                if global_circuit_breaker.is_open:
                    logger.error("连续网络超时达到阈值，触发断路器熔断！")
                    return {"error": f"触发断路器：网络持续无法连接或超时 ({last_error})", "__circuit_breaker_open__": True}
            
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(RETRY_BASE_DELAY ** attempt, 3.0))
            continue

        if status >= 500:
            last_error = f"HTTP {status}: {data}"
            logger.warning("服务端错误 (第 %d 次): %s", attempt, last_error)
            global_circuit_breaker.record_failure()
            if global_circuit_breaker.is_open:
                logger.error("连续服务端报错 (>=500) 达到熔断阈值，触发断路器熔断！")
                return {"error": f"触发断路器：服务端持续返回错误 ({last_error})", "__circuit_breaker_open__": True}
                    
            if attempt < MAX_RETRIES:
                import random
                # 对于 500 错误采用指数退避并增加明显的随机抖动，防止重试风暴打垮服务端
                backoff = min(RETRY_BASE_DELAY ** (attempt + 1), 15.0) + random.uniform(1.0, 3.0)
                logger.debug("触发 5xx 错误降级策略，等待 %.1f 秒后重试", backoff)
                await asyncio.sleep(backoff)
            continue

        if status >= 400:
            # 某些后端（如 vLLM）当视频并发解码时资源耗尽会返回 "Could not open video stream" 的 HTTP 400
            # 这种情况视为服务端临时资源瓶颈，允许进行降级重试
            err_msg = str(data).lower()
            if "could not open video stream" in err_msg or "too many requests" in err_msg:
                last_error = f"HTTP {status}: {data}"
                logger.warning(">>>>>>>> [系统拦截] 服务端解码队列已满，触发降级重试 (第 %d 次): %s", attempt, last_error)
                if attempt < MAX_RETRIES:
                    import random
                    backoff = min(RETRY_BASE_DELAY ** (attempt + 1), 15.0) + random.uniform(2.0, 5.0)
                    logger.warning(">>>>>>>> 等待 %.1f 秒后重新投递...", backoff)
                    await asyncio.sleep(backoff)
                    continue

            return {"error": f"HTTP {status}: {data}", "llm_raw": json.dumps(data, ensure_ascii=False)}

        choices = data.get("choices") or []
        if not choices:
            return {"error": "API 返回空 choices", "llm_raw": json.dumps(data, ensure_ascii=False)}

        content = choices[0].get("message", {}).get("content", "")
        raw_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

        parsed = _extract_json_from_text(raw_text)

        def _to_str(val):
            if isinstance(val, list):
                return ", ".join(map(str, val))
            return str(val or "")

        if parsed and isinstance(parsed, dict):
            return {
                "scene": _to_str(parsed.get("scene")),
                "objects": parsed.get("objects") if isinstance(parsed.get("objects"), list) else [],
                "style": _to_str(parsed.get("style")),
                "location_time": _to_str(parsed.get("location_time")),
                "wallpaper_hint": _to_str(parsed.get("wallpaper_hint")),
                "llm_raw": raw_text,
            }
        # JSON 解析失败，存 raw
        return {
            "scene": "",
            "objects": [],
            "style": "",
            "location_time": "",
            "wallpaper_hint": "",
            "llm_raw": raw_text,
            "parse_failed": True,
        }

    return {"error": f"重试 {MAX_RETRIES} 次后仍失败: {last_error}", "llm_raw": ""}


async def async_call_embedding_model(
    text: str,
    session: aiohttp.ClientSession,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = "BAAI/bge-m3",
    timeout: int = DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> bytes | None:
    """调用 Embedding 模型将文本转化为向量（异步版本）。

    Args:
        text: 需要向量化的文本。
        session: aiohttp ClientSession
        api_key: API Key。
        base_url: API Base URL。
        model: Embedding 模型名称。
        timeout: HTTP 超时秒数。
        rate_limiter: 速率限制器实例。

    Returns:
        如果成功，返回序列化后的 float32 向量（bytes，可直接存入 SQLite BLOB）；
        失败返回 None。
    """
    if not text.strip():
        return None

    payload = {
        "model": model,
        "input": text,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 简单估算 token cost
    token_cost = max(1, len(text) // 2)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if rate_limiter:
            await rate_limiter.wait(token_cost)

        try:
            status, data = await _request_json_async(
                f"{base_url}/embeddings", payload, headers, timeout, session
            )
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Embedding API 请求异常 (第 %d 次): %s", attempt, exc)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(RETRY_BASE_DELAY ** attempt, 3.0))
            continue

        if status >= 500:
            last_error = f"HTTP {status}: {data}"
            logger.warning("Embedding 服务端错误 (第 %d 次): %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                import random
                await asyncio.sleep(min(RETRY_BASE_DELAY ** (attempt + 1), 15.0) + random.uniform(1.0, 3.0))
            continue

        if status >= 400:
            logger.error("Embedding 客户端错误: HTTP %d %s", status, data)
            return None

        emb_data = data.get("data")
        if not emb_data or not isinstance(emb_data, list):
            logger.error("Embedding API 返回格式异常: %s", data)
            return None

        embedding_list = emb_data[0].get("embedding")
        if not embedding_list or not isinstance(embedding_list, list):
            logger.error("未能从返回体中提取出 embedding 数组")
            return None

        # 将 list 转换为 bytes (float32 数组)
        import struct
        try:
            return struct.pack(f"<{len(embedding_list)}f", *embedding_list)
        except Exception as e:
            logger.error("序列化 Embedding 失败: %s", e)
            return None

    logger.error("Embedding 调用失败: %s", last_error)
    return None

