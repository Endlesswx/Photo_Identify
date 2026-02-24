"""LLM API 调用层：负责多模态模型调用、速率控制和自动重试。

封装与 SiliconFlow API 的通信逻辑，包括：
- base64 编码图片构建请求载荷
- RPM/TPM 双维度节流
- 失败自动重试（指数退避）
- JSON 返回值解析与校验
"""

import base64
import json
import logging
import re
import time
import urllib.error
import urllib.request

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_VISION_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)


def _build_prompt() -> str:
    """生成图片分析的中文提示词。

    Returns:
        用于多模态模型的中文提示词。
    """
    return (
        "请用中文分析图片，直接输出纯 JSON（不要用 markdown 代码块包裹），包含字段："
        "scene（场景描述）、objects（主要物体列表，JSON 数组）、style（风格与色调）、"
        "location_time（可能的地点/时间）、wallpaper_hint（壁纸建议）。"
    )


def _build_payload(model: str, temperature: float, max_tokens: int, prompt: str, image_b64: str, image_format: str) -> dict:
    """构建 chat-completions 请求载荷。

    Args:
        model: 模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        prompt: 图片分析提示词。
        image_b64: 图片 base64 编码字符串。
        image_format: 图片格式（如 jpeg/png）。

    Returns:
        API 请求载荷字典。
    """
    image_url = f"data:image/{image_format};base64,{image_b64}"
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "你是图片分析助手。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }


def _request_json(url: str, payload: dict, headers: dict, timeout: int) -> tuple[int, dict]:
    """发送 JSON POST 请求并返回状态码与响应内容。

    Args:
        url: 请求地址。
        payload: JSON 请求载荷。
        headers: 请求头字典。
        timeout: 超时时间（秒）。

    Returns:
        (HTTP 状态码, 解析后的响应字典)。
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        data = json.loads(raw.decode("utf-8")) if raw else {}
    except json.JSONDecodeError:
        data = {"raw": raw.decode("utf-8", errors="ignore")}
    return status, data


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
        self._lock = threading.Lock()

    def wait(self, token_cost: int):
        """根据速率限制等待适当时间后放行（线程安全）。

        Args:
            token_cost: 本次请求估算的 token 成本。
        """
        # 第一步：在锁内计算需要等待的时间，但不在锁内 sleep
        sleep_time = 0.0
        with self._lock:
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

        # 第二步：在锁外 sleep，不阻塞其他线程
        if sleep_time > 0:
            logger.debug("速率节流，等待 %.1f 秒", sleep_time)
            time.sleep(sleep_time)

        # 第三步：重新获取锁，记录本次请求
        with self._lock:
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


def call_image_model(
    image_bytes: bytes,
    image_format: str = "jpeg",
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    rate_limiter: RateLimiter | None = None,
) -> dict:
    """调用多模态模型分析图片，包含自动重试和 JSON 解析。

    Args:
        image_bytes: 压缩后的图片字节内容。
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
    prompt = _build_prompt()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = _build_payload(model, temperature, max_tokens, prompt, image_b64, image_format)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    token_cost = _estimate_tokens(prompt, max_tokens)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if rate_limiter:
            rate_limiter.wait(token_cost)
        try:
            status, data = _request_json(f"{base_url}/chat/completions", payload, headers, timeout)
        except Exception as exc:
            last_error = str(exc)
            logger.warning("API 请求异常 (第 %d 次): %s", attempt, exc)
            if attempt < MAX_RETRIES:
                time.sleep(min(RETRY_BASE_DELAY ** attempt, 3.0))
            continue

        if status >= 500:
            last_error = f"HTTP {status}: {data}"
            logger.warning("服务端错误 (第 %d 次): %s", attempt, last_error)
            if attempt < MAX_RETRIES:
                time.sleep(min(RETRY_BASE_DELAY ** attempt, 3.0))
            continue

        if status >= 400:
            return {"error": f"HTTP {status}: {data}", "llm_raw": json.dumps(data, ensure_ascii=False)}

        choices = data.get("choices") or []
        if not choices:
            return {"error": "API 返回空 choices", "llm_raw": json.dumps(data, ensure_ascii=False)}

        content = choices[0].get("message", {}).get("content", "")
        raw_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)

        parsed = _extract_json_from_text(raw_text)
        if parsed and isinstance(parsed, dict):
            return {
                "scene": parsed.get("scene", ""),
                "objects": parsed.get("objects", []),
                "style": parsed.get("style", ""),
                "location_time": parsed.get("location_time", ""),
                "wallpaper_hint": parsed.get("wallpaper_hint", ""),
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
