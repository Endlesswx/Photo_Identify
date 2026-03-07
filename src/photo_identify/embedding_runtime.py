"""本文件用于统一封装 embedding 模型的本地推理与远程 API 调用。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MAX_LENGTH = 1024

try:
    import torch
except ImportError:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


_LOCAL_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_LOCAL_MODEL_CACHE_LOCK = threading.Lock()


def normalize_embedding_backend(backend: str, base_url: str = "") -> str:
    """将 embedding 后端标识标准化为 ``local`` 或 ``api``。"""

    normalized = backend.strip().lower()
    if normalized in {"local", "api"}:
        return normalized
    return "local" if not base_url.strip() else "api"


def ensure_local_embedding_runtime() -> None:
    """确保本地 embedding 推理所需的依赖已经安装。"""

    if SentenceTransformer is None:
        raise RuntimeError("未安装 sentence-transformers，无法运行本地 embedding 模型。")
    if torch is None:
        raise RuntimeError("未安装 torch，无法运行本地 embedding 模型。")


def resolve_local_embedding_device(device_arg: str = "auto") -> str:
    """解析本地 embedding 模型的运行设备。"""

    normalized = device_arg.strip().lower()
    if normalized == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        torch_version = getattr(torch, "__version__", "未安装") if torch is not None else "未安装"
        logger.warning("未检测到可用 CUDA，将回退到 CPU。当前 torch 版本: %s", torch_version)
        return "cpu"

    if normalized == "cpu":
        return "cpu"

    if normalized == "cuda":
        normalized = "cuda:0"

    if normalized.startswith("cuda"):
        if torch is None:
            raise RuntimeError("当前环境未安装 torch，无法使用 CUDA 运行本地 embedding 模型。")
        if not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 未检测到可用的 CUDA 设备。")
        return normalized

    raise ValueError(f"不支持的本地 embedding 设备: {device_arg}")


def describe_local_embedding_device(device: str) -> str:
    """返回更适合日志输出的本地 embedding 设备描述。"""

    if torch is None or not device.startswith("cuda"):
        return device

    try:
        device_index = torch.device(device).index or 0
        device_name = torch.cuda.get_device_name(device_index)
    except Exception:
        return device
    return f"{device} ({device_name})"


def get_local_embedding_model(model_id: str, device: str = "auto", model_cache_dir: Path | None = None) -> tuple[object, str]:
    """加载并缓存本地 embedding 模型实例。"""

    ensure_local_embedding_runtime()
    resolved_device = resolve_local_embedding_device(device)
    cache_dir = str(model_cache_dir.resolve()) if model_cache_dir is not None else ""
    cache_key = (model_id, resolved_device, cache_dir)

    with _LOCAL_MODEL_CACHE_LOCK:
        cached_model = _LOCAL_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model, resolved_device

        logger.info(
            "加载本地 embedding 模型: %s，设备=%s",
            model_id,
            describe_local_embedding_device(resolved_device),
        )
        model = SentenceTransformer(
            model_id,
            device=resolved_device,
            cache_folder=cache_dir or None,
        )
        if resolved_device.startswith("cuda"):
            model.half()
        _LOCAL_MODEL_CACHE[cache_key] = model
        return model, resolved_device


def encode_texts_locally(
    texts: list[str],
    model_id: str,
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = DEFAULT_LOCAL_MAX_LENGTH,
    model_cache_dir: Path | None = None,
) -> np.ndarray:
    """使用本地 embedding 模型批量编码文本。"""

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model, _ = get_local_embedding_model(model_id=model_id, device=device, model_cache_dir=model_cache_dir)
    model.max_seq_length = max_length
    embeddings = model.encode(
        texts,
        batch_size=max(1, batch_size),
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def request_remote_embedding_sync(
    text: str,
    model: str,
    base_url: str,
    api_key: str = "",
    timeout: int = 15,
) -> list[float]:
    """通过同步 HTTP 请求调用远程 embedding 接口。"""

    if not base_url.strip():
        raise ValueError("API embedding 模式必须提供 base_url。")

    payload = {
        "model": model,
        "input": text,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{base_url}/embeddings" if not base_url.endswith("/embeddings") else base_url
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    embedding = data.get("data", [{}])[0].get("embedding")
    if embedding and isinstance(embedding, list):
        return [float(item) for item in embedding]
    raise RuntimeError(f"远程 embedding 接口返回异常: {data}")


def get_text_embedding_sync(
    text: str,
    model: str,
    backend: str = "",
    api_key: str = "",
    base_url: str = "",
    timeout: int = 15,
    device: str = "auto",
    workers: int = 1,
    max_length: int = DEFAULT_LOCAL_MAX_LENGTH,
    model_cache_dir: Path | None = None,
) -> list[float]:
    """按指定后端同步获取单条文本的 embedding 向量。"""

    normalized_backend = normalize_embedding_backend(backend, base_url)
    if normalized_backend == "local":
        embeddings = encode_texts_locally(
            texts=[text],
            model_id=model,
            device=device,
            batch_size=max(1, workers),
            max_length=max_length,
            model_cache_dir=model_cache_dir,
        )
        return embeddings[0].astype(np.float32).tolist()

    return request_remote_embedding_sync(
        text=text,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


async def get_text_embedding_async(
    text: str,
    model: str,
    backend: str,
    session,
    api_key: str = "",
    base_url: str = "",
    timeout: int = 15,
    rate_limiter=None,
    device: str = "auto",
    workers: int = 1,
    max_length: int = DEFAULT_LOCAL_MAX_LENGTH,
    model_cache_dir: Path | None = None,
) -> list[float]:
    """按指定后端异步获取单条文本的 embedding 向量。"""

    normalized_backend = normalize_embedding_backend(backend, base_url)
    if normalized_backend == "local":
        return await asyncio.to_thread(
            get_text_embedding_sync,
            text,
            model,
            "local",
            api_key,
            base_url,
            timeout,
            device,
            workers,
            max_length,
            model_cache_dir,
        )

    from photo_identify.llm import async_call_embedding_model

    embedding = await async_call_embedding_model(
        text=text,
        session=session,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        rate_limiter=rate_limiter,
    )
    return embedding or []
