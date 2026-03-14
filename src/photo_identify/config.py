"""全局配置常量与默认值。

集中管理所有可调参数，避免硬编码散落在各模块中。
"""

import os
from pathlib import Path

# ── 运行时兼容（支持 PyInstaller 打包）──────────────────────
from photo_identify.runtime_compat import get_default_db_path as _get_default_db_path

# ── API 配置 ──────────────────────────────────────────────
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

# 支持的模型
# 默认图片分析模型（主要用于 scan 扫描看图分析）
DEFAULT_IMAGE_MODEL = "THUDM/GLM-4.1V-9B-Thinking"
# 默认文本处理模型（主要用于 search / rerank 重排序与同义词扩展）
DEFAULT_TEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 向量模型（用于 Semantic Search 语义检索）
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

# 默认采样参数
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 300

# ── 速率限制 ──────────────────────────────────────────────
DEFAULT_RPM_LIMIT = 1000
DEFAULT_TPM_LIMIT = 50000

# ── 图片扫描 ──────────────────────────────────────────────
DEFAULT_IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif",
    ".heic", ".heif",
    ".livp", ".mp4", ".mov", ".avi", ".mkv",
})
# 上传前缩放的最长边像素数
MAX_UPLOAD_DIMENSION = 1280
# 上传时转换为 JPEG 的质量
UPLOAD_JPEG_QUALITY = 85

# ── 并发 ──────────────────────────────────────────────────
DEFAULT_WORKERS = 4

# ── 重试 ──────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2  # 秒，指数退避基数

# ── 存储 ──────────────────────────────────────────────────
# 使用 runtime_compat 获取默认数据库路径，支持打包后路径
DEFAULT_DB_PATH = _get_default_db_path()

# ── 搜索 ──────────────────────────────────────────────────
DEFAULT_SEARCH_LIMIT = 20

# ── 浏览模式 ──────────────────────────────────────────────
DEFAULT_BROWSE_PAGE_SIZE = 48


def load_api_key(explicit_key: str = "") -> str:
    """读取 API Key，优先使用显式传入的值，其次读取环境变量。

    Args:
        explicit_key: 命令行显式传入的 key，为空时走环境变量。

    Returns:
        API Key 字符串，找不到时返回空字符串。
    """
    if explicit_key:
        return explicit_key
    return (
        os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("LLM_API_KEY_GPSQA")
        or ""
    )
