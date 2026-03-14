"""PyInstaller 运行时兼容层。

提供统一的资源路径解析、外部工具路径注入、字体 fallback 等功能，
确保在开发环境与 PyInstaller onedir 打包后均能正常工作。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# PyInstaller 打包后会设置 sys._MEIPASS 指向临时解压目录
# 开发环境下该属性不存在，此时使用当前文件所在目录
if getattr(sys, "frozen", False):
    # PyInstaller onedir 模式：exe 所在目录
    _BUNDLE_DIR = Path(sys.executable).parent.resolve()
    # PyInstaller 解压的临时资源目录
    _RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", _BUNDLE_DIR)).resolve()
else:
    # 开发环境：使用源码目录
    _BUNDLE_DIR = Path(__file__).resolve().parent.parent.parent
    _RESOURCE_DIR = _BUNDLE_DIR / "src" / "photo_identify"


def get_bundle_dir() -> Path:
    """返回可执行文件所在目录（打包后）或项目根目录（开发环境）。"""
    return _BUNDLE_DIR


def get_resource_dir() -> Path:
    """返回资源目录（PyInstaller 解压目录或源码目录）。"""
    return _RESOURCE_DIR


def get_bin_dir() -> Path:
    """返回外部工具目录（bin/），用于存放 ffmpeg/ffprobe 等。"""
    return _BUNDLE_DIR / "bin"


def get_default_data_dir() -> Path:
    """返回默认数据目录（数据库、缓存等）。

    打包后：exe 所在目录下的 data/ 子目录
    开发环境：项目根目录
    """
    if getattr(sys, "frozen", False):
        data_dir = _BUNDLE_DIR / "data"
    else:
        data_dir = _BUNDLE_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_default_db_path() -> Path:
    """返回默认数据库路径。"""
    return get_default_data_dir() / "photo_identify.db"


def get_default_cache_dir() -> Path:
    """返回默认缓存目录。"""
    return get_default_data_dir() / "cache"


def inject_bin_to_path() -> None:
    """将 bin/ 目录注入到 PATH 环境变量（如果存在且尚未注入）。

    用于确保 ffmpeg/ffprobe 等外部工具可被直接调用。
    """
    bin_dir = get_bin_dir()
    if not bin_dir.exists():
        return

    bin_dir_str = str(bin_dir)
    current_path = os.environ.get("PATH", "")

    # 检查是否已注入（避免重复注入）
    path_parts = current_path.split(os.pathsep)
    if bin_dir_str in path_parts:
        return

    # 将 bin/ 目录添加到 PATH 最前面，优先使用随包版本
    os.environ["PATH"] = bin_dir_str + os.pathsep + current_path


def get_ffmpeg_path() -> Path | None:
    """返回 ffmpeg 可执行文件路径（如果存在于 bin/ 目录）。"""
    ffmpeg = get_bin_dir() / "ffmpeg.exe"
    return ffmpeg if ffmpeg.exists() else None


def get_ffprobe_path() -> Path | None:
    """返回 ffprobe 可执行文件路径（如果存在于 bin/ 目录）。"""
    ffprobe = get_bin_dir() / "ffprobe.exe"
    return ffprobe if ffprobe.exists() else None


def get_font_path(font_name: str) -> Path | None:
    """尝试定位字体文件路径。

    优先级：
    1. bin/fonts/ 目录下的字体文件
    2. Windows 系统字体目录

    Args:
        font_name: 字体文件名，如 "msyh.ttc"、"seguisym.ttf"

    Returns:
        字体文件路径，找不到时返回 None
    """
    # 1. 检查随包字体目录
    bundled_font = get_bin_dir() / "fonts" / font_name
    if bundled_font.exists():
        return bundled_font

    # 2. 检查 Windows 系统字体目录
    if sys.platform == "win32":
        system_font = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts" / font_name
        if system_font.exists():
            return system_font

    return None


def get_bundled_script_path(script_relative: str) -> Path | None:
    """返回源码脚本路径。

    仅在开发环境或保留源码分发时可用。
    打包后的主 GUI exe 不应依赖该方式执行子任务；
    优先使用独立 helper exe。
    """
    if getattr(sys, "frozen", False):
        script_path = _RESOURCE_DIR / script_relative
    else:
        script_path = _BUNDLE_DIR / "src" / script_relative
    return script_path if script_path.exists() else None


def get_helper_executable(executable_name: str) -> Path | None:
    """返回随包 helper 可执行文件路径。

    支持两种布局：
    1. 与主程序同级：`photo_identify.exe` 同目录下直接放置 helper exe
    2. 子目录布局：`video_compression/video_compression.exe`
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    candidates = [
        get_bundle_dir() / f"{executable_name}{suffix}",
        get_bundle_dir() / "tools" / f"{executable_name}{suffix}",
        get_bundle_dir() / executable_name / f"{executable_name}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# 模块加载时自动注入 PATH
inject_bin_to_path()
