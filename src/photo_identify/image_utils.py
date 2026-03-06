"""图片工具函数：读取、压缩、元数据提取和哈希计算。

负责所有与图片文件本身相关的操作，不涉及 LLM 或存储逻辑。
"""

import hashlib
import io
import os
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import io
import logging
import pillow_heif
from PIL import ExifTags, Image, ImageFile

# 注册支持 HEIC/HEIF 等 Apple 格式的解码器到原生 Pillow 系统中
pillow_heif.register_heif_opener()

ImageFile.LOAD_TRUNCATED_IMAGES = True

from photo_identify.config import MAX_UPLOAD_DIMENSION, UPLOAD_JPEG_QUALITY


def compute_md5(content: bytes) -> str:
    """计算字节内容的 MD5 哈希。

    Args:
        content: 输入字节内容。

    Returns:
        十六进制 MD5 字符串。
    """
    return hashlib.md5(content).hexdigest()

def compute_file_md5_chunked(path: str | Path, chunk_size: int = 8192 * 1024) -> str:
    """流式计算文件的 MD5 哈希，避免将大文件整个载入内存。

    Args:
        path: 文件路径。
        chunk_size: 每次读取的块大小（默认 8MB）。

    Returns:
        十六进制 MD5 字符串。
    """
    m = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            m.update(chunk)
    return m.hexdigest()


def compute_sha256(content: bytes) -> str:
    """计算字节内容的 SHA256 哈希。

    Args:
        content: 输入字节内容。

    Returns:
        十六进制 SHA256 字符串。
    """
    return hashlib.sha256(content).hexdigest()

def compute_file_sha256_chunked(path: str | Path, chunk_size: int = 8192 * 1024) -> str:
    """流式计算文件的 SHA256 哈希。

    Args:
        path: 文件路径。
        chunk_size: 每次读取的块大小（默认 8MB）。

    Returns:
        十六进制 SHA256 字符串。
    """
    m = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            m.update(chunk)
    return m.hexdigest()


def read_image_bytes(path: str | Path) -> bytes:
    """读取图片的原始字节内容。

    Args:
        path: 图片文件路径。

    Returns:
        图片二进制内容。
    """
    return Path(path).read_bytes()

def get_image_frame_bytes(path: str | Path) -> bytes:
    """智能获取文件用于展示/分析的图片字节。
    
    如果是普通图片，直接返回字节。
    如果是视频或包含视频的 .livp ，提取其中一帧并返回 JPEG 字节，避免大文件内存溢出。
    支持带有 #t=x.xs 的视频路径，直接抽取指定时间戳的帧。
    
    Args:
        path: 文件路径。
        
    Returns:
        代表文件画面的字节流。
    """
    path_str = str(path)
    timestamp_sec = None
    if "#t=" in path_str:
        actual_path_str, ts_str = path_str.split("#t=", 1)
        if ts_str.endswith("s"):
            ts_str = ts_str[:-1]
        try:
            timestamp_sec = float(ts_str)
        except ValueError:
            pass
    else:
        actual_path_str = path_str

    path_obj = Path(actual_path_str)
    ext = path_obj.suffix.lower()
    
    # 视频格式：用 opencv 抽帧
    if ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return _extract_frame_from_video(actual_path_str, timestamp_sec)
        
    # .livp 格式: 寻找 zip 中的 heic 或 mp4
    if ext == ".livp":
        return _extract_from_livp(actual_path_str)
        
    # 默认回退：直接读取
    return path_obj.read_bytes()

def _extract_frame_from_video(video_path: str, timestamp_sec: float | None = None) -> bytes:
    """使用 OpenCV 从视频中提取一帧转为 JPEG 字节。"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        if timestamp_sec is not None and timestamp_sec >= 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ValueError("无法从视频中提取视频帧")
        else:
            # 尝试跳过开头的纯黑帧，取第10帧或者中间帧，这里简单取第 10 帧或者读到的第一帧
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                    
            # 兜底：如果前10帧都没读到或者视频太短，重置重新读第一帧
            if 'frame' not in locals() or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ValueError("无法从视频中提取视频帧")
                
        cap.release()
        
        # 将 opencv BGR 转为 RGB 以便 PIL 打开或直接保存 JPEG
        # 我们这里直接用 cv2 imencode 编成 jpeg 返回字节
        success, encoded_image = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("帧编码为 JPEG 失败")
        return encoded_image.tobytes()
        
    except Exception as e:
        raise RuntimeError(f"提取视频帧失败 {video_path}: {e}")


def calc_dynamic_frame_count(duration: float) -> int:
    """根据视频时长动态计算最优抽帧数量。

    策略依据：短视频（≤30s）占绝大多数，少量帧即可覆盖内容；
    长视频使用对数增长，上限 10 帧避免过度消耗。

    Args:
        duration: 视频时长（秒）。

    Returns:
        建议的抽帧数量（1-10）。
    """
    if duration <= 3:
        return 1
    if duration <= 10:
        return 2
    if duration <= 30:
        return 3
    if duration <= 60:
        return 4
    if duration <= 120:
        return 5
    if duration <= 180:
        return 6
    if duration <= 300:
        return 7
    if duration <= 600:
        return 8
    # >10min：对数增长，上限 10 帧
    import math
    return min(10, 8 + int(math.log2(max(1, duration / 300))))


def extract_video_frames(video_path: str, frame_count: int | None = None) -> list[tuple[float, bytes]]:
    """从视频中动态抽取多帧，至少抽取 1 帧。

    使用均匀分布策略选取时间点，自动避开首尾纯黑帧。
    当 frame_count 为 None 时，根据视频时长自动计算最优抽帧数。

    Args:
        video_path: 视频文件路径。
        frame_count: 指定抽帧数量；None 表示自动计算。

    Returns:
        [(timestamp_seconds, jpeg_bytes), ...] 列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 动态计算抽帧数
    if frame_count is None:
        frame_count = calc_dynamic_frame_count(duration)

    # 计算均匀分布的时间点：ts[i] = duration * (i+1) / (count+1)
    # 自动避开 0s（可能是黑帧）和末尾
    timestamps: list[float] = []
    if duration <= 0:
        timestamps = [0.0]
    else:
        for i in range(frame_count):
            ts = duration * (i + 1) / (frame_count + 1)
            timestamps.append(ts)

    frames: list[tuple[float, bytes]] = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        success, encoded = cv2.imencode('.jpg', frame)
        if success:
            frames.append((ts, encoded.tobytes()))

    cap.release()

    # 兜底：如果上面全部失败，尝试读第一帧
    if not frames:
        cap2 = cv2.VideoCapture(video_path)
        ret, frame = cap2.read()
        cap2.release()
        if ret and frame is not None:
            success, encoded = cv2.imencode('.jpg', frame)
            if success:
                frames.append((0.0, encoded.tobytes()))

    if not frames:
        raise RuntimeError(f"无法从视频中提取任何帧: {video_path}")

    return frames

def _extract_from_livp(livp_path: str) -> bytes:
    """从 .livp (本质是 zip) 提取代表图像。优先找 heic/jpg，其次 mp4。"""
    try:
        if not zipfile.is_zipfile(livp_path):
            # 兼容：有些 LIVP 实际上不是 ZIP，而是已经重命名的普通 HEIC 或图片，尝试直接读取内容
            return Path(livp_path).read_bytes()
            
        with zipfile.ZipFile(livp_path, 'r') as zf:
            namelist = zf.namelist()
            # 找图片
            image_exts = {".heic", ".jpg", ".jpeg", ".png"}
            img_files = [f for f in namelist if Path(f).suffix.lower() in image_exts]
            if img_files:
                return zf.read(img_files[0])
                
            # 没有图片，找视频
            video_exts = {".mp4", ".mov"}
            vid_files = [f for f in namelist if Path(f).suffix.lower() in video_exts]
            if vid_files:
                # 把视频提取到临时文件再用 OpenCV 读取
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(zf.read(vid_files[0]))
                    tmp_path = tmp.name
                
                try:
                    return _extract_frame_from_video(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        raise ValueError(f"在 {livp_path} 中没有找到图片或视频内容")
    except Exception as e:
        raise RuntimeError(f"解析 .livp 失败 {livp_path}: {e}")


def format_timestamp(timestamp: float) -> str:
    """将 Unix 时间戳格式化为 ISO 时间字符串。

    Args:
        timestamp: Unix 时间戳。

    Returns:
        ISO 格式时间字符串；无法转换时返回空字符串。
    """
    try:
        return datetime.fromtimestamp(timestamp).isoformat()
    except (OSError, ValueError):
        return ""


def extract_exif(image: Image.Image) -> dict:
    """提取常用 EXIF 字段并转为可序列化字典。

    Args:
        image: PIL 图片对象。

    Returns:
        常用 EXIF 字段的字符串字典。
    """
    exif_data = {}
    raw_exif = image.getexif()
    if not raw_exif:
        return exif_data
    tag_map = {value: key for key, value in ExifTags.TAGS.items()}
    for name in ["Make", "Model", "DateTime", "FNumber", "ExposureTime", "ISOSpeedRatings", "FocalLength"]:
        key = tag_map.get(name)
        if key is None:
            continue
        value = raw_exif.get(key)
        if value is None:
            continue
        exif_data[name] = str(value)
    return exif_data


def extract_metadata(path: str | Path) -> dict:
    """提取图片的基础信息与 EXIF 数据。

    Args:
        path: 图片文件路径。

    Returns:
        包含文件名、路径、大小、尺寸、EXIF 等字段的字典。
    """
    path_obj = Path(path)
    stat = path_obj.stat()
    
    # 获取真正的图片帧字节，防止 PIL 直接去打开 .livp 或 .mp4 等非图片后缀而报错
    try:
        frame_bytes = get_image_frame_bytes(path_obj)
        with Image.open(io.BytesIO(frame_bytes)) as image:
            info = {
                "width": image.width,
                "height": image.height,
                "image_mode": image.mode,
                "image_format": image.format or "",
                "exif": extract_exif(image),
            }
    except Exception as e:
        logger.warning("无法提取 %s 元数据或图像属性: %s", path_obj.name, e)
        info = {
            "width": 0,
            "height": 0,
            "image_mode": "",
            "image_format": "",
            "exif": {},
        }
        
    return {
        "file_name": path_obj.name,
        "path": str(path_obj),
        "size_bytes": stat.st_size,
        "created_time": format_timestamp(stat.st_ctime),
        "modified_time": format_timestamp(stat.st_mtime),
        **info,
    }


def compress_for_upload(image_bytes: bytes, max_dim: int = MAX_UPLOAD_DIMENSION, quality: int = UPLOAD_JPEG_QUALITY) -> tuple[bytes, str]:
    """将图片缩放并压缩为 JPEG 格式，用于 API 上传。

    大图片（如 8MB PNG）经压缩后通常只有 200-500KB，
    大幅减少 base64 编码体积和 API token 消耗。

    Args:
        image_bytes: 原始图片字节内容。
        max_dim: 缩放后最长边不超过的像素数。
        quality: JPEG 压缩质量（1-100）。

    Returns:
        (压缩后的字节内容, 格式字符串 "jpeg")。
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            # 主动防御巨无霸图片造成的假死
            if image.width * image.height > 60000000:
                raise ValueError(f"巨无霸图片拦截 ({image.width}x{image.height})")

            # 转换 RGBA/P 等模式为 RGB 以便保存为 JPEG
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            # 按比例缩放超大图片
            w, h = image.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                image = image.resize(new_size, Image.LANCZOS)
            
            # 补充：对过小的图片进行 Padding (至少 32 像素)，以兼容部分模型的边界限制
            w, h = image.size
            if w < 32 or h < 32:
                new_w = max(w, 32)
                new_h = max(h, 32)
                pad_image = Image.new("RGB", (new_w, new_h), (255, 255, 255))
                # 居中粘贴原图
                pad_image.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
                image = pad_image

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), "jpeg"
    except Image.DecompressionBombError as e:
        raise ValueError(f"图片尺寸过大触发 PIL 防爆拦截: {e}")


def list_images(root: str | Path, extensions: frozenset[str]) -> list[str]:
    """递归遍历目录并返回符合扩展名的图片路径列表。

    Args:
        root: 根目录路径。
        extensions: 允许的扩展名集合（小写，含点号）。

    Returns:
        排序后的图片文件路径列表。
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        str(p) for p in root_path.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
