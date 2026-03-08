from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from photo_identify.image_utils import get_image_frame_bytes

DEFAULT_CACHE_DIR = Path(r"E:\Caches\Photo_Identify_Cahces")
DEFAULT_CACHE_MAX_SIZE_MB = 1024
DEFAULT_THUMBNAIL_SIZE = (150, 150)


def format_bytes(size_bytes: int) -> str:
    """将字节数格式化为易读文本。"""

    size = float(max(int(size_bytes), 0))
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            if size >= 100:
                return f"{size:.0f}{unit}"
            if size >= 10:
                return f"{size:.1f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size_bytes}B"



def build_thumbnail_jpeg_bytes_from_frame_bytes(
    frame_bytes: bytes,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> bytes:
    """将原始帧字节解码、缩放并编码为 JPEG 缩略图字节。"""

    if not frame_bytes:
        raise ValueError("缓存源字节不能为空")

    with Image.open(io.BytesIO(frame_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


def _normalize_face_bbox_str(bbox_str: str | None) -> str:
    normalized_bbox = str(bbox_str or "[]").strip() or "[]"
    try:
        bbox = json.loads(normalized_bbox)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            normalized_numbers = [float(value) for value in bbox]
            return json.dumps(normalized_numbers, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        pass
    return normalized_bbox


def _crop_and_circle_face_image(image: Image.Image, bbox_str: str, size: int = 80) -> Image.Image:
    image = ImageOps.exif_transpose(image)

    try:
        bbox = json.loads(bbox_str)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * 1.6
        left = int(cx - side / 2)
        top = int(cy - side / 2)
        right = int(cx + side / 2)
        bottom = int(cy + side / 2)
        face_img = image.crop((left, top, right, bottom))
    except Exception:
        min_dim = min(image.width, image.height)
        left = (image.width - min_dim) // 2
        top = (image.height - min_dim) // 2
        face_img = image.crop((left, top, left + min_dim, top + min_dim))

    face_img = face_img.resize((size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)

    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if face_img.mode != "RGBA":
        face_img = face_img.convert("RGBA")
    result.paste(face_img, (0, 0), mask)
    return result


def build_face_avatar_png_bytes_from_frame_bytes(
    frame_bytes: bytes,
    bbox_str: str,
    size: int = 80,
) -> bytes:
    """将原始帧字节解码、裁剪成人物圆形头像，并编码为 PNG 字节。"""

    if not frame_bytes:
        raise ValueError("头像缓存源字节不能为空")

    normalized_bbox = _normalize_face_bbox_str(bbox_str)
    with Image.open(io.BytesIO(frame_bytes)) as image:
        avatar = _crop_and_circle_face_image(image, normalized_bbox, size=size)
        buffer = io.BytesIO()
        avatar.save(buffer, format="PNG")
        return buffer.getvalue()


class DiskThumbnailCache:
    """基于磁盘的缩略图缓存，支持容量限制与 LRU 清理。"""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        max_size_bytes: int = DEFAULT_CACHE_MAX_SIZE_MB * 1024 * 1024,
    ) -> None:
        self._lock = threading.RLock()
        self._cache_dir = Path(DEFAULT_CACHE_DIR)
        self._max_size_bytes = DEFAULT_CACHE_MAX_SIZE_MB * 1024 * 1024
        self._estimated_size_bytes: int | None = None
        self._pending_prune_bytes = 0
        self._prune_check_threshold_bytes = 64 * 1024 * 1024
        self.configure(cache_dir=cache_dir, max_size_bytes=max_size_bytes)

    def configure(self, cache_dir: str | Path, max_size_bytes: int) -> None:
        """更新缓存目录与容量限制，并确保目录存在。"""

        resolved_dir = Path(str(cache_dir or DEFAULT_CACHE_DIR)).expanduser()
        if not resolved_dir.is_absolute():
            resolved_dir = resolved_dir.resolve()
        resolved_limit = max(int(max_size_bytes or 0), 1 * 1024 * 1024)

        with self._lock:
            self._cache_dir = resolved_dir
            self._max_size_bytes = resolved_limit
            self._estimated_size_bytes = None
            self._pending_prune_bytes = 0
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def max_size_bytes(self) -> int:
        return self._max_size_bytes

    def folder_size(self) -> int:
        """统计当前缓存目录内所有文件占用总大小。"""

        total = 0
        for file_path in self._iter_cache_files():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
        with self._lock:
            self._estimated_size_bytes = total
        return total

    def clear_files(self) -> int:
        """清空缓存目录中的所有文件，但保留目录结构。"""

        deleted_count = 0
        with self._lock:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            for file_path in list(self._iter_cache_files()):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except OSError:
                    continue
            self._estimated_size_bytes = 0
            self._pending_prune_bytes = 0
        return deleted_count

    def warm_thumbnail(self, file_path: str | Path, size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> Path:
        """确保指定文件的缩略图已写入缓存，并返回缓存文件路径。"""

        normalized_path = str(file_path or "").strip()
        if not normalized_path:
            raise ValueError("缓存目标路径不能为空")

        cache_path = self._build_cache_path(normalized_path, size)
        with self._lock:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                self._touch_file(cache_path)
                return cache_path

        frame_bytes = get_image_frame_bytes(normalized_path)
        return self.warm_thumbnail_from_bytes(normalized_path, frame_bytes, size=size)

    def warm_thumbnail_from_bytes(
        self,
        file_path: str | Path,
        frame_bytes: bytes,
        size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
    ) -> Path:
        """使用已预读到内存的图片字节生成缓存缩略图。"""

        thumbnail_bytes = build_thumbnail_jpeg_bytes_from_frame_bytes(frame_bytes, size=size)
        return self.warm_thumbnail_encoded_bytes(file_path=file_path, thumbnail_bytes=thumbnail_bytes, size=size)

    def warm_thumbnail_encoded_bytes(
        self,
        file_path: str | Path,
        thumbnail_bytes: bytes,
        size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
        suffix: str = ".jpg",
    ) -> Path:
        """将已编码完成的缩略图字节写入缓存。"""

        normalized_path = str(file_path or "").strip()
        if not normalized_path:
            raise ValueError("缓存目标路径不能为空")
        if not thumbnail_bytes:
            raise ValueError("缓存缩略图字节不能为空")

        cache_path = self._build_cache_path(normalized_path, size, suffix=suffix)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

        with self._lock:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                self._touch_file(cache_path)
                return cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                self._touch_file(cache_path)
                return cache_path
            try:
                temp_path.write_bytes(thumbnail_bytes)
                os.replace(temp_path, cache_path)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
            self._touch_file(cache_path)
            if self._estimated_size_bytes is None:
                self._estimated_size_bytes = self.folder_size()
            self._estimated_size_bytes += len(thumbnail_bytes)
            self._pending_prune_bytes += len(thumbnail_bytes)
            self._prune_if_needed_locked()
        return cache_path

    def load_thumbnail(self, file_path: str | Path, size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE) -> Image.Image:
        """读取缓存缩略图并返回独立的 PIL 图像对象。"""

        cache_path = self.warm_thumbnail(file_path, size=size)
        with Image.open(cache_path) as image:
            return image.copy()

    def has_thumbnail(
        self,
        file_path: str | Path,
        size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
        suffix: str = ".jpg",
    ) -> bool:
        """判断指定文件的缩略图是否已经存在于缓存中。"""

        normalized_path = str(file_path or "").strip()
        if not normalized_path:
            return False
        cache_path = self._build_cache_path(normalized_path, size, suffix=suffix)
        return cache_path.exists()

    def _build_cache_path(self, file_path: str, size: tuple[int, int], suffix: str = ".jpg") -> Path:
        width, height = size
        normalized_suffix = str(suffix or ".jpg").strip() or ".jpg"
        if not normalized_suffix.startswith("."):
            normalized_suffix = f".{normalized_suffix}"
        digest = hashlib.sha1(f"{file_path}|{width}x{height}".encode("utf-8")).hexdigest()
        return self._cache_dir / digest[:2] / digest[2:4] / f"{digest}{normalized_suffix}"

    def _build_thumbnail_image(self, file_path: str, size: tuple[int, int]) -> Image.Image:
        frame_bytes = get_image_frame_bytes(file_path)
        return self._build_thumbnail_image_from_bytes(frame_bytes, size)

    def _build_thumbnail_image_from_bytes(self, frame_bytes: bytes, size: tuple[int, int]) -> Image.Image:
        with Image.open(io.BytesIO(frame_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            return image.copy()

    def _iter_cache_files(self):
        if not self._cache_dir.exists():
            return []
        return [path for path in self._cache_dir.rglob("*") if path.is_file()]

    def _touch_file(self, file_path: Path) -> None:
        try:
            stat = file_path.stat()
            now = time.time()
            os.utime(file_path, (now, stat.st_mtime))
        except OSError:
            pass

    def _prune_if_needed_locked(self) -> None:
        estimated_size = self._estimated_size_bytes
        if estimated_size is None:
            estimated_size = self.folder_size()
        if estimated_size <= self._max_size_bytes and self._pending_prune_bytes < self._prune_check_threshold_bytes:
            return

        total_size = self.folder_size()
        self._pending_prune_bytes = 0
        if total_size <= self._max_size_bytes:
            self._estimated_size_bytes = total_size
            return

        target_size = int(self._max_size_bytes * 0.9)
        sortable_files: list[tuple[float, float, Path]] = []
        for file_path in self._iter_cache_files():
            try:
                stat = file_path.stat()
            except OSError:
                continue
            sortable_files.append((stat.st_atime, stat.st_mtime, file_path))

        sortable_files.sort(key=lambda item: (item[0], item[1], str(item[2])))
        for _, _, file_path in sortable_files:
            if total_size <= target_size:
                break
            try:
                file_size = file_path.stat().st_size
                file_path.unlink()
                total_size -= file_size
            except OSError:
                continue
        self._estimated_size_bytes = max(total_size, 0)


_GLOBAL_CACHE = DiskThumbnailCache()


def configure_thumbnail_cache(cache_dir: str | Path, max_size_bytes: int) -> DiskThumbnailCache:
    """更新全局缩略图缓存配置。"""

    _GLOBAL_CACHE.configure(cache_dir=cache_dir, max_size_bytes=max_size_bytes)
    return _GLOBAL_CACHE


def get_thumbnail_cache() -> DiskThumbnailCache:
    """返回全局缩略图缓存实例。"""

    return _GLOBAL_CACHE


def load_cached_thumbnail_image(
    file_path: str | Path,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> Image.Image:
    """读取或生成缓存缩略图。"""

    return _GLOBAL_CACHE.load_thumbnail(file_path=file_path, size=size)


def warm_cached_thumbnail_from_bytes(
    file_path: str | Path,
    frame_bytes: bytes,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> Path:
    """使用预读字节写入全局缓存缩略图。"""

    return _GLOBAL_CACHE.warm_thumbnail_from_bytes(file_path=file_path, frame_bytes=frame_bytes, size=size)


def warm_cached_thumbnail_encoded_bytes(
    file_path: str | Path,
    thumbnail_bytes: bytes,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> Path:
    """使用已编码好的缩略图 JPEG 字节写入全局缓存。"""

    return _GLOBAL_CACHE.warm_thumbnail_encoded_bytes(file_path=file_path, thumbnail_bytes=thumbnail_bytes, size=size)


def _build_face_avatar_cache_key(file_path: str | Path, bbox_str: str | None) -> str:
    normalized_path = str(file_path or "").strip()
    if not normalized_path:
        raise ValueError("头像缓存目标路径不能为空")
    normalized_bbox = _normalize_face_bbox_str(bbox_str)
    return f"face_avatar|{normalized_path}|{normalized_bbox}"


def warm_cached_face_avatar(file_path: str | Path, bbox_str: str | None, size: int = 80) -> Path:
    """确保指定人物头像已写入磁盘缓存，并返回缓存文件路径。"""

    cache_size = (size, size)
    cache_key = _build_face_avatar_cache_key(file_path, bbox_str)
    cache_path = _GLOBAL_CACHE._build_cache_path(cache_key, cache_size, suffix=".png")

    with _GLOBAL_CACHE._lock:
        _GLOBAL_CACHE._cache_dir.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            _GLOBAL_CACHE._touch_file(cache_path)
            return cache_path

    frame_bytes = get_image_frame_bytes(file_path)
    avatar_bytes = build_face_avatar_png_bytes_from_frame_bytes(frame_bytes, str(bbox_str or "[]"), size=size)
    return _GLOBAL_CACHE.warm_thumbnail_encoded_bytes(
        file_path=cache_key,
        thumbnail_bytes=avatar_bytes,
        size=cache_size,
        suffix=".png",
    )


def load_cached_face_avatar_image(file_path: str | Path, bbox_str: str | None, size: int = 80) -> Image.Image:
    """读取或生成缓存的人物圆形头像。"""

    cache_path = warm_cached_face_avatar(file_path=file_path, bbox_str=bbox_str, size=size)
    with Image.open(cache_path) as image:
        return image.copy()


def has_cached_face_avatar(file_path: str | Path, bbox_str: str | None, size: int = 80) -> bool:
    """判断人物圆形头像是否已存在于全局缓存中。"""

    cache_key = _build_face_avatar_cache_key(file_path, bbox_str)
    return _GLOBAL_CACHE.has_thumbnail(file_path=cache_key, size=(size, size), suffix=".png")


def has_cached_thumbnail(
    file_path: str | Path,
    size: tuple[int, int] = DEFAULT_THUMBNAIL_SIZE,
) -> bool:
    """判断缩略图是否已存在于全局缓存中。"""

    return _GLOBAL_CACHE.has_thumbnail(file_path=file_path, size=size)
