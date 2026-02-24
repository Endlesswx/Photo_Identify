"""图片扫描主流程：并发遍历目录，调用 LLM 分析并入库。

负责编排整个扫描过程：
- 递归收集图片文件
- 基于 path+size+mtime 快速跳过 + MD5 精确去重
- 多线程并发调用 LLM API
- 逐条写入 SQLite（断点续扫）
- 实时进度条显示
"""

import logging
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from datetime import datetime
from pathlib import Path

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_VIDEO_FRAME_INTERVAL,
    DEFAULT_VISION_MODEL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TPM_LIMIT,
    DEFAULT_WORKERS,
)
from photo_identify.image_utils import (
    compress_for_upload,
    compute_md5,
    compute_sha256,
    compute_file_md5_chunked,
    compute_file_sha256_chunked,
    extract_metadata,
    extract_video_frames,
    list_images,
    read_image_bytes,
    get_image_frame_bytes,
)
from photo_identify.llm import RateLimiter, call_image_model
from photo_identify.storage import Storage

logger = logging.getLogger(__name__)


class ScanStats:
    """扫描统计计数器（线程安全不要求精确，仅用于进度显示）。"""

    def __init__(self):
        """初始化所有计数器为 0。"""
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.current = 0


def _render_progress_bar(current: int, total: int, width: int = 28) -> str:
    """渲染进度条字符串。

    Args:
        current: 当前进度。
        total: 总数。
        width: 进度条宽度（字符数）。

    Returns:
        进度条字符串，如 [####----]。
    """
    if total <= 0:
        total = 1
    ratio = min(1.0, max(0.0, current / total))
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


_last_print_time = 0.0

def _print_progress(stats: ScanStats, last_path: str, force: bool = False):
    """在控制台打印单行进度条（含有节流逻辑以应对万级并发）。

    Args:
        stats: 扫描统计对象。
        last_path: 当前处理的图片路径。
        force: 是否强制刷新（无视节流限制）。
    """
    global _last_print_time
    now = time.monotonic()
    if not force and now - _last_print_time < 0.1:
        return
    _last_print_time = now

    terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
    bar = _render_progress_bar(stats.current, stats.total)
    pct = f"{stats.current * 100 // max(1, stats.total)}%"
    suffix = f"{stats.current}/{stats.total} 新增:{stats.processed} 跳过:{stats.skipped} 失败:{stats.failed}"
    name = Path(last_path).name if last_path else ""
    line = f"{bar} {pct} {suffix} {name}"
    if len(line) > terminal_width - 1:
        line = line[: terminal_width - 1]
    
    # 获取当前的全局 sys.stdout（可能是被 GUI 替换过的重定向器）
    sys.stdout.write("\r" + line.ljust(max(0, terminal_width - 1)))
    sys.stdout.flush()


def _analyze_single(
    path: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    rate_limiter: RateLimiter,
) -> dict:
    """分析单张图片：读取→压缩→调用 LLM→合并结果。

    Args:
        path: 图片文件路径。
        api_key: API Key。
        base_url: API Base URL。
        model: 模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        timeout: HTTP 超时秒数。
        rate_limiter: 速率限制器。

    Returns:
        合并了元数据和 LLM 分析结果的完整记录字典。
    """
    image_bytes = get_image_frame_bytes(path)
    metadata = extract_metadata(path)
    metadata["md5"] = compute_file_md5_chunked(path)
    metadata["sha256"] = compute_file_sha256_chunked(path)

    # 压缩图片用于上传
    compressed_bytes, upload_format = compress_for_upload(image_bytes)
    logger.debug(
        "图片压缩: %s  %d KB → %d KB",
        Path(path).name,
        len(image_bytes) // 1024,
        len(compressed_bytes) // 1024,
    )

    llm_result = call_image_model(
        image_bytes=compressed_bytes,
        image_format=upload_format,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        rate_limiter=rate_limiter,
    )

    metadata.update(llm_result)
    metadata["analyzed_at"] = datetime.now().isoformat()
    return metadata


def scan(
    paths: list[str],
    db_path: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_VISION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    rpm_limit: int = DEFAULT_RPM_LIMIT,
    tpm_limit: int = DEFAULT_TPM_LIMIT,
    workers: int = DEFAULT_WORKERS,
    extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS,
    cancel_event: threading.Event | None = None,
    video_frame_interval: float = DEFAULT_VIDEO_FRAME_INTERVAL,
):
    """执行图片扫描主流程。

    遍历指定目录，跳过已分析的图片，并发调用 LLM 分析新图片并存入 SQLite。

    Args:
        paths: 要扫描的目录列表。
        db_path: SQLite 数据库文件路径。
        api_key: API Key。
        base_url: API Base URL。
        model: 模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        timeout: HTTP 超时秒数。
        rpm_limit: 每分钟最大请求数。
        tpm_limit: 每分钟最大 token 数。
        workers: 并发线程数。
        extensions: 允许的图片扩展名集合。
    """
    storage = Storage(db_path)
    rate_limiter = RateLimiter(rpm_limit, tpm_limit)

    # 收集所有图片路径
    all_images: list[str] = []
    for p in paths:
        found = list_images(p, extensions)
        logger.info("目录 %s 发现 %d 张图片", p, len(found))
        all_images.extend(found)

    if not all_images:
        print("未发现任何图片文件。")
        storage.close()
        return

    # 加载已有数据用于跳过判断
    known_md5s = storage.get_known_md5s()
    known_paths = storage.get_known_paths()

    stats = ScanStats()
    stats.total = len(all_images)

    # 筛选需要分析的图片（仅做快速路径匹配，不计算 MD5）
    to_analyze: list[str] = []
    for img_path in all_images:
        # 检查取消信号
        if cancel_event and cancel_event.is_set():
            break
        stats.current += 1
        # 快速路径跳过：path + size + mtime 完全匹配
        cached = known_paths.get(img_path)
        if cached:
            try:
                st = Path(img_path).stat()
                cached_size, cached_mtime = cached
                if cached_size == st.st_size and cached_mtime == datetime.fromtimestamp(st.st_mtime).isoformat():
                    stats.skipped += 1
                    _print_progress(stats, img_path, force=(stats.current == stats.total))
                    continue
            except OSError:
                pass
        to_analyze.append(img_path)
        _print_progress(stats, img_path, force=(stats.current == stats.total))

    if cancel_event and cancel_event.is_set():
        storage.close()
        return

    if not to_analyze:
        stats.current = stats.total
        _print_progress(stats, "", force=True)
        sys.stdout.write("\n")
        print(f"所有 {stats.total} 张图片均已分析，无需处理。")
        storage.close()
        return

    # 重置进度用于分析阶段
    stats.current = stats.skipped + stats.failed
    print(f"\n待分析: {len(to_analyze)} 张  |  跳过: {stats.skipped}  |  总计: {stats.total}")

    # 用于线程安全的 MD5 去重
    _md5_lock = threading.Lock()

    # 视频后缀集合
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

    # 并发分析（含 MD5 去重 + 视频多帧）
    _SKIPPED_SENTINEL = {"__skipped__": True}
    def _worker(img_path: str) -> tuple[str, dict | list[dict] | None]:
        """单文件分析工作函数。视频文件返回 list[dict]，图片返回 dict。"""
        if cancel_event and cancel_event.is_set():
            return img_path, None

        ext = Path(img_path).suffix.lower()

        # ---- 视频多帧路径 ----
        if ext in _VIDEO_EXTS:
            try:
                frames = extract_video_frames(img_path, video_frame_interval)
            except Exception as exc:
                logger.error("视频抽帧失败 %s: %s", img_path, exc)
                return img_path, None

            file_metadata = extract_metadata(img_path)
            file_md5 = compute_file_md5_chunked(img_path)
            file_sha256 = compute_file_sha256_chunked(img_path)
            results: list[dict] = []

            for ts, frame_bytes in frames:
                if cancel_event and cancel_event.is_set():
                    break
                # 每帧用帧内容的 MD5 做唯一键
                frame_md5 = compute_md5(frame_bytes)
                with _md5_lock:
                    if frame_md5 in known_md5s:
                        results.append(_SKIPPED_SENTINEL)
                        continue
                    known_md5s.add(frame_md5)

                ts_label = f"{ts:.1f}s"
                compressed, fmt = compress_for_upload(frame_bytes)
                try:
                    llm_result = call_image_model(
                        image_bytes=compressed,
                        image_format=fmt,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        rate_limiter=rate_limiter,
                    )
                except Exception as exc:
                    logger.error("视频帧分析失败 %s@%s: %s", img_path, ts_label, exc)
                    continue

                record = dict(file_metadata)  # 复制文件级元数据
                record["md5"] = frame_md5
                record["sha256"] = file_sha256
                record["path"] = f"{img_path}#t={ts_label}"
                record["file_name"] = f"{Path(img_path).stem}_{ts_label}{Path(img_path).suffix}"
                record.update(llm_result)
                record["analyzed_at"] = datetime.now().isoformat()
                results.append(record)

            return img_path, results if results else None

        # ---- 普通图片路径 ----
        # 先计算 MD5 做精确去重
        try:
            md5 = compute_file_md5_chunked(img_path)
        except (OSError, PermissionError) as exc:
            logger.warning("无法读取文件 %s: %s", img_path, exc)
            return img_path, None
        with _md5_lock:
            if md5 in known_md5s:
                return img_path, _SKIPPED_SENTINEL
            known_md5s.add(md5)
        try:
            return img_path, _analyze_single(
                path=img_path,
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                rate_limiter=rate_limiter,
            )
        except Exception as exc:
            logger.error("分析失败 %s: %s", img_path, exc, exc_info=False)
            return img_path, None

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, p): p for p in to_analyze}
            for future in as_completed(futures):
                # 检查取消信号
                if cancel_event and cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    sys.stdout.write("\n")
                    print("\n⚠️  扫描已被用户中止，已分析的数据已保存。")
                    break
                try:
                    img_path, result = future.result()
                except CancelledError:
                    continue
                # 视频多帧结果：list[dict]
                if isinstance(result, list):
                    for r in result:
                        stats.current += 1
                        if r is _SKIPPED_SENTINEL:
                            stats.skipped += 1
                        elif "error" in r:
                            stats.failed += 1
                            logger.error("API 错误 %s: %s", img_path, r["error"])
                        else:
                            storage.upsert(r)
                            stats.processed += 1
                        _print_progress(stats, img_path, force=(stats.current == stats.total))
                # 单条结果
                else:
                    stats.current += 1
                    if result is _SKIPPED_SENTINEL:
                        stats.skipped += 1
                    elif result is None:
                        stats.failed += 1
                    elif "error" in result:
                        stats.failed += 1
                        logger.error("API 错误 %s: %s", img_path, result["error"])
                    else:
                        storage.upsert(result)
                        stats.processed += 1
                    _print_progress(stats, img_path, force=(stats.current == stats.total))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("\n⚠️  用户中断 (Ctrl+C)，已分析的数据已保存。")

    sys.stdout.write("\n")
    print(
        f"\n扫描完成 — 总计: {stats.total}  新增: {stats.processed}  "
        f"跳过: {stats.skipped}  失败: {stats.failed}  "
        f"数据库记录数: {storage.count()}"
    )
    storage.close()
