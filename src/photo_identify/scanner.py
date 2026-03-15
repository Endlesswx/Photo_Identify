"""图片扫描主流程：异步并发遍历目录，调用 LLM 分析并入库。

负责编排整个扫描过程：
- 递归收集图片文件
- 基于 path+size+mtime 快速跳过 + MD5 精确去重
- 使用 asyncio 和 aiohttp 异步并发调用模型 API 极大提升吞吐量 (如 vLLM Continuous Batching)
- 把图片压缩预处理等 CPU 密集型任务拆分并使用协程在独立线程池中处理，解放事件循环
- 逐条写入 SQLite（断点续扫）
- tqdm 实时进度条显示
"""


import logging
import sys
import time
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import IO

from tqdm import tqdm

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_IMAGE_EXTENSIONS,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TPM_LIMIT,
    DEFAULT_WORKERS,
)
from photo_identify.image_utils import (
    compress_for_upload,
    compute_md5,
    compute_file_md5_chunked,
    compute_file_sha256_chunked,
    extract_metadata,
    extract_video_frames,
    list_images,
    get_image_frame_bytes,
)
from photo_identify.embedding_runtime import get_text_embedding_async
from photo_identify.llm import RateLimiter, async_call_image_model
from photo_identify.storage import Storage
import aiohttp
import asyncio

logger = logging.getLogger(__name__)

IMAGE_EXTRACTION_LABEL = "图片/视频信息提取"
FACE_SCAN_LABEL = "人物扫描"


class ScanStats:
    """扫描统计计数器（线程安全不要求精确，仅用于进度显示）。"""

    def __init__(self):
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.current = 0


def _wait_if_paused(cancel_event: threading.Event | None, pause_event: threading.Event | None, interval: float = 0.2) -> bool:
    """在同步流程中等待暂停结束；若期间收到停止信号则返回 True。"""

    while pause_event is not None and pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return True
        time.sleep(interval)
    return bool(cancel_event is not None and cancel_event.is_set())


async def _wait_if_paused_async(
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    interval: float = 0.2,
) -> bool:
    """在异步流程中等待暂停结束；若期间收到停止信号则返回 True。"""

    while pause_event is not None and pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return True
        await asyncio.sleep(interval)
    return bool(cancel_event is not None and cancel_event.is_set())


def _extract_faces_with_resize(image_bytes: bytes, *, raise_on_error: bool = False) -> list[dict]:
    """提取人脸前修正方向并适当缩放，以避免高分原图导致 CPU 转换瓶颈和内存激增。"""
    try:
        from photo_identify.face_manager import extract_faces
        import numpy as np
        from PIL import Image, ImageOps
        import io

        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        orig_w, orig_h = img.width, img.height
        max_dim = 1920
        scale = 1.0

        if max(orig_w, orig_h) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)
            scale = orig_w / img.width

        image_np = np.array(img)
        faces = extract_faces(image_np)

        if scale != 1.0 and faces:
            for face in faces:
                bbox = face["bbox"]
                face["bbox"] = [
                    bbox[0] * scale,
                    bbox[1] * scale,
                    bbox[2] * scale,
                    bbox[3] * scale,
                ]

        return faces
    except Exception as e:
        logger.error("人脸提取过程中出错: %s", e, exc_info=False)
        if raise_on_error:
            raise
        return []


async def _analyze_single_async(
    path: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    session: aiohttp.ClientSession,
    rate_limiter: RateLimiter,
    enable_face_scan: bool = False,
    face_cache: dict | None = None,
    face_cache_lock: threading.Lock | None = None,
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    embedding_model: str = "",
    embedding_backend: str = "",
    embedding_workers: int = 1,
    embedding_semaphore: asyncio.Semaphore | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> dict:
    """分析单张图片：读取→压缩(CPU多核)→调用 LLM(异步)→合并结果。"""
    if await _wait_if_paused_async(cancel_event=cancel_event, pause_event=pause_event):
        return {"__cancelled__": True}

    # CPU 密集型操作：读取、元数据提取、计算 Hash、压缩 等，放到线程池里以免阻塞事件循环
    def _cpu_bound_prep():
        b = get_image_frame_bytes(path)
        m = extract_metadata(path)
        m["md5"] = compute_file_md5_chunked(path)
        m["sha256"] = compute_file_sha256_chunked(path)
        
        # 针对 Ultra 5 245KF 优化，引入 max_dim=1120 的限制
        c_bytes, c_fmt = compress_for_upload(b, max_dim=1120)
        return b, m, c_bytes, c_fmt

    image_bytes, metadata, compressed_bytes, upload_format = await asyncio.to_thread(_cpu_bound_prep)
    if await _wait_if_paused_async(cancel_event=cancel_event, pause_event=pause_event):
        return {"__cancelled__": True}

    logger.debug(
        "图片压缩: %s  %d KB → %d KB",
        Path(path).name,
        len(image_bytes) // 1024,
        len(compressed_bytes) // 1024,
    )

    llm_result = await async_call_image_model(
        image_bytes=compressed_bytes,
        session=session,
        image_format=upload_format,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        rate_limiter=rate_limiter,
    )
    if await _wait_if_paused_async(cancel_event=cancel_event, pause_event=pause_event):
        return {"__cancelled__": True}

    if embedding_model and "error" not in llm_result:
        parts = []
        if llm_result.get("scene"):
            parts.append(str(llm_result["scene"]))
        if llm_result.get("objects"):
            obj = llm_result["objects"]
            parts.append(", ".join(obj) if isinstance(obj, list) else str(obj))
        text_to_embed = " ".join(parts).strip()
        if text_to_embed:
            async def _encode_embedding() -> list[float]:
                """根据当前 embedding 配置获取单条文本向量。"""

                return await get_text_embedding_async(
                    text=text_to_embed,
                    model=embedding_model,
                    backend=embedding_backend,
                    session=session,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    timeout=timeout,
                    rate_limiter=rate_limiter,
                    workers=embedding_workers,
                )

            if embedding_backend == "local" and embedding_semaphore is not None:
                async with embedding_semaphore:
                    emb = await _encode_embedding()
            else:
                emb = await _encode_embedding()
            if emb:
                llm_result["text_embedding"] = emb

    metadata.update(llm_result)

    if enable_face_scan:
        cached_faces = None
        if face_cache is not None and face_cache_lock is not None:
            with face_cache_lock:
                cached_faces = face_cache.pop(path, None)
        if cached_faces is not None:
            metadata["faces"] = cached_faces
        else:
            metadata["faces"] = await asyncio.to_thread(_extract_faces_with_resize, image_bytes)

    metadata["analyzed_at"] = datetime.now().isoformat()
    return metadata


def scan(
    paths: list[str],
    db_path: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_IMAGE_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    rpm_limit: int = DEFAULT_RPM_LIMIT,
    tpm_limit: int = DEFAULT_TPM_LIMIT,
    workers: int = DEFAULT_WORKERS,
    extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    enable_face_scan: bool = False,
    progress_writers: tuple[IO, IO] | None = None,
    video_workers: int = 3,
    model_supports_video: bool = False,
    transcoded_video_path: str = "",
    embedding_api_key: str = "",
    embedding_base_url: str = "",
    embedding_model: str = "",
    embedding_backend: str = "",
    embedding_workers: int = 1,
):
    """执行图片扫描主流程。

    Args:
        paths: 要扫描的目录列表。
        db_path: SQLite 数据库文件路径。
        api_key: API Key。
        base_url: API Base URL。
        model: 图片分析模型名称。
        temperature: 采样温度。
        max_tokens: 最大输出 token 数。
        timeout: HTTP 超时秒数。
        rpm_limit: 每分钟最大请求数。
        tpm_limit: 每分钟最大 token 数。
        workers: 并发线程数。
        extensions: 允许的图片扩展名集合。
        enable_face_scan: 是否进行人脸扫描和聚类。
        progress_writers: GUI 进度条写入器 ``(image_writer, face_writer)``。
    """
    import time as _time
    import queue
    t_start = _time.perf_counter()

    storage = Storage(db_path)
    rate_limiter = RateLimiter(rpm_limit, tpm_limit)

    # progress writer 引用（用于 _do_clustering / 最终状态写入 GUI）
    _llm_writer = progress_writers[0] if progress_writers else None
    _face_writer = progress_writers[1] if progress_writers else None
    embedding_sem: asyncio.Semaphore | None = None

    if enable_face_scan:
        from photo_identify.face_manager import get_face_app, get_device_mode
        get_face_app()
        logger.info("人脸引擎已启动，当前模式: %s", get_device_mode())

    # 收集所有图片路径
    all_images: list[str] = []
    for p in paths:
        if _wait_if_paused(cancel_event, pause_event):
            break
        found = list_images(p, extensions)
        logger.info("目录 %s 发现 %d 张图片", p, len(found))
        all_images.extend(found)

    def _do_clustering():
        if not enable_face_scan:
            return
        try:
            from photo_identify.face_manager import cluster_face_embeddings
            import numpy as np
            logger.info("正在聚类相似人脸...")
            if _face_writer:
                _face_writer.write(f"[{FACE_SCAN_LABEL}] 正在聚类相似人脸...\n")
            all_faces = storage.get_all_faces()
            if not all_faces:
                logger.info("未发现人脸数据，跳过聚类。")
                if _face_writer:
                    _face_writer.write(f"[{FACE_SCAN_LABEL}] 未发现人脸数据\n")
                return

            embeddings_with_ids = []
            for face_id, embedding_bytes in all_faces:
                embedding_np = np.frombuffer(embedding_bytes, dtype=np.float32)
                embeddings_with_ids.append((face_id, embedding_np))

            t_clust_start = _time.perf_counter()
            cluster_mapping = cluster_face_embeddings(embeddings_with_ids)
            storage.update_face_clusters(cluster_mapping)
            n_persons = len(set(c for c in cluster_mapping.values() if c >= 0))
            t_clust_cost = _time.perf_counter() - t_clust_start
            logger.info("人脸聚类完成(耗时 %.1fs)，共识别出 %d 位主要人物", t_clust_cost, n_persons)
            if _face_writer:
                _face_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 聚类完成(耗时 {t_clust_cost:.1f}s) — {n_persons} 位人物\n")
        except Exception as e:
            logger.error("人脸聚类出错: %s", e)
            if _face_writer:
                _face_writer.write(f"[{FACE_SCAN_LABEL}] 聚类出错: {e}\n")

    if not all_images:
        logger.info("未发现任何图片文件。")
        if _llm_writer:
            _llm_writer.write(f"[{IMAGE_EXTRACTION_LABEL}] 未发现任何图片\n")
        _do_clustering()
        storage.close()
        return

    # 加载已有数据用于跳过判断
    known_md5s = storage.get_known_md5s()
    known_paths = storage.get_known_paths()
    face_scanned_md5s = storage.get_face_scanned_md5s() if enable_face_scan else set()
    skipped_paths = storage.get_skipped_paths()

    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
    face_todo: list[tuple[str, str]] = []   # (img_path, md5) — 仅需人脸提取

    # 构建已处理视频的基础路径集合（视频帧存储为 path#t=ts）
    known_video_bases: set[str] = set()
    for p in known_paths:
        idx = p.find("#t=")
        if idx >= 0:
            known_video_bases.add(p[:idx])

    stats = ScanStats()
    stats.total = len(all_images)

    # 筛选需要分析的图片（仅做快速路径匹配，不计算 MD5）
    to_analyze: list[str] = []
    for img_path in all_images:
        if _wait_if_paused(cancel_event, pause_event):
            break
        if cancel_event and cancel_event.is_set():
            break
        stats.current += 1

        try:
            st = Path(img_path).stat()
            if st.st_size == 0:
                stats.skipped += 1
                logger.debug("跳过大小为 0 的文件: %s", img_path)
                continue
        except OSError:
            stats.skipped += 1
            continue

        ext = Path(img_path).suffix.lower()

        # 1.如果视频文件同目录存在与视频文件名称一样的jpg文件，跳过该视频的扫描。
        if ext in _VIDEO_EXTS:
            img_p = Path(img_path)
            if img_p.with_suffix(".jpg").exists() or img_p.with_suffix(".JPG").exists():
                stats.skipped += 1
                continue

        # 已处理过的视频直接跳过（视频帧不做人脸提取）
        if ext in _VIDEO_EXTS and img_path in known_video_bases:
            stats.skipped += 1
            continue

        # 曾经持续失败的文件直接跳过
        if img_path in skipped_paths:
            stats.skipped += 1
            continue

        cached = known_paths.get(img_path)
        if cached:
            cached_size, cached_mtime, cached_md5 = cached
            if cached_size == st.st_size and cached_mtime == datetime.fromtimestamp(st.st_mtime).isoformat():
                if enable_face_scan and cached_md5 not in face_scanned_md5s:
                    # LLM 已完成，仅需人脸提取 → 视频跳过，图片进入 Phase 2
                    if ext not in _VIDEO_EXTS:
                        face_todo.append((img_path, cached_md5))
                    face_scanned_md5s.add(cached_md5)
                stats.skipped += 1
                continue
        to_analyze.append(img_path)

    if cancel_event and cancel_event.is_set():
        _do_clustering()
        storage.close()
        return

    if not to_analyze and not face_todo:
        logger.info("所有 %d 张图片均已分析，无需处理。", stats.total)
        if _llm_writer:
            _llm_writer.write(f"[{IMAGE_EXTRACTION_LABEL}] ✓ 全部 {stats.total} 张已完成\n")
        if enable_face_scan and _face_writer:
            _face_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 全部 {stats.total} 张已完成\n")
        _do_clustering()
        storage.close()
        return

    # 重置进度用于分析阶段
    stats.current = stats.skipped + stats.failed
    logger.info(
        "待 LLM 分析: %d 张 | 待人脸识别: %d 张 | 跳过: %d | 总计: %d",
        len(to_analyze), len(face_todo), stats.skipped, stats.total,
    )

    # 线程安全锁
    _md5_lock = threading.Lock()
    _face_found_lock = threading.Lock()
    _face_found = 0

    # --- GPU 人脸提取流水线（为 Phase 2 预缓存） ---
    _face_cache: dict[str, list] = {}
    _face_cache_lock = threading.Lock()

    # 将所有需要做人脸识别的（新图片 + 旧图片补扫）都加入预提取队列
    _all_face_todo = []
    if enable_face_scan:
        _all_face_todo.extend(face_todo)
        # 预先将需要扫描的图片（普通图片）加入提取队列
        for img_path in to_analyze:
            ext = Path(img_path).suffix.lower()
            if ext not in _VIDEO_EXTS:
                _all_face_todo.append((img_path, None)) # md5 is None for to_analyze initially

    if enable_face_scan and _all_face_todo:
        def _face_pipeline():
            _pipeline_count = 0
            for img_path, _md5 in _all_face_todo:
                if _wait_if_paused(cancel_event, pause_event):
                    break
                if cancel_event and cancel_event.is_set():
                    break
                ext = Path(img_path).suffix.lower()
                if ext in _VIDEO_EXTS:
                    continue
                while len(_face_cache) > 200:
                    if cancel_event and cancel_event.is_set():
                        return
                    time.sleep(0.5)
                try:
                    # 避免对已被其他线程缓存的数据重复提取
                    with _face_cache_lock:
                        if img_path in _face_cache:
                            continue
                    
                    image_bytes = get_image_frame_bytes(img_path)
                    if not image_bytes:
                        logger.info("[GPU流水线] 跳过 0KB/空数据文件: %s", img_path)
                        continue
                    faces = _extract_faces_with_resize(image_bytes, raise_on_error=True)
                    with _face_cache_lock:
                        _face_cache[img_path] = faces
                    _pipeline_count += 1
                    if _pipeline_count % 50 == 0:
                        logger.info("[GPU流水线] 已预提取 %d 张图片的人脸特征", _pipeline_count)
                except Exception as exc:
                    logger.warning("[GPU流水线] 人脸提取失败 %s: %s", img_path, exc)
            if _pipeline_count > 0:
                logger.info("[GPU流水线] 预提取完成，共处理 %d 张图片", _pipeline_count)

        threading.Thread(target=_face_pipeline, daemon=True, name="face-pipeline").start()
        logger.info("GPU 人脸提取流水线已启动")

    # ── 共用常量 ──
    _bar_fmt = "{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    _SKIPPED_SENTINEL = {"__skipped__": True}
    
    _face_queue = queue.Queue()
    _face_p2_completed_event = threading.Event()
    
    for item in face_todo:
        _face_queue.put(item)
        
    _face_found_lock = threading.Lock()
    _face_found = 0

    # ── Phase 2: 人物扫描（后台线程，与 Phase 1 真正并行） ──
    _phase2_threads = []
    _face_bar = None

    if enable_face_scan:
        _face_bar = tqdm(
            total=len(face_todo) + len(to_analyze), desc=f"[{FACE_SCAN_LABEL}]",
            file=_face_writer or sys.stderr,
            bar_format=_bar_fmt, leave=True, mininterval=0.3, ascii=True,
        )

        def _face_worker_loop():
            nonlocal _face_found
            p2_storage = Storage(db_path)
            try:
                while True:
                    if _wait_if_paused(cancel_event, pause_event):
                        break
                    if cancel_event and cancel_event.is_set():
                        break
                    
                    try:
                        item = _face_queue.get(timeout=0.5)
                    except queue.Empty:
                        if _face_p2_completed_event.is_set():
                            break
                        continue
                        
                    if item is None:
                        _face_queue.task_done()
                        break
                        
                    img_path, md5 = item
                    fname = Path(img_path).name
                    logger.info("开始人物扫描: %s", fname)
                    if _face_bar:
                        _face_bar.set_description_str(f"[{FACE_SCAN_LABEL}] {fname}", refresh=True)
                        
                    cached_faces = None
                    with _face_cache_lock:
                        cached_faces = _face_cache.pop(img_path, None)
                    if cached_faces is not None:
                        faces = cached_faces
                    else:
                        try:
                            image_bytes = get_image_frame_bytes(img_path)
                            if not image_bytes:
                                logger.info("人物扫描跳过 0KB/空数据文件: %s", img_path)
                                try:
                                    p2_storage.add_skipped_file(img_path, "人脸扫描解码失败: 读取图片帧为空")
                                except Exception as skip_exc:
                                    logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
                                if _face_bar:
                                    _face_bar.update(1)
                                _face_queue.task_done()
                                continue
                            faces = _extract_faces_with_resize(image_bytes, raise_on_error=True)
                        except Exception as exc:
                            logger.warning("人物扫描跳过异常文件 %s: %s", img_path, exc)
                            try:
                                p2_storage.add_skipped_file(img_path, f"人脸扫描解码失败: {exc}")
                            except Exception as skip_exc:
                                logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
                            faces = []

                    face_count = len(faces)
                    with _face_found_lock:
                        if face_count > 0:
                            _face_found += 1
                        found = _face_found
                    
                    logger.debug("人脸识别 %s → %d 张人脸", fname, face_count)
                    if _face_bar:
                        _face_bar.set_description_str(
                            f"[{FACE_SCAN_LABEL}] ({found}张有人脸) {fname}", refresh=False,
                        )
                        _face_bar.update(1)

                    try:
                        cur = p2_storage._conn.cursor()
                        cur.execute("SELECT id FROM images WHERE md5 = ?", (md5,))
                        row = cur.fetchone()
                        if row:
                            p2_storage.delete_face_embeddings_for_image(row[0])
                            if faces:
                                p2_storage.add_face_embeddings(row[0], faces)
                            p2_storage.mark_face_scanned(row[0])
                    except Exception as e:
                        logger.error("更新人脸数据失败 %s: %s", img_path, e)
                        
                    _face_queue.task_done()
            finally:
                p2_storage.close()

        logger.info("Phase 2 人物扫描已在后台启动，并行工作线程数: %d", workers)
        for _ in range(workers):
            t = threading.Thread(target=_face_worker_loop, daemon=True)
            t.start()
            _phase2_threads.append(t)

    # ── Phase 1: 图片/视频信息提取 ──
    circuit_breaker_err = None
    if to_analyze:
        _llm_bar = tqdm(
            total=len(to_analyze), desc=f"[{IMAGE_EXTRACTION_LABEL}]",
            file=_llm_writer or sys.stderr,
            bar_format=_bar_fmt, leave=True, mininterval=0.3, ascii=True,
        )

        async def _llm_worker(img_path: str, session: aiohttp.ClientSession, sem: asyncio.Semaphore, video_sem: asyncio.Semaphore, io_sem: asyncio.Semaphore) -> tuple[str, dict | list[dict] | None]:
            async with sem:
                if await _wait_if_paused_async(cancel_event, pause_event):
                    return img_path, None
                if cancel_event and cancel_event.is_set():
                    return img_path, None

                async def _try_attach_embedding(llm_res: dict):
                    if embedding_model and "error" not in llm_res:
                        parts = []
                        if llm_res.get("scene"):
                            parts.append(str(llm_res["scene"]))
                        if llm_res.get("objects"):
                            obj = llm_res["objects"]
                            parts.append(", ".join(obj) if isinstance(obj, list) else str(obj))
                        text_to_embed = " ".join(parts).strip()
                        if text_to_embed:
                            async def _encode_embedding() -> list[float]:
                                """根据当前 embedding 配置获取单条文本向量。"""

                                return await get_text_embedding_async(
                                    text=text_to_embed,
                                    model=embedding_model,
                                    backend=embedding_backend,
                                    session=session,
                                    api_key=embedding_api_key,
                                    base_url=embedding_base_url,
                                    timeout=timeout,
                                    rate_limiter=rate_limiter,
                                    workers=embedding_workers,
                                )

                            if embedding_backend == "local" and embedding_sem is not None:
                                async with embedding_sem:
                                    emb = await _encode_embedding()
                            else:
                                emb = await _encode_embedding()
                            if emb:
                                llm_res["text_embedding"] = emb

                ext = Path(img_path).suffix.lower()
                fname = Path(img_path).name
                logger.info("开始图片/视频信息提取: %s", fname)
                _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] {fname}", refresh=True)

                # ---- 视频多帧路径 ----
                if ext in _VIDEO_EXTS:
                    if model_supports_video and transcoded_video_path:
                        # 尝试通过基础扫描路径提取相对目录结构，以便精准匹配转码后的子目录文件
                        rel_dir = ""
                        for base_p in paths:
                            try:
                                # 找到当前文件属于哪个指定的扫描根目录
                                if Path(img_path).is_relative_to(Path(base_p)):
                                    rel_dir = Path(img_path).parent.relative_to(Path(base_p))
                                    break
                            except ValueError:
                                pass
                                
                        original_name = Path(img_path).stem
                        # 优先尝试带相对目录路径的文件 (兼容 video_compression.py 保持目录结构的逻辑)
                        transcoded_file_with_dir = Path(transcoded_video_path) / rel_dir / f"{original_name}.mp4"
                        transcoded_file_root = Path(transcoded_video_path) / f"{original_name}.mp4"
                        
                        transcoded_file = transcoded_file_with_dir if transcoded_file_with_dir.exists() else transcoded_file_root

                        if transcoded_file.exists():
                            file_metadata = extract_metadata(img_path)
                            # 因为是直接传整个视频给大模型，MD5 可以用原视频的 MD5 来作为记录标识
                            try:
                                async with io_sem:
                                    file_md5 = await asyncio.to_thread(compute_file_md5_chunked, img_path)
                                    file_sha256 = await asyncio.to_thread(compute_file_sha256_chunked, img_path)
                            except OSError:
                                return img_path, None

                            with _md5_lock:
                                if file_md5 in known_md5s:
                                    _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] ✓ {fname}", refresh=False)
                                    _llm_bar.update(1)
                                    if _face_bar:
                                        _face_bar.update(1)
                                    return img_path, [_SKIPPED_SENTINEL]
                                known_md5s.add(file_md5)
                                if enable_face_scan:
                                    face_scanned_md5s.add(file_md5)
                                    
                            logger.info("分析转码后视频文件 %s", fname)
                            try:
                                def _read_video():
                                    with open(transcoded_file, "rb") as f:
                                        return f.read()
                                video_bytes = await asyncio.to_thread(_read_video)
                                
                                file_size_mb = len(video_bytes) / 1024 / 1024
                                if not video_bytes:
                                    raise ValueError(f"转码后的视频文件为空(0 bytes): {transcoded_file}")
                                if len(video_bytes) < 10240: # 小于 10KB
                                    raise ValueError(f"转码后的视频文件极小({len(video_bytes)} bytes)，文件可能已损坏或转码失败: {transcoded_file}")
                                    
                                logger.info("推送视频给大模型: %s (原文件: %s, 转码文件: %s, 大小: %.2f MB)", 
                                            fname, img_path, transcoded_file, file_size_mb)
                                
                                async with video_sem:
                                    llm_result = await async_call_image_model(
                                        image_bytes=video_bytes,
                                        session=session,
                                        image_format="mp4",
                                        api_key=api_key,
                                        base_url=base_url,
                                        model=model,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        timeout=timeout,
                                        rate_limiter=rate_limiter,
                                    )
                                await _try_attach_embedding(llm_result)
                                record = dict(file_metadata)
                                record["md5"] = file_md5
                                record["sha256"] = file_sha256
                                record["path"] = img_path
                                record["file_name"] = fname
                                record.update(llm_result)
                                record["analyzed_at"] = datetime.now().isoformat()
                                
                                _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] {fname}", refresh=False)
                                _llm_bar.update(1)
                                return img_path, [record]
                            except Exception as exc:
                                logger.error("转码视频分析失败 %s: %s", img_path, exc)
                                _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] 失败 {fname}", refresh=False)
                                _llm_bar.update(1)
                                if _face_bar:
                                    _face_bar.update(1)
                                return img_path, {"__failed__": True, "reason": str(exc)}

                    # 对于极端 CPU 耗时的抽帧操作，必须限制其瞬时并发，防止耗尽默认 ThreadPoolExecutor 导致死锁
                    try:
                        async with video_sem:
                            frames = await asyncio.to_thread(extract_video_frames, img_path)
                    except Exception as exc:
                        logger.error("视频抽帧失败 %s: %s", img_path, exc)
                        _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] 失败 {fname}", refresh=False)
                        _llm_bar.update(1)
                        if _face_bar:
                            _face_bar.update(1)
                        return img_path, {"__failed__": True, "reason": str(exc)}

                    file_metadata = extract_metadata(img_path)
                    file_md5 = compute_file_md5_chunked(img_path)
                    file_sha256 = compute_file_sha256_chunked(img_path)
                    results: list[dict] = []

                    for ts, frame_bytes in frames:
                        if await _wait_if_paused_async(cancel_event, pause_event):
                            break
                        if cancel_event and cancel_event.is_set():
                            break
                        frame_md5 = compute_md5(frame_bytes)
                        with _md5_lock:
                            if frame_md5 in known_md5s:
                                results.append(_SKIPPED_SENTINEL)
                                continue
                            known_md5s.add(frame_md5)
                            if enable_face_scan:
                                face_scanned_md5s.add(frame_md5)

                        ts_label = f"{ts:.1f}s"
                        
                        def _cpu_video_prep():
                            return compress_for_upload(frame_bytes, max_dim=1120)
                            
                        compressed, fmt = await asyncio.to_thread(_cpu_video_prep)
                        
                        logger.info("视频帧分析 %s @ %s", fname, ts_label)
                        try:
                            llm_result = await async_call_image_model(
                                image_bytes=compressed,
                                session=session,
                                image_format=fmt,
                                api_key=api_key,
                                base_url=base_url,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                timeout=timeout,
                                rate_limiter=rate_limiter,
                            )
                            await _try_attach_embedding(llm_result)
                        except Exception as exc:
                            logger.error("视频帧分析失败 %s@%s: %s", img_path, ts_label, exc)
                            continue

                        record = dict(file_metadata)
                        record["md5"] = frame_md5
                        record["sha256"] = file_sha256
                        record["path"] = f"{img_path}#t={ts_label}"
                        record["file_name"] = f"{Path(img_path).stem}_{ts_label}{Path(img_path).suffix}"
                        record.update(llm_result)
                        record["analyzed_at"] = datetime.now().isoformat()
                        results.append(record)

                    _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] {fname}", refresh=False)
                    _llm_bar.update(1)
                    return img_path, results if results else None

                    # ---- 普通图片路径 ----
                try:
                    # 限定 IO 型和 CPU 轻量 Hash 操作的最大并发以免占用太多句柄
                    async with io_sem:
                        md5 = await asyncio.to_thread(compute_file_md5_chunked, img_path)
                except (OSError, PermissionError) as exc:
                    logger.warning("无法读取文件 %s: %s", img_path, exc)
                    _llm_bar.update(1)
                    if _face_bar:
                        _face_bar.update(1)
                    return img_path, None

                need_llm = True
                need_face = enable_face_scan

                with _md5_lock:
                    if md5 in known_md5s:
                        need_llm = False
                    if need_face and md5 in face_scanned_md5s:
                        need_face = False
                    if need_llm:
                        known_md5s.add(md5)
                    if need_face:
                        face_scanned_md5s.add(md5)

                # 完全跳过
                if not need_llm and not need_face:
                    _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] ✓ {fname}", refresh=False)
                    _llm_bar.update(1)
                    if _face_bar:
                        _face_bar.update(1)
                    return img_path, _SKIPPED_SENTINEL

                # LLM 已做过，人脸待处理 → 推迟到 Phase 2 处理
                if not need_llm and need_face:
                    _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] 已有 {fname}", refresh=False)
                    _llm_bar.update(1)
                    
                    # 放在主线程进行 put，这里标记为需要 face 但无需 llm
                    return img_path, {"__needs_face_only__": True, "md5": md5}

                # 正常 LLM 分析（人脸推迟到 Phase 2 补充批次）
                try:
                    result = await _analyze_single_async(
                        path=img_path,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        session=session,
                        rate_limiter=rate_limiter,
                        enable_face_scan=False,
                        embedding_api_key=embedding_api_key,
                        embedding_base_url=embedding_base_url,
                        embedding_model=embedding_model,
                        embedding_backend=embedding_backend,
                        embedding_workers=embedding_workers,
                        embedding_semaphore=embedding_sem,
                        cancel_event=cancel_event,
                        pause_event=pause_event,
                    )
                    if isinstance(result, dict) and result.get("__cancelled__"):
                        if _face_bar:
                            _face_bar.update(1)
                        _llm_bar.update(1)
                        return img_path, None

                    _llm_bar.set_description_str(f"[{IMAGE_EXTRACTION_LABEL}] {fname}", refresh=False)
                    _llm_bar.update(1)

                    if need_face and enable_face_scan:
                        # 不在这里 put，在主线程 upsert 后统一 put
                        result["__needs_face_after_upsert__"] = True
                        result["md5"] = result.get("md5", md5)
                    else:
                        if _face_bar:
                            _face_bar.update(1)

                    return img_path, result
                except Exception as exc:
                    logger.error("分析失败 %s: %s", img_path, exc, exc_info=False)
                    _llm_bar.update(1)
                    if _face_bar:
                        _face_bar.update(1)
                    return img_path, {"__failed__": True, "reason": str(exc)}

        async def _run_async_scan():
            nonlocal circuit_breaker_err, embedding_sem
            # 连接数池放大以支持大量并发
            import socket
            connector = aiohttp.TCPConnector(limit=100, family=socket.AF_INET)
            
            video_sem = asyncio.Semaphore(max(1, video_workers))
            io_sem = asyncio.Semaphore(20)
            embedding_sem = asyncio.Semaphore(max(1, embedding_workers)) if embedding_backend == "local" and embedding_model else None
            
            async with aiohttp.ClientSession(connector=connector) as session:
                sem = asyncio.Semaphore(workers) # 使用用户配置的并发数
                tasks = {asyncio.create_task(_llm_worker(p, session, sem, video_sem, io_sem)) for p in to_analyze}

                def _handle_future(future: asyncio.Future) -> bool:
                    nonlocal circuit_breaker_err
                    try:
                        img_path, result = future.result()
                    except asyncio.CancelledError:
                        return False
                    except Exception as e:
                        logger.error("意外错误 %s", e)
                        return False
                    if isinstance(result, list):
                        for r in result:
                            # 提前检查是否是熔断指示
                            if r is not _SKIPPED_SENTINEL and isinstance(r, dict) and r.get("__circuit_breaker_open__"):
                                circuit_breaker_err = r.get("error", "API连通性持续失败，触发断路器熔断")
                                logger.error("检测到 API 熔断，已提前终止后续扫描。")
                                if cancel_event:
                                    cancel_event.set()
                                return True

                            stats.current += 1
                            if r is _SKIPPED_SENTINEL:
                                stats.skipped += 1
                            elif "error" in r:
                                stats.failed += 1
                                logger.error("API 错误 %s: %s (返回: %s)", img_path, r["error"], r.get("llm_raw", ""))
                            else:
                                if enable_face_scan:
                                    r["face_scanned"] = True
                                image_id = storage.upsert(r)
                                stats.processed += 1
                    else:
                        if result is not _SKIPPED_SENTINEL and result is not None and isinstance(result, dict) and result.get("__circuit_breaker_open__"):
                            circuit_breaker_err = result.get("error", "API连通性持续失败，触发断路器熔断")
                            logger.error("检测到 API 熔断，已提前终止后续扫描。")
                            if cancel_event:
                                cancel_event.set()
                            return True

                        stats.current += 1
                        if result is _SKIPPED_SENTINEL:
                            stats.skipped += 1
                        elif result is None:
                            stats.failed += 1
                        elif isinstance(result, dict) and result.get("__failed__"):
                            stats.failed += 1
                            storage.add_skipped_file(img_path, result["reason"])
                        elif isinstance(result, dict) and result.get("__needs_face_only__"):
                            if enable_face_scan:
                                _face_queue.put((img_path, result["md5"]))
                        elif "error" in result:
                            stats.failed += 1
                            logger.error("API 错误 %s: %s (返回: %s)", img_path, result["error"], result.get("llm_raw", ""))
                        else:
                            needs_face = result.pop("__needs_face_after_upsert__", False)
                            image_id = storage.upsert(result)
                            stats.processed += 1
                            if needs_face and enable_face_scan:
                                _face_queue.put((img_path, result.get("md5")))
                    return False

                async def _wait_for_cancel() -> bool:
                    while cancel_event is not None and not cancel_event.is_set():
                        await asyncio.sleep(0.1)
                    return cancel_event is not None and cancel_event.is_set()

                cancel_task = asyncio.create_task(_wait_for_cancel()) if cancel_event is not None else None
                pending = set(tasks)

                while pending:
                    if await _wait_if_paused_async(cancel_event, pause_event):
                        for t in pending:
                            t.cancel()
                        if cancel_task:
                            cancel_task.cancel()
                        logger.warning("图片/视频信息提取已被用户中止，已分析的数据已保存。")
                        break

                    wait_set = set(pending)
                    if cancel_task:
                        wait_set.add(cancel_task)
                    done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                    if cancel_task:
                        pending.discard(cancel_task)

                    if cancel_task and cancel_task in done:
                        done.discard(cancel_task)
                        stop_due_to_circuit = False
                        for future in done:
                            if _handle_future(future):
                                stop_due_to_circuit = True
                        for t in pending:
                            t.cancel()
                        logger.warning("图片/视频信息提取已被用户中止，已分析的数据已保存。")
                        break

                    stop_due_to_circuit = False
                    for future in done:
                        if future is cancel_task:
                            continue
                        if _handle_future(future):
                            stop_due_to_circuit = True
                    if stop_due_to_circuit:
                        for t in pending:
                            t.cancel()
                        break

                if cancel_task:
                    cancel_task.cancel()

        # 启动异步协程执行并阻塞等待结束
        try:
            asyncio.run(_run_async_scan())
        except KeyboardInterrupt:
            logger.warning("用户中断 (Ctrl+C)，已分析的数据已保存。")

        _llm_bar.close()

        t_llm_end = _time.perf_counter()
        llm_cost = t_llm_end - t_start

        if circuit_breaker_err:
            logger.error("扫描因 API 故障中止: %s", circuit_breaker_err)
            if _llm_writer:
                _llm_writer.write(f"[{IMAGE_EXTRACTION_LABEL}] ❌ 扫描中止 — {circuit_breaker_err}\n")
        else:
            logger.info(
                "图片/视频信息提取完成(耗时 %.1fs) — 总计: %d  新增/更新: %d  跳过: %d  失败: %d",
                llm_cost, stats.total, stats.processed, stats.skipped, stats.failed,
            )
            if _llm_writer:
                _llm_writer.write(
                    f"[{IMAGE_EXTRACTION_LABEL}] ✓ 完成 — 扫描 {stats.total} 张 "
                    f"(新增 {stats.processed}, 跳过 {stats.skipped})\n"
                )
    else:
        t_llm_end = _time.perf_counter()
        llm_cost = t_llm_end - t_start
        logger.info("图片/视频信息提取: 全部 %d 张已完成，无需 LLM 分析 (耗时 %.1fs)", stats.total, llm_cost)
        if _llm_writer:
            _llm_writer.write(
                f"[{IMAGE_EXTRACTION_LABEL}] ✓ 完成 — 全部 {stats.total} 张已就绪\n"
            )

    # ── 等待 Phase 2 后台线程完成 ──
    _face_p2_completed_event.set()
    if enable_face_scan:
        for _ in range(workers):
            _face_queue.put(None)

    for t in locals().get('_phase2_threads', []):
        try:
            t.join()
        except:
            pass

    _cancelled = cancel_event and cancel_event.is_set()
    t_face_end = _time.perf_counter()
    total_face_cost = t_face_end - t_start

    if _face_bar:
        _face_bar.close()
        if _cancelled:
            logger.info("人物扫描已中止(总耗时 %.1fs) — 已处理 %d 张", total_face_cost, _face_found)
            if _face_writer:
                _face_writer.write(f"[{FACE_SCAN_LABEL}] 已中止\n")
        else:
            logger.info("人物扫描完成(总耗时 %.1fs) — 共 %d 张有人脸", total_face_cost, _face_found)
            if _face_writer:
                _face_writer.write(
                    f"[{FACE_SCAN_LABEL}] ✓ 完成 — 共 {_face_found} 张有人脸，正在聚类...\n"
                )
    elif enable_face_scan and _face_writer:
        if _cancelled:
            _face_writer.write(f"[{FACE_SCAN_LABEL}] 已中止\n")
        else:
            _face_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 全部 {stats.total} 张已完成\n")

    logger.info(
        "扫描完成 — 总计: %d  数据库记录数: %d",
        stats.total, storage.count(),
    )

    # ── Phase 3: 清理已删除的文件 (Cleanup) ──
    try:
        if not _cancelled:
            all_db_paths = list(known_paths.keys())
            to_delete = []
            _path_exists_cache = {}
            
            resolved_scan_parts = []
            for p in paths:
                try:
                    resolved_scan_parts.append(Path(p).resolve().parts)
                except (OSError, ValueError):
                    pass

            for db_path in all_db_paths:
                base_path = db_path.split("#t=")[0]
                
                try:
                    base_parts = Path(base_path).resolve().parts
                except (OSError, ValueError):
                    continue
                    
                in_scope = False
                for scan_parts in resolved_scan_parts:
                    if base_parts[:len(scan_parts)] == scan_parts:
                        in_scope = True
                        break
                        
                if not in_scope:
                    continue
                    
                if base_path not in _path_exists_cache:
                    _path_exists_cache[base_path] = Path(base_path).exists()
                    
                if not _path_exists_cache[base_path]:
                    to_delete.append(db_path)
                    
            if to_delete:
                logger.info("发现 %d 条失效的数据库记录，正在清理...", len(to_delete))
                if _llm_writer:
                    _llm_writer.write(f"[清理记录] 正在清理 {len(to_delete)} 条失效的数据库记录...\n")
                
                deleted_count = storage.delete_by_paths(to_delete)
                logger.info("清理完成，成功删除 %d 条记录。", deleted_count)
                if _llm_writer:
                    _llm_writer.write(f"[清理记录] ✓ 成功清理 {deleted_count} 条失效记录\n")
    except Exception as e:
        logger.error("清理失效记录时发生错误: %s", e)

    if not _cancelled:
        _do_clustering()

    storage.close()
    
    t_total_end = _time.perf_counter()
    total_cost = t_total_end - t_start
    logger.info("整个扫描流程完成，总耗时: %.1fs", total_cost)

    if 'circuit_breaker_err' in locals() and circuit_breaker_err:
        raise RuntimeError(circuit_breaker_err)
        
    return {
        "total": stats.total,
        "processed": stats.processed,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "llm_cost": llm_cost if 'llm_cost' in locals() else 0.0,
        "face_cost": total_face_cost if 'total_face_cost' in locals() else 0.0,
        "total_cost": total_cost,
        "face_found": _face_found if '_face_found' in locals() else 0
    }


def _build_face_scan_record(path: str, md5: str, size_bytes: int) -> dict:
    """构造仅用于人物扫描入库的最小图片记录。"""

    return {
        "path": path,
        "file_name": Path(path).name,
        "size_bytes": size_bytes,
        "md5": md5,
        "face_scanned": True,
    }


def _cluster_faces_for_people_scan(storage: Storage, progress_writer: IO | None) -> None:
    """对当前库中的全部人脸执行聚类，并将结果同步到 persons 表。"""

    from photo_identify.face_manager import cluster_face_embeddings
    import numpy as np

    logger.info("开始聚类人物关系...")
    if progress_writer:
        progress_writer.write(f"[{FACE_SCAN_LABEL}] 正在聚类相似人脸...\n")

    all_faces = storage.get_all_faces()
    if not all_faces:
        logger.info("未发现任何人脸数据，跳过聚类。")
        if progress_writer:
            progress_writer.write(f"[{FACE_SCAN_LABEL}] 未发现人脸数据\n")
        return

    embeddings_with_ids = []
    for face_id, embedding_bytes in all_faces:
        embeddings_with_ids.append((face_id, np.frombuffer(embedding_bytes, dtype=np.float32)))

    t_start = time.perf_counter()
    cluster_mapping = cluster_face_embeddings(embeddings_with_ids)
    storage.update_face_clusters(cluster_mapping)
    n_persons = len(set(cluster_id for cluster_id in cluster_mapping.values() if cluster_id >= 0))
    elapsed = time.perf_counter() - t_start
    logger.info("人物聚类完成(耗时 %.1fs)，共识别出 %d 位主要人物", elapsed, n_persons)
    if progress_writer:
        progress_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 聚类完成(耗时 {elapsed:.1f}s) — {n_persons} 位人物\n")


def scan_faces(
    paths: list[str],
    db_path: str,
    workers: int = DEFAULT_WORKERS,
    extensions: frozenset[str] = DEFAULT_IMAGE_EXTENSIONS,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    progress_writer: IO | None = None,
) -> dict:
    """独立执行人物扫描：仅提取人脸、写入数据库并做聚类，不调用图片理解模型。"""

    import concurrent.futures

    t_start = time.perf_counter()
    storage = Storage(db_path)
    stats = ScanStats()
    _video_exts = {".mp4", ".mov", ".avi", ".mkv"}

    from photo_identify.face_manager import get_face_app, get_device_mode

    get_face_app()
    logger.info("人脸引擎已启动，当前模式: %s", get_device_mode())

    all_images: list[str] = []
    for path in paths:
        if _wait_if_paused(cancel_event, pause_event):
            break
        found = list_images(path, extensions)
        logger.info("目录 %s 发现 %d 个候选文件", path, len(found))
        all_images.extend(found)

    if not all_images:
        logger.info("未发现任何可用于人物扫描的文件。")
        if progress_writer:
            progress_writer.write(f"[{FACE_SCAN_LABEL}] 未发现任何图片\n")
        storage.close()
        return {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "face_found": 0,
            "total_cost": time.perf_counter() - t_start,
        }

    known_paths = storage.get_known_paths()
    face_scanned_md5s = storage.get_face_scanned_md5s()
    skipped_paths = storage.get_skipped_paths()

    candidates: list[tuple[str, str, int]] = []
    for img_path in all_images:
        if _wait_if_paused(cancel_event, pause_event):
            break
        if cancel_event and cancel_event.is_set():
            break

        stats.total += 1
        if img_path in skipped_paths:
            stats.skipped += 1
            logger.info("人物扫描跳过已标记异常文件: %s", img_path)
            if progress_writer:
                progress_writer.write(f"[{FACE_SCAN_LABEL}] 跳过异常文件: {Path(img_path).name}\n")
            continue

        ext = Path(img_path).suffix.lower()
        if ext in _video_exts:
            stats.skipped += 1
            continue

        try:
            stat_result = Path(img_path).stat()
        except OSError:
            stats.failed += 1
            continue

        if stat_result.st_size == 0:
            stats.skipped += 1
            logger.info("人物扫描跳过 0KB 文件: %s", img_path)
            if progress_writer:
                progress_writer.write(f"[{FACE_SCAN_LABEL}] 跳过 0KB 文件: {Path(img_path).name}\n")
            try:
                storage.add_skipped_file(img_path, "0KB 文件")
                md5 = compute_file_md5_chunked(img_path)
                record = _build_face_scan_record(img_path, md5, stat_result.st_size)
                image_id = storage.upsert(record)
                storage.mark_face_scanned(image_id)
                face_scanned_md5s.add(md5)
            except Exception as exc:
                logger.warning("人物扫描写入 0KB 跳过标记失败 %s: %s", img_path, exc)
            continue

        cached = known_paths.get(img_path)
        if cached:
            cached_size, cached_mtime, cached_md5 = cached
            if cached_size == stat_result.st_size and cached_mtime == datetime.fromtimestamp(stat_result.st_mtime).isoformat():
                if cached_md5 in face_scanned_md5s:
                    stats.skipped += 1
                    continue
                candidates.append((img_path, cached_md5, stat_result.st_size))
                continue

        try:
            md5 = compute_file_md5_chunked(img_path)
        except Exception as exc:
            stats.failed += 1
            logger.warning("人物扫描计算 MD5 失败 %s: %s", img_path, exc)
            continue

        if md5 in face_scanned_md5s:
            stats.skipped += 1
            logger.info("人物扫描跳过重复内容文件: %s", img_path)
            if progress_writer:
                progress_writer.write(f"[{FACE_SCAN_LABEL}] 跳过重复内容文件: {Path(img_path).name}\n")
            try:
                storage.add_skipped_file(img_path, "重复文件或MD5已扫描")
            except Exception as skip_exc:
                logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
            continue

        candidates.append((img_path, md5, stat_result.st_size))

    if cancel_event and cancel_event.is_set():
        storage.close()
        return {
            "total": stats.total,
            "processed": stats.processed,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "face_found": 0,
            "total_cost": time.perf_counter() - t_start,
        }

    if not candidates:
        logger.info("没有需要执行人物扫描的图片。")
        if progress_writer:
            progress_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 全部 {stats.total} 张已完成\n")
        _cluster_faces_for_people_scan(storage, progress_writer)
        storage.close()
        return {
            "total": stats.total,
            "processed": 0,
            "skipped": stats.skipped,
            "failed": stats.failed,
            "face_found": 0,
            "total_cost": time.perf_counter() - t_start,
        }

    progress_bar = tqdm(
        total=len(candidates),
        desc=f"[{FACE_SCAN_LABEL}]",
        file=progress_writer or sys.stderr,
        bar_format="{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        leave=True,
        mininterval=0.3,
        ascii=True,
    )

    def _scan_single_face(item: tuple[str, str, int]) -> dict:
        """处理单张图片的人脸提取和入库。"""

        img_path, known_md5, size_bytes = item
        if _wait_if_paused(cancel_event, pause_event):
            return {"cancelled": True, "path": img_path}
        if cancel_event and cancel_event.is_set():
            return {"cancelled": True, "path": img_path}

        worker_storage = Storage(db_path)
        try:
            if size_bytes == 0:
                logger.info("人物扫描跳过 0KB 文件: %s", img_path)
                try:
                    worker_storage.add_skipped_file(img_path, "0KB 文件")
                except Exception as skip_exc:
                    logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
                return {
                    "path": img_path,
                    "skipped": True,
                }

            md5 = known_md5 or compute_file_md5_chunked(img_path)
            record = _build_face_scan_record(img_path, md5, size_bytes)
            image_id = worker_storage.upsert(record)

            image_bytes = get_image_frame_bytes(img_path)
            if not image_bytes:
                logger.info("人物扫描跳过 0KB/空数据文件: %s", img_path)
                try:
                    worker_storage.add_skipped_file(img_path, "人脸扫描解码失败: 读取图片帧为空")
                except Exception as skip_exc:
                    logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
                return {
                    "path": img_path,
                    "skipped": True,
                }

            faces = _extract_faces_with_resize(image_bytes, raise_on_error=True)

            worker_storage.delete_face_embeddings_for_image(image_id)
            if faces:
                worker_storage.add_face_embeddings(image_id, faces)
            worker_storage.mark_face_scanned(image_id)

            return {
                "path": img_path,
                "face_count": len(faces),
            }
        except Exception as exc:
            logger.warning("人物扫描跳过异常文件 %s: %s", img_path, exc)
            try:
                worker_storage.add_skipped_file(img_path, f"人脸扫描解码失败: {exc}")
            except Exception as skip_exc:
                logger.warning("记录人物扫描跳过失败 %s: %s", img_path, skip_exc)
            return {
                "path": img_path,
                "error": str(exc),
            }
        finally:
            worker_storage.close()

    face_found = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            future_to_path = {
                executor.submit(_scan_single_face, item): item[0]
                for item in candidates
            }
            for future in concurrent.futures.as_completed(future_to_path):
                if cancel_event and cancel_event.is_set():
                    break

                result = future.result()
                fname = Path(result.get("path", future_to_path[future])).name
                progress_bar.set_description_str(f"[{FACE_SCAN_LABEL}] {fname}", refresh=False)
                progress_bar.update(1)

                if result.get("cancelled"):
                    continue
                if result.get("skipped"):
                    stats.skipped += 1
                    continue
                if result.get("error"):
                    stats.failed += 1
                    continue

                stats.processed += 1
                if result.get("face_count", 0) > 0:
                    face_found += 1
    finally:
        progress_bar.close()

    if not (cancel_event and cancel_event.is_set()):
        _cluster_faces_for_people_scan(storage, progress_writer)

    storage.close()
    total_cost = time.perf_counter() - t_start
    logger.info(
        "人物扫描完成 — 总计: %d  处理: %d  跳过: %d  失败: %d  含人脸: %d",
        stats.total,
        stats.processed,
        stats.skipped,
        stats.failed,
        face_found,
    )
    if progress_writer:
        if cancel_event and cancel_event.is_set():
            progress_writer.write(f"[{FACE_SCAN_LABEL}] 已中止\n")
        else:
            progress_writer.write(f"[{FACE_SCAN_LABEL}] ✓ 完成 — 处理 {stats.processed} 张，含人脸 {face_found} 张\n")

    return {
        "total": stats.total,
        "processed": stats.processed,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "face_found": face_found,
        "total_cost": total_cost,
    }
