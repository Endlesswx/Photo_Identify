"""图片扫描主流程：并发遍历目录，调用 LLM 分析并入库。

负责编排整个扫描过程：
- 递归收集图片文件
- 基于 path+size+mtime 快速跳过 + MD5 精确去重
- 多线程并发调用 LLM API
- 逐条写入 SQLite（断点续扫）
- tqdm 实时进度条显示
"""

import logging
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from datetime import datetime
from pathlib import Path
from typing import IO

from tqdm import tqdm

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
    compute_file_md5_chunked,
    compute_file_sha256_chunked,
    extract_metadata,
    extract_video_frames,
    list_images,
    get_image_frame_bytes,
)
from photo_identify.llm import RateLimiter, call_image_model
from photo_identify.storage import Storage

logger = logging.getLogger(__name__)


class ScanStats:
    """扫描统计计数器（线程安全不要求精确，仅用于进度显示）。"""

    def __init__(self):
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.failed = 0
        self.current = 0


def _extract_faces_with_resize(image_bytes: bytes) -> list[dict]:
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
        return []


def _analyze_single(
    path: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    rate_limiter: RateLimiter,
    enable_face_scan: bool = False,
    face_cache: dict | None = None,
    face_cache_lock: threading.Lock | None = None,
) -> dict:
    """分析单张图片：读取→压缩→调用 LLM→合并结果。"""
    image_bytes = get_image_frame_bytes(path)
    metadata = extract_metadata(path)
    metadata["md5"] = compute_file_md5_chunked(path)
    metadata["sha256"] = compute_file_sha256_chunked(path)

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

    if enable_face_scan:
        cached_faces = None
        if face_cache is not None and face_cache_lock is not None:
            with face_cache_lock:
                cached_faces = face_cache.pop(path, None)
        if cached_faces is not None:
            metadata["faces"] = cached_faces
        else:
            metadata["faces"] = _extract_faces_with_resize(image_bytes)

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
    enable_face_scan: bool = False,
    progress_writers: tuple[IO, IO] | None = None,
):
    """执行图片扫描主流程。

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
        enable_face_scan: 是否进行人脸扫描和聚类。
        progress_writers: GUI 进度条写入器 ``(llm_writer, face_writer)``。
    """
    import time as _time
    import queue
    t_start = _time.perf_counter()

    storage = Storage(db_path)
    rate_limiter = RateLimiter(rpm_limit, tpm_limit)

    # progress writer 引用（用于 _do_clustering / 最终状态写入 GUI）
    _llm_writer = progress_writers[0] if progress_writers else None
    _face_writer = progress_writers[1] if progress_writers else None

    if enable_face_scan:
        from photo_identify.face_manager import get_face_app, get_device_mode
        get_face_app()
        logger.info("人脸引擎已启动，当前模式: %s", get_device_mode())

    # 收集所有图片路径
    all_images: list[str] = []
    for p in paths:
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
                _face_writer.write("[人物识别] 正在聚类相似人脸...\n")
            all_faces = storage.get_all_faces()
            if not all_faces:
                logger.info("未发现人脸数据，跳过聚类。")
                if _face_writer:
                    _face_writer.write("[人物识别] 未发现人脸数据\n")
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
                _face_writer.write(f"[人物识别] ✓ 聚类完成(耗时 {t_clust_cost:.1f}s) — {n_persons} 位人物\n")
        except Exception as e:
            logger.error("人脸聚类出错: %s", e)
            if _face_writer:
                _face_writer.write(f"[人物识别] 聚类出错: {e}\n")

    if not all_images:
        logger.info("未发现任何图片文件。")
        if _llm_writer:
            _llm_writer.write("[信息扫描] 未发现任何图片\n")
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
        if cancel_event and cancel_event.is_set():
            break
        stats.current += 1

        # 已处理过的视频直接跳过（视频帧不做人脸提取）
        ext = Path(img_path).suffix.lower()
        if ext in _VIDEO_EXTS and img_path in known_video_bases:
            stats.skipped += 1
            continue

        # 曾经持续失败的文件直接跳过
        if img_path in skipped_paths:
            stats.skipped += 1
            continue

        cached = known_paths.get(img_path)
        if cached:
            try:
                st = Path(img_path).stat()
                cached_size, cached_mtime, cached_md5 = cached
                if cached_size == st.st_size and cached_mtime == datetime.fromtimestamp(st.st_mtime).isoformat():
                    if enable_face_scan and cached_md5 not in face_scanned_md5s:
                        # LLM 已完成，仅需人脸提取 → 直接进入 Phase 2
                        face_todo.append((img_path, cached_md5))
                        face_scanned_md5s.add(cached_md5)
                    stats.skipped += 1
                    continue
            except OSError:
                pass
        to_analyze.append(img_path)

    if cancel_event and cancel_event.is_set():
        _do_clustering()
        storage.close()
        return

    if not to_analyze and not face_todo:
        logger.info("所有 %d 张图片均已分析，无需处理。", stats.total)
        if _llm_writer:
            _llm_writer.write(f"[信息扫描] ✓ 全部 {stats.total} 张已完成\n")
        if enable_face_scan and _face_writer:
            _face_writer.write(f"[人物识别] ✓ 全部 {stats.total} 张已完成\n")
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
                    faces = _extract_faces_with_resize(image_bytes)
                    with _face_cache_lock:
                        _face_cache[img_path] = faces
                    _pipeline_count += 1
                    if _pipeline_count % 50 == 0:
                        logger.info("[GPU流水线] 已预提取 %d 张图片的人脸特征", _pipeline_count)
                except Exception:
                    pass
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

    # ── Phase 2: 人物识别（后台线程，与 Phase 1 真正并行） ──
    _phase2_threads = []
    _face_bar = None

    if enable_face_scan:
        _face_bar = tqdm(
            total=len(face_todo) + len(to_analyze), desc="[人物识别]",
            file=_face_writer or sys.stderr,
            bar_format=_bar_fmt, leave=True, mininterval=0.3, ascii=True,
        )

        def _face_worker_loop():
            nonlocal _face_found
            p2_storage = Storage(db_path)
            try:
                while True:
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
                    cached_faces = None
                    with _face_cache_lock:
                        cached_faces = _face_cache.pop(img_path, None)
                    if cached_faces is not None:
                        faces = cached_faces
                    else:
                        try:
                            image_bytes = get_image_frame_bytes(img_path)
                            faces = _extract_faces_with_resize(image_bytes)
                        except Exception as exc:
                            logger.error("人脸提取失败 %s: %s", img_path, exc)
                            faces = []
                            
                    face_count = len(faces)
                    with _face_found_lock:
                        if face_count > 0:
                            _face_found += 1
                        found = _face_found
                    
                    logger.debug("人脸识别 %s → %d 张人脸", fname, face_count)
                    if _face_bar:
                        _face_bar.set_description_str(
                            f"[人物识别] ({found}张有人脸) {fname}", refresh=False,
                        )
                        _face_bar.update(1)

                    try:
                        cur = p2_storage._conn.cursor()
                        cur.execute("SELECT id FROM images WHERE md5 = ?", (md5,))
                        row = cur.fetchone()
                        if row:
                            if faces:
                                p2_storage.add_face_embeddings(row[0], faces)
                            p2_storage.mark_face_scanned(row[0])
                    except Exception as e:
                        logger.error("更新人脸数据失败 %s: %s", img_path, e)
                        
                    _face_queue.task_done()
            finally:
                p2_storage.close()

        logger.info("Phase 2 人物识别已在后台启动，并行工作线程数: %d", workers)
        for _ in range(workers):
            t = threading.Thread(target=_face_worker_loop, daemon=True)
            t.start()
            _phase2_threads.append(t)

    # ── Phase 1: 信息扫描 (LLM 分析) ──
    circuit_breaker_err = None
    if to_analyze:
        _llm_bar = tqdm(
            total=len(to_analyze), desc="[信息扫描]",
            file=_llm_writer or sys.stderr,
            bar_format=_bar_fmt, leave=True, mininterval=0.3, ascii=True,
        )

        def _llm_worker(img_path: str) -> tuple[str, dict | list[dict] | None]:
            if cancel_event and cancel_event.is_set():
                return img_path, None

            ext = Path(img_path).suffix.lower()
            fname = Path(img_path).name

            # ---- 视频多帧路径 ----
            if ext in _VIDEO_EXTS:
                try:
                    frames = extract_video_frames(img_path, video_frame_interval)
                except Exception as exc:
                    logger.error("视频抽帧失败 %s: %s", img_path, exc)
                    _llm_bar.set_description_str(f"[信息扫描] 失败 {fname}", refresh=False)
                    _llm_bar.update(1)
                    if _face_bar:
                        _face_bar.update(1)
                    return img_path, {"__failed__": True, "reason": str(exc)}

                file_metadata = extract_metadata(img_path)
                file_md5 = compute_file_md5_chunked(img_path)
                file_sha256 = compute_file_sha256_chunked(img_path)
                results: list[dict] = []

                for ts, frame_bytes in frames:
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
                    compressed, fmt = compress_for_upload(frame_bytes)
                    logger.info("视频帧分析 %s @ %s", fname, ts_label)
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

                    record = dict(file_metadata)
                    record["md5"] = frame_md5
                    record["sha256"] = file_sha256
                    record["path"] = f"{img_path}#t={ts_label}"
                    record["file_name"] = f"{Path(img_path).stem}_{ts_label}{Path(img_path).suffix}"
                    record.update(llm_result)
                    record["analyzed_at"] = datetime.now().isoformat()
                    results.append(record)

                _llm_bar.set_description_str(f"[信息扫描] {fname}", refresh=False)
                _llm_bar.update(1)
                return img_path, results if results else None

                # ---- 普通图片路径 ----
            try:
                md5 = compute_file_md5_chunked(img_path)
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
                _llm_bar.set_description_str(f"[信息扫描] ✓ {fname}", refresh=False)
                _llm_bar.update(1)
                if _face_bar:
                    _face_bar.update(1)
                return img_path, _SKIPPED_SENTINEL

            # LLM 已做过，人脸待处理 → 推迟到 Phase 2 处理
            if not need_llm and need_face:
                _llm_bar.set_description_str(f"[信息扫描] 已有 {fname}", refresh=False)
                _llm_bar.update(1)
                
                # 放在主线程进行 put，这里标记为需要 face 但无需 llm
                return img_path, {"__needs_face_only__": True, "md5": md5}

            # 正常 LLM 分析（人脸推迟到 Phase 2 补充批次）
            try:
                result = _analyze_single(
                    path=img_path,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    rate_limiter=rate_limiter,
                    enable_face_scan=False,
                )

                _llm_bar.set_description_str(f"[信息扫描] {fname}", refresh=False)
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

        # Phase 1 并发执行
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_llm_worker, p): p for p in to_analyze}
                for future in as_completed(futures):
                    if cancel_event and cancel_event.is_set():
                        for f in futures:
                            f.cancel()
                        logger.warning("扫描已被用户中止，已分析的数据已保存。")
                        break
                    try:
                        img_path, result = future.result()
                    except CancelledError:
                        continue
                    if isinstance(result, list):
                        for r in result:
                            # 提前检查是否是熔断指示
                            if r is not _SKIPPED_SENTINEL and isinstance(r, dict) and r.get("__circuit_breaker_open__"):
                                circuit_breaker_err = r.get("error", "API连通性持续失败，触发断路器熔断")
                                logger.error("检测到 API 熔断，已提前终止后续扫描。")
                                if cancel_event:
                                    cancel_event.set()
                                for f in futures:
                                    f.cancel()
                                break
                                
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
                            for f in futures:
                                f.cancel()
                            break
                            
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
        except KeyboardInterrupt:
            logger.warning("用户中断 (Ctrl+C)，已分析的数据已保存。")

        _llm_bar.close()

        t_llm_end = _time.perf_counter()
        llm_cost = t_llm_end - t_start

        if circuit_breaker_err:
            logger.error("扫描因 API 故障中止: %s", circuit_breaker_err)
            if _llm_writer:
                _llm_writer.write(f"[信息扫描] ❌ 扫描中止 — {circuit_breaker_err}\n")
        else:
            logger.info(
                "信息扫描完成(耗时 %.1fs) — 总计: %d  新增/更新: %d  跳过: %d  失败: %d",
                llm_cost, stats.total, stats.processed, stats.skipped, stats.failed,
            )
            if _llm_writer:
                _llm_writer.write(
                    f"[信息扫描] ✓ 完成 — 扫描 {stats.total} 张 "
                    f"(新增 {stats.processed}, 跳过 {stats.skipped})\n"
                )
    else:
        t_llm_end = _time.perf_counter()
        llm_cost = t_llm_end - t_start
        logger.info("信息扫描: 全部 %d 张已完成，无需 LLM 分析 (耗时 %.1fs)", stats.total, llm_cost)
        if _llm_writer:
            _llm_writer.write(
                f"[信息扫描] ✓ 完成 — 全部 {stats.total} 张已就绪\n"
            )

    # ── 等待 Phase 2 后台线程完成 ──
    _face_p2_completed_event.set()
    if enable_face_scan:
        for _ in range(workers):
            _face_queue.put(None)

    for t in getattr(locals(), '_phase2_threads', []):
        try:
            t.join()
        except:
            pass

    _cancelled = cancel_event and cancel_event.is_set()
    t_face_end = _time.perf_counter()
    total_face_cost = t_face_end - t_start

    if _face_bar:
        _face_bar.close()
        logger.info("人物识别完成(总耗时 %.1fs) — 共 %d 张有人脸", total_face_cost, _face_found)
        if _face_writer:
            _face_writer.write(
                f"[人物识别] ✓ 识别完成 — 共 {_face_found} 张有人脸，正在聚类...\n"
            )
    elif enable_face_scan and _face_writer:
        if _cancelled:
            _face_writer.write("[人物识别] 已中止\n")
        else:
            _face_writer.write(f"[人物识别] ✓ 全部 {stats.total} 张已完成\n")

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
