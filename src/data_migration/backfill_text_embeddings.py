"""本文件用于为历史图片记录补全缺失的 text_embedding 字段，支持本地与 API 两种向量后端。"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from photo_identify.config import DEFAULT_DB_PATH
from photo_identify.embedding_runtime import (
    DEFAULT_LOCAL_MAX_LENGTH,
    describe_local_embedding_device,
    encode_texts_locally,
    ensure_local_embedding_runtime as ensure_local_runtime,
    get_local_embedding_model,
    get_text_embedding_sync,
    normalize_embedding_backend,
    resolve_local_embedding_device,
)
from photo_identify.storage import Storage

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = DEFAULT_LOCAL_MAX_LENGTH


@dataclass(slots=True)
class PendingImageRecord:
    """表示一条待补全 text_embedding 的图片记录。"""

    image_id: int
    scene: str
    objects_raw: str


@dataclass(slots=True)
class MigrationStats:
    """记录迁移脚本执行过程中的统计信息。"""

    total_candidates: int = 0
    processed: int = 0
    updated: int = 0
    skipped_empty_text: int = 0
    failed: int = 0


def build_argument_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="为 images 表中缺失的 text_embedding 生成并回填向量，支持本地与 API 两种后端。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="主图片数据库路径。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="运行设备，可选 auto、cpu、cuda、cuda:0 等。",
    )
    parser.add_argument(
        "--backend",
        default="local",
        help="向量后端，可选 local 或 api。",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL_ID,
        help="向量模型名称或模型ID。",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="API 向量接口地址；仅在 backend=api 时使用。",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API 向量接口密钥；仅在 backend=api 时使用。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="本地模型推荐并发/批大小提示；GUI 调用时会映射到本地批处理大小。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="本地模型单批次编码条数。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="编码时的最大 token 长度。",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="模型缓存目录；为空时使用 Hugging Face 默认缓存目录。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 条缺失记录；0 表示处理全部。",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="每成功写入多少条后执行一次提交。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计待处理记录数量，不加载模型也不写数据库。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出更详细的日志。",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    """配置脚本运行时的日志输出级别和格式。"""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def validate_arguments(args: argparse.Namespace) -> None:
    """校验命令行参数是否合法。"""

    if args.batch_size < 1:
        raise ValueError("--batch-size 必须大于等于 1。")
    if args.max_length < 1:
        raise ValueError("--max-length 必须大于等于 1。")
    if args.limit < 0:
        raise ValueError("--limit 不能为负数。")
    if args.commit_every < 1:
        raise ValueError("--commit-every 必须大于等于 1。")
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers 必须大于等于 1。")
    backend = normalize_embedding_backend(getattr(args, "backend", "local"), getattr(args, "base_url", ""))
    if backend == "api" and not getattr(args, "base_url", "").strip():
        raise ValueError("API 向量后端必须提供 --base-url。")


def ensure_existing_database(db_path: Path) -> Path:
    """确保目标数据库存在，并返回其绝对路径。"""

    resolved = db_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"数据库不存在: {resolved}")
    return resolved


def ensure_database_schema(db_path: Path) -> None:
    """确保数据库已经包含当前版本所需的表结构和字段。"""

    storage = Storage(db_path)
    storage.close()


def open_database_connection(db_path: Path) -> sqlite3.Connection:
    """打开 SQLite 连接并设置 Row 工厂。"""

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_pending_records(conn: sqlite3.Connection, limit: int) -> list[PendingImageRecord]:
    """读取尚未生成 text_embedding 且具备可用文本的图片记录。"""

    sql = """
        SELECT id, scene, objects
        FROM images
        WHERE text_embedding IS NULL
          AND (
                COALESCE(TRIM(scene), '') <> ''
                OR COALESCE(TRIM(objects), '') <> ''
              )
        ORDER BY id
    """
    params: tuple[int, ...] = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (limit,)

    cursor = conn.execute(sql, params)
    records = [
        PendingImageRecord(
            image_id=int(row["id"]),
            scene=str(row["scene"] or ""),
            objects_raw=str(row["objects"] or ""),
        )
        for row in cursor.fetchall()
    ]
    return [record for record in records if build_text_to_embed(record)]


def normalize_objects_text(objects_raw: str) -> str:
    """将数据库中的 objects 字段统一转换为适合向量化的文本。"""

    raw = objects_raw.strip()
    if not raw:
        return ""

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    if isinstance(parsed, list):
        return ", ".join(str(item).strip() for item in parsed if str(item).strip())
    if isinstance(parsed, dict):
        return ", ".join(f"{key}: {value}" for key, value in parsed.items())
    return str(parsed).strip()


def build_text_to_embed(record: PendingImageRecord) -> str:
    """根据图片记录中的 scene 和 objects 字段拼接出待向量化文本。"""

    parts: list[str] = []
    scene = record.scene.strip()
    objects_text = normalize_objects_text(record.objects_raw)

    if scene:
        parts.append(scene)
    if objects_text:
        parts.append(objects_text)

    return " ".join(parts).strip()


def chunk_records(records: list[PendingImageRecord], batch_size: int) -> Iterator[list[PendingImageRecord]]:
    """按固定大小将待处理记录切分为多个批次。"""

    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


def resolve_runtime_device(device_arg: str) -> str:
    """解析脚本应当使用的本地推理设备。"""

    return resolve_local_embedding_device(device_arg)


def describe_runtime_device(device: str) -> str:
    """返回更适合日志输出的设备描述字符串。"""

    return describe_local_embedding_device(device)


def ensure_local_embedding_runtime() -> None:
    """确保本地 embedding 推理所需的 Python 依赖已经安装。"""

    ensure_local_runtime()


def load_embedding_model(model_id: str, device: str, model_cache_dir: Path | None):
    """加载指定的本地 embedding 模型并返回可复用的编码器实例。"""

    model, _ = get_local_embedding_model(model_id=model_id, device=device, model_cache_dir=model_cache_dir)
    return model


def encode_dense_embeddings(model_id: str, texts: list[str], batch_size: int, max_length: int, device: str, model_cache_dir: Path | None) -> np.ndarray:
    """使用本地 embedding 模型批量生成 dense 向量。"""

    return encode_texts_locally(
        texts=texts,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        model_cache_dir=model_cache_dir,
    )


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """将单条 dense 向量序列化为可写入 SQLite BLOB 的字节串。"""

    return np.asarray(embedding, dtype=np.float32).tobytes()


def update_text_embedding(conn: sqlite3.Connection, image_id: int, embedding_bytes: bytes) -> None:
    """将生成好的向量写回指定图片记录的 text_embedding 字段。"""

    conn.execute(
        """
        UPDATE images
        SET text_embedding = ?
        WHERE id = ? AND text_embedding IS NULL
        """,
        (embedding_bytes, image_id),
    )


def wait_if_paused_or_cancelled(
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    interval: float = 0.2,
) -> bool:
    """在批处理循环中响应暂停/停止信号；若检测到停止则返回 True。"""

    while pause_event is not None and pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return True
        time.sleep(interval)
    return bool(cancel_event is not None and cancel_event.is_set())


def build_batch_payload(records: list[PendingImageRecord]) -> tuple[list[int], list[str], int]:
    """把批次记录整理为模型输入文本，并统计空文本条数。"""

    image_ids: list[int] = []
    texts: list[str] = []
    skipped_empty = 0

    for record in records:
        text = build_text_to_embed(record)
        if not text:
            skipped_empty += 1
            continue
        image_ids.append(record.image_id)
        texts.append(text)

    return image_ids, texts, skipped_empty


def process_record_batch(
    conn: sqlite3.Connection,
    records: list[PendingImageRecord],
    embedding_model: str,
    batch_size: int,
    max_length: int,
    commit_every: int,
    device: str,
    model_cache_dir: Path | None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> MigrationStats:
    """按批次执行本地向量生成，并将成功结果写回数据库。"""

    stats = MigrationStats(total_candidates=len(records))

    for batch in chunk_records(records, batch_size):
        if wait_if_paused_or_cancelled(cancel_event, pause_event):
            logger.info("检测到停止信号，提前结束图片向量刷新。")
            break
        image_ids, texts, skipped_empty = build_batch_payload(batch)
        stats.processed += skipped_empty
        stats.skipped_empty_text += skipped_empty

        if not texts:
            continue

        try:
            embeddings = encode_dense_embeddings(
                model_id=embedding_model,
                texts=texts,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                model_cache_dir=model_cache_dir,
            )
            for image_id, embedding in zip(image_ids, embeddings, strict=True):
                update_text_embedding(conn, image_id, serialize_embedding(embedding))
                stats.processed += 1
                stats.updated += 1
                if stats.updated % commit_every == 0:
                    conn.commit()
        except Exception as exc:
            logger.warning("批量编码失败，回退逐条处理: %s", exc)
            for image_id, text in zip(image_ids, texts, strict=True):
                if wait_if_paused_or_cancelled(cancel_event, pause_event):
                    logger.info("检测到停止信号，提前结束图片向量刷新。")
                    break
                try:
                    embedding = encode_dense_embeddings(
                        model_id=embedding_model,
                        texts=[text],
                        batch_size=1,
                        max_length=max_length,
                        device=device,
                        model_cache_dir=model_cache_dir,
                    )[0]
                    update_text_embedding(conn, image_id, serialize_embedding(embedding))
                    stats.updated += 1
                except Exception as item_exc:
                    stats.failed += 1
                    logger.error("图片 ID=%s 的向量生成失败: %s", image_id, item_exc)
                finally:
                    stats.processed += 1
                    if stats.updated > 0 and stats.updated % commit_every == 0:
                        conn.commit()

        if stats.processed == stats.total_candidates or stats.processed % 50 == 0:
            logger.info(
                "处理进度: %d/%d，已更新 %d，空文本跳过 %d，失败 %d",
                stats.processed,
                stats.total_candidates,
                stats.updated,
                stats.skipped_empty_text,
                stats.failed,
            )

    conn.commit()
    return stats


def process_record_batch_via_api(
    conn: sqlite3.Connection,
    records: list[PendingImageRecord],
    embedding_model: str,
    base_url: str,
    api_key: str,
    commit_every: int,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> MigrationStats:
    """通过远程 API 逐条生成向量，并将成功结果写回数据库。"""

    stats = MigrationStats(total_candidates=len(records))

    for record in records:
        if wait_if_paused_or_cancelled(cancel_event, pause_event):
            logger.info("检测到停止信号，提前结束图片向量刷新。")
            break
        text = build_text_to_embed(record)
        if not text:
            stats.processed += 1
            stats.skipped_empty_text += 1
            continue

        try:
            embedding = get_text_embedding_sync(
                text=text,
                model=embedding_model,
                backend="api",
                api_key=api_key,
                base_url=base_url,
            )
            update_text_embedding(conn, record.image_id, serialize_embedding(np.asarray(embedding, dtype=np.float32)))
            stats.updated += 1
        except Exception as exc:
            stats.failed += 1
            logger.error("图片 ID=%s 的远程向量生成失败: %s", record.image_id, exc)
        finally:
            stats.processed += 1
            if stats.updated > 0 and stats.updated % commit_every == 0:
                conn.commit()

        if stats.processed == stats.total_candidates or stats.processed % 50 == 0:
            logger.info(
                "处理进度: %d/%d，已更新 %d，空文本跳过 %d，失败 %d",
                stats.processed,
                stats.total_candidates,
                stats.updated,
                stats.skipped_empty_text,
                stats.failed,
            )

    conn.commit()
    return stats


def run_dry_run(records: list[PendingImageRecord]) -> int:
    """执行 dry-run，仅输出候选记录统计信息。"""

    logger.info("dry-run: 发现 %d 条待补全 text_embedding 的记录。", len(records))
    return 0


def run_migration(
    args: argparse.Namespace,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> int:
    """执行 text_embedding 回填迁移流程，并返回进程退出码。"""

    db_path = ensure_existing_database(args.db)
    ensure_database_schema(db_path)
    backend = normalize_embedding_backend(getattr(args, "backend", "local"), getattr(args, "base_url", ""))
    embedding_model = getattr(args, "embedding_model", EMBEDDING_MODEL_ID)
    conn = open_database_connection(db_path)
    try:
        records = fetch_pending_records(conn, args.limit)
        if args.dry_run:
            return run_dry_run(records)
        if not records:
            logger.info("没有发现需要补全 text_embedding 的历史记录。")
            return 0

        logger.info(
            "开始补全 text_embedding: 待处理 %d 条，模型=%s，后端=%s",
            len(records),
            embedding_model,
            backend,
        )

        if backend == "local":
            device = resolve_runtime_device(args.device)
            load_embedding_model(model_id=embedding_model, device=device, model_cache_dir=args.model_cache_dir)
            stats = process_record_batch(
                conn=conn,
                records=records,
                embedding_model=embedding_model,
                batch_size=args.batch_size,
                max_length=args.max_length,
                commit_every=args.commit_every,
                device=device,
                model_cache_dir=args.model_cache_dir,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )
        else:
            stats = process_record_batch_via_api(
                conn=conn,
                records=records,
                embedding_model=embedding_model,
                base_url=getattr(args, "base_url", ""),
                api_key=getattr(args, "api_key", ""),
                commit_every=args.commit_every,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )
        logger.info(
            "迁移完成: 候选 %d，已处理 %d，成功更新 %d，空文本跳过 %d，失败 %d",
            stats.total_candidates,
            stats.processed,
            stats.updated,
            stats.skipped_empty_text,
            stats.failed,
        )
        return 0 if stats.failed == 0 else 1
    finally:
        conn.close()


def main() -> int:
    """解析命令行参数并启动迁移脚本。"""

    parser = build_argument_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    try:
        validate_arguments(args)
        return run_migration(args)
    except Exception as exc:
        logger.error("迁移失败: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
