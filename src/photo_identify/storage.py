"""SQLite 存储层，负责图片分析结果的持久化与全文搜索。

使用 SQLite + FTS5 实现：
- 增量写入（单条 INSERT，无需全量序列化）
- 按 MD5 内容哈希去重
- FTS5 全文搜索支持口语化检索
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

import jieba
import numpy as np


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    size_bytes INTEGER,
    md5 TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    width INTEGER,
    height INTEGER,
    image_mode TEXT,
    image_format TEXT,
    exif_json TEXT,
    created_time TEXT,
    modified_time TEXT,
    scene TEXT,
    objects TEXT,
    style TEXT,
    location_time TEXT,
    wallpaper_hint TEXT,
    llm_raw TEXT,
    analyzed_at TEXT,
    face_scanned INTEGER DEFAULT 0,
    is_favorite INTEGER DEFAULT 0,
    text_embedding BLOB
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    bbox TEXT NOT NULL,
    embedding BLOB NOT NULL,
    cluster_id INTEGER DEFAULT -1,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS persons (
    id TEXT PRIMARY KEY,
    cluster_id INTEGER UNIQUE,
    name TEXT NOT NULL,
    cover_image_id INTEGER,
    cover_face_id INTEGER,
    sort_order INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    face_id INTEGER NOT NULL UNIQUE,
    person_id TEXT NOT NULL,
    source_cluster_id INTEGER,
    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY(face_id) REFERENCES face_embeddings(id) ON DELETE CASCADE,
    FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
CREATE INDEX IF NOT EXISTS idx_face_image_id ON face_embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster_id ON face_embeddings(cluster_id);
CREATE INDEX IF NOT EXISTS idx_photos_person_id ON photos(person_id);
CREATE INDEX IF NOT EXISTS idx_photos_image_id ON photos(image_id);
CREATE INDEX IF NOT EXISTS idx_photos_cluster_id ON photos(source_cluster_id);
"""

_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
    file_name, scene, objects, style, location_time, wallpaper_hint,
    content='images',
    content_rowid='id'
);
"""

_FTS_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
    INSERT INTO images_fts(rowid, file_name, scene, objects, style, location_time, wallpaper_hint)
    VALUES (new.id, new.file_name, new.scene, new.objects, new.style, new.location_time, new.wallpaper_hint);
END;
"""

_FTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, file_name, scene, objects, style, location_time, wallpaper_hint)
    VALUES ('delete', old.id, old.file_name, old.scene, old.objects, old.style, old.location_time, old.wallpaper_hint);
END;
"""

_FTS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
    INSERT INTO images_fts(images_fts, rowid, file_name, scene, objects, style, location_time, wallpaper_hint)
    VALUES ('delete', old.id, old.file_name, old.scene, old.objects, old.style, old.location_time, old.wallpaper_hint);
    INSERT INTO images_fts(rowid, file_name, scene, objects, style, location_time, wallpaper_hint)
    VALUES (new.id, new.file_name, new.scene, new.objects, new.style, new.location_time, new.wallpaper_hint);
END;
"""


class Storage:
    """SQLite 存储管理器。"""

    def __init__(self, db_path: str | Path):
        """初始化数据库连接并创建表结构。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        """创建数据库表和全文搜索索引，并兼容迁移旧版人物结构。"""
        cursor = self._conn.cursor()

        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                size_bytes INTEGER,
                md5 TEXT NOT NULL UNIQUE,
                sha256 TEXT,
                width INTEGER,
                height INTEGER,
                image_mode TEXT,
                image_format TEXT,
                exif_json TEXT,
                created_time TEXT,
                modified_time TEXT,
                scene TEXT,
                objects TEXT,
                style TEXT,
                location_time TEXT,
                wallpaper_hint TEXT,
                llm_raw TEXT,
                analyzed_at TEXT,
                face_scanned INTEGER DEFAULT 0,
                is_favorite INTEGER DEFAULT 0,
                text_embedding BLOB
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                bbox TEXT NOT NULL,
                embedding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT -1,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5);
            CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
            CREATE INDEX IF NOT EXISTS idx_face_image_id ON face_embeddings(image_id);
            CREATE INDEX IF NOT EXISTS idx_face_cluster_id ON face_embeddings(cluster_id);
            """
        )

        # 增加 face_scanned 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN face_scanned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        # 增加 is_favorite 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN is_favorite INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        # 增加 text_embedding 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN text_embedding BLOB")
        except sqlite3.OperationalError:
            pass

        self._ensure_person_schema(cursor)
        self._ensure_photos_schema(cursor)
        self._bootstrap_persons_from_clustered_faces_if_needed(cursor)
        self._backfill_photos_if_needed(cursor)

        cursor.executescript(_FTS_SCHEMA_SQL)
        cursor.executescript(_FTS_TRIGGER_INSERT)
        cursor.executescript(_FTS_TRIGGER_DELETE)
        cursor.executescript(_FTS_TRIGGER_UPDATE)

        # 持续失败的文件记录表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS skipped_files (
                path TEXT PRIMARY KEY,
                reason TEXT,
                skipped_at TEXT
            )
            """
        )

        self._conn.commit()

    def _table_exists(self, cursor: sqlite3.Cursor, table_name: str) -> bool:
        row = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _get_table_info(self, cursor: sqlite3.Cursor, table_name: str) -> dict[str, dict]:
        if not self._table_exists(cursor, table_name):
            return {}
        info = {}
        for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall():
            info[row[1]] = {
                "type": str(row[2] or "").upper(),
                "notnull": int(row[3] or 0),
                "default": row[4],
                "pk": int(row[5] or 0),
            }
        return info

    def _create_persons_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                cluster_id INTEGER UNIQUE,
                name TEXT NOT NULL,
                cover_image_id INTEGER,
                cover_face_id INTEGER,
                sort_order INTEGER DEFAULT 0,
                is_deleted INTEGER DEFAULT 0
            )
            """
        )

    def _create_photos_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                face_id INTEGER NOT NULL UNIQUE,
                person_id TEXT NOT NULL,
                source_cluster_id INTEGER,
                FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE,
                FOREIGN KEY(face_id) REFERENCES face_embeddings(id) ON DELETE CASCADE,
                FOREIGN KEY(person_id) REFERENCES persons(id) ON DELETE CASCADE ON UPDATE CASCADE
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_person_id ON photos(person_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_image_id ON photos(image_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_photos_cluster_id ON photos(source_cluster_id)")

    def _ensure_person_schema(self, cursor: sqlite3.Cursor) -> None:
        info = self._get_table_info(cursor, "persons")
        if not info:
            self._create_persons_table(cursor)
            return

        id_type = info.get("id", {}).get("type", "")
        id_pk = info.get("id", {}).get("pk", 0)
        if id_type != "TEXT" or id_pk != 1:
            self._migrate_legacy_persons(cursor)
            info = self._get_table_info(cursor, "persons")

        if "sort_order" not in info:
            cursor.execute("ALTER TABLE persons ADD COLUMN sort_order INTEGER DEFAULT 0")
        if "is_deleted" not in info:
            cursor.execute("ALTER TABLE persons ADD COLUMN is_deleted INTEGER DEFAULT 0")
        if "cluster_id" not in info:
            cursor.execute("ALTER TABLE persons ADD COLUMN cluster_id INTEGER")
        if "cover_image_id" not in info:
            cursor.execute("ALTER TABLE persons ADD COLUMN cover_image_id INTEGER")
        if "cover_face_id" not in info:
            cursor.execute("ALTER TABLE persons ADD COLUMN cover_face_id INTEGER")

    def _ensure_photos_schema(self, cursor: sqlite3.Cursor) -> None:
        self._create_photos_table(cursor)

    def _migrate_legacy_persons(self, cursor: sqlite3.Cursor) -> None:
        legacy_table = "persons_legacy_uuid_migration"
        if self._table_exists(cursor, legacy_table):
            cursor.execute(f"DROP TABLE {legacy_table}")

        cursor.execute("ALTER TABLE persons RENAME TO persons_legacy_uuid_migration")
        self._create_persons_table(cursor)
        self._create_photos_table(cursor)

        legacy_info = self._get_table_info(cursor, legacy_table)
        select_columns = [
            "id" if "id" in legacy_info else "NULL AS id",
            "cluster_id" if "cluster_id" in legacy_info else "NULL AS cluster_id",
            "name" if "name" in legacy_info else "'' AS name",
            "cover_image_id" if "cover_image_id" in legacy_info else "NULL AS cover_image_id",
            "cover_face_id" if "cover_face_id" in legacy_info else "NULL AS cover_face_id",
            "sort_order" if "sort_order" in legacy_info else "0 AS sort_order",
            "is_deleted" if "is_deleted" in legacy_info else "0 AS is_deleted",
        ]
        rows = cursor.execute(
            f"SELECT {', '.join(select_columns)} FROM {legacy_table} ORDER BY id ASC"
        ).fetchall()

        for row in rows:
            person_uuid = self._new_person_uuid()
            cluster_id = row["cluster_id"]
            cover_face_id = row["cover_face_id"]
            cover_image_id = row["cover_image_id"]

            if cluster_id is not None:
                cluster_faces = cursor.execute(
                    """
                    SELECT fe.id, fe.image_id
                    FROM face_embeddings fe
                    JOIN images i ON i.id = fe.image_id
                    WHERE fe.cluster_id = ?
                    ORDER BY fe.id ASC
                    """,
                    (cluster_id,),
                ).fetchall()
                if cluster_faces:
                    if cover_face_id is None or cover_image_id is None:
                        cover_face_id = cluster_faces[0]["id"]
                        cover_image_id = cluster_faces[0]["image_id"]
                    for face_row in cluster_faces:
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO photos (image_id, face_id, person_id, source_cluster_id)
                            VALUES (?, ?, ?, ?)
                            """,
                            (face_row["image_id"], face_row["id"], person_uuid, cluster_id),
                        )

            cursor.execute(
                """
                INSERT INTO persons (id, cluster_id, name, cover_image_id, cover_face_id, sort_order, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_uuid,
                    cluster_id,
                    row["name"] or "未命名人物",
                    cover_image_id,
                    cover_face_id,
                    row["sort_order"] or 0,
                    row["is_deleted"] or 0,
                ),
            )

        cursor.execute(f"DROP TABLE {legacy_table}")

    def _bootstrap_persons_from_clustered_faces_if_needed(self, cursor: sqlite3.Cursor) -> None:
        persons_count = cursor.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        if persons_count > 0:
            return

        clustered_rows = cursor.execute(
            """
            SELECT fe.cluster_id, fe.id, fe.image_id
            FROM face_embeddings fe
            JOIN images i ON i.id = fe.image_id
            WHERE fe.cluster_id >= 0
            ORDER BY fe.cluster_id ASC, fe.id ASC
            """
        ).fetchall()
        if not clustered_rows:
            return

        grouped_faces: dict[int, list[sqlite3.Row]] = {}
        for row in clustered_rows:
            grouped_faces.setdefault(int(row["cluster_id"]), []).append(row)

        for sort_order, cluster_id in enumerate(sorted(grouped_faces)):
            cluster_faces = grouped_faces[cluster_id]
            cover_face = cluster_faces[0]
            person_uuid = self._new_person_uuid()
            person_name = f"人物_{cluster_id:03d}"
            cursor.execute(
                """
                INSERT INTO persons (id, cluster_id, name, cover_image_id, cover_face_id, sort_order, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    person_uuid,
                    cluster_id,
                    person_name,
                    cover_face["image_id"],
                    cover_face["id"],
                    sort_order,
                ),
            )
            for face_row in cluster_faces:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO photos (image_id, face_id, person_id, source_cluster_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (face_row["image_id"], face_row["id"], person_uuid, cluster_id),
                )

    def _backfill_photos_if_needed(self, cursor: sqlite3.Cursor) -> None:
        photos_count = cursor.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        if photos_count > 0:
            return

        persons = cursor.execute(
            "SELECT id, cluster_id FROM persons WHERE cluster_id IS NOT NULL"
        ).fetchall()
        for row in persons:
            cluster_id = row["cluster_id"]
            if cluster_id is None:
                continue
            cluster_faces = cursor.execute(
                """
                SELECT fe.id, fe.image_id
                FROM face_embeddings fe
                JOIN images i ON i.id = fe.image_id
                WHERE fe.cluster_id = ?
                ORDER BY fe.id ASC
                """,
                (cluster_id,),
            ).fetchall()
            for face_row in cluster_faces:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO photos (image_id, face_id, person_id, source_cluster_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (face_row["image_id"], face_row["id"], row["id"], cluster_id),
                )

    def _new_person_uuid(self) -> str:
        return str(uuid.uuid4())

    def get_known_md5s(self) -> set[str]:
        """获取所有已入库图片的 MD5 集合，用于批量跳过检测。

        Returns:
            MD5 字符串的集合。
        """
        cursor = self._conn.execute("SELECT md5 FROM images")
        return {row[0] for row in cursor.fetchall()}

    def get_face_scanned_md5s(self) -> set[str]:
        """获取所有已完成人脸扫描的图片的 MD5 集合。"""
        try:
            cursor = self._conn.execute("SELECT md5 FROM images WHERE face_scanned = 1")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set()

    def get_known_paths(self) -> dict[str, tuple[int, str, str]]:
        """获取所有已入库图片的路径及其 size+mtime+md5 信息，用于快速跳过判断。

        Returns:
            字典，key 为路径，value 为 (size_bytes, modified_time 字符串, md5 字符串)。
        """
        cursor = self._conn.execute("SELECT path, size_bytes, modified_time, md5 FROM images")
        return {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}

    def delete_by_paths(self, paths_to_delete: list[str]) -> int:
        """从数据库中删除指定路径的记录（支持批量）。

        Args:
            paths_to_delete: 需要删除的图片路径列表。

        Returns:
            删除的记录数。
        """
        if not paths_to_delete:
            return 0

        deleted_count = 0
        cursor = self._conn.cursor()

        # SQLite IN clause 变量数量限制，因此分批删除
        batch_size = 500
        for i in range(0, len(paths_to_delete), batch_size):
            batch = paths_to_delete[i:i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            cursor.execute(f"DELETE FROM images WHERE path IN ({placeholders})", batch)
            deleted_count += cursor.rowcount

        self._conn.commit()
        return deleted_count

    def get_skipped_paths(self) -> set[str]:
        """获取所有持续失败的文件路径集合，扫描时跳过。"""
        try:
            cursor = self._conn.execute("SELECT path FROM skipped_files")
            return {row[0] for row in cursor.fetchall()}
        except sqlite3.OperationalError:
            return set()

    def add_skipped_file(self, path: str, reason: str):
        """记录持续失败的文件，下次扫描时跳过。"""
        from datetime import datetime

        self._conn.execute(
            "INSERT OR REPLACE INTO skipped_files (path, reason, skipped_at) VALUES (?, ?, ?)",
            (path, reason, datetime.now().isoformat()),
        )
        self._conn.commit()

    def has_md5(self, md5: str) -> bool:
        """检查指定 MD5 是否已入库。

        Args:
            md5: 图片内容的 MD5 哈希值。

        Returns:
            是否存在。
        """
        cursor = self._conn.execute("SELECT 1 FROM images WHERE md5 = ?", (md5,))
        return cursor.fetchone() is not None

    def upsert(self, record: dict) -> int:
        """插入或更新一条图片分析记录（按 MD5 去重）。

        Args:
            record: 包含所有字段的字典。

        Returns:
            int: 插入或更新的图片记录的数据库 ID (image_id)
        """
        objects_str = json.dumps(record.get("objects", []), ensure_ascii=False) if isinstance(record.get("objects"), list) else record.get("objects", "")
        cursor = self._conn.cursor()

        # 尝试查询已有的 id
        cursor.execute("SELECT id FROM images WHERE md5 = ?", (record["md5"],))
        existing_row = cursor.fetchone()

        update_face_scanned = record.get("face_scanned", False)
        if existing_row:
            image_id = existing_row[0]
            if update_face_scanned:
                cursor.execute(
                    """
                    UPDATE images SET
                        path = ?,
                        file_name = ?,
                        size_bytes = ?,
                        analyzed_at = ?,
                        text_embedding = COALESCE(?, text_embedding),
                        face_scanned = 1
                    WHERE md5 = ?
                    """,
                    (
                        record.get("path", ""),
                        record.get("file_name", ""),
                        record.get("size_bytes"),
                        record.get("analyzed_at", ""),
                        record.get("text_embedding"),
                        record["md5"],
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE images SET
                        path = ?,
                        file_name = ?,
                        size_bytes = ?,
                        text_embedding = COALESCE(?, text_embedding),
                        analyzed_at = ?
                    WHERE md5 = ?
                    """,
                    (
                        record.get("path", ""),
                        record.get("file_name", ""),
                        record.get("size_bytes"),
                        record.get("text_embedding"),
                        record.get("analyzed_at", ""),
                        record["md5"],
                    ),
                )
        else:
            cursor.execute(
                """
                INSERT INTO images (
                    path, file_name, size_bytes, md5, sha256,
                    width, height, image_mode, image_format, exif_json,
                    created_time, modified_time,
                    scene, objects, style, location_time, wallpaper_hint,
                    llm_raw, analyzed_at, face_scanned, text_embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("path", ""),
                    record.get("file_name", ""),
                    record.get("size_bytes"),
                    record["md5"],
                    record.get("sha256", ""),
                    record.get("width"),
                    record.get("height"),
                    record.get("image_mode", ""),
                    record.get("image_format", ""),
                    json.dumps(record.get("exif", {}), ensure_ascii=False),
                    record.get("created_time", ""),
                    record.get("modified_time", ""),
                    record.get("scene", ""),
                    objects_str,
                    record.get("style", ""),
                    record.get("location_time", ""),
                    record.get("wallpaper_hint", ""),
                    record.get("llm_raw", ""),
                    record.get("analyzed_at", ""),
                    1 if update_face_scanned else 0,
                    record.get("text_embedding"),
                ),
            )
            image_id = cursor.lastrowid or 0

        self._conn.commit()
        return int(image_id)

    def toggle_favorite(self, image_id: int, is_favorite: bool):
        """更新图片的收藏状态"""
        self._conn.execute(
            "UPDATE images SET is_favorite = ? WHERE id = ?",
            (1 if is_favorite else 0, image_id),
        )
        self._conn.commit()

    def get_favorites(self) -> list[dict]:
        """获取所有收藏的图片（按修改时间倒序）。"""
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE is_favorite = 1 ORDER BY modified_time DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_face_scanned(self, image_id: int):
        """将指定图片的 face_scanned 标记为 1"""
        self._conn.execute("UPDATE images SET face_scanned = 1 WHERE id = ?", (image_id,))
        self._conn.commit()

    def add_face_embeddings(self, image_id: int, faces: list[dict]):
        """将提取到的人脸特征写入数据库。

        Args:
            image_id: 关联的图片 ID
            faces: face_manager.extract_faces 返回的字典列表
        """
        if not faces:
            return

        cursor = self._conn.cursor()
        for face in faces:
            bbox_str = json.dumps(face["bbox"])
            embedding_blob = face["embedding"].tobytes()
            cursor.execute(
                "INSERT INTO face_embeddings (image_id, bbox, embedding) VALUES (?, ?, ?)",
                (image_id, bbox_str, embedding_blob),
            )
        self._conn.commit()

    def delete_face_embeddings_for_image(self, image_id: int) -> None:
        """删除指定图片已有的人脸特征记录，避免重复写入。"""
        self._conn.execute("DELETE FROM photos WHERE image_id = ?", (image_id,))
        self._conn.execute("DELETE FROM face_embeddings WHERE image_id = ?", (image_id,))
        self._conn.commit()

    def get_unclustered_faces(self) -> list[tuple[int, bytes]]:
        """获取所有尚未聚类的人脸数据。

        Returns:
            list of (face_id, embedding_bytes)
        """
        cursor = self._conn.execute("SELECT id, embedding FROM face_embeddings WHERE cluster_id = -1")
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_all_faces(self) -> list[tuple[int, bytes]]:
        """获取所有的人脸数据，用于全量聚类。

        Returns:
            list of (face_id, embedding_bytes)
        """
        cursor = self._conn.execute("SELECT id, embedding FROM face_embeddings")
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def update_face_clusters(self, cluster_mapping: dict[int, int]):
        """更新人脸聚类结果并同步至 persons / photos 表。

        Args:
            cluster_mapping: {face_id: cluster_id}
        """
        if not cluster_mapping:
            return

        cursor = self._conn.cursor()

        existing_persons = cursor.execute(
            "SELECT id, cluster_id, name, cover_image_id, cover_face_id, sort_order, is_deleted FROM persons"
        ).fetchall()
        person_by_id = {row["id"]: dict(row) for row in existing_persons}
        legacy_by_cluster = {
            row["cluster_id"]: dict(row)
            for row in existing_persons
            if row["cluster_id"] is not None
        }
        existing_face_person = {
            row["face_id"]: row["person_id"]
            for row in cursor.execute("SELECT face_id, person_id FROM photos").fetchall()
        }

        for face_id, cluster_id in cluster_mapping.items():
            cursor.execute("UPDATE face_embeddings SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))

        unique_clusters = sorted({cid for cid in cluster_mapping.values() if cid >= 0})
        cluster_entries: list[dict] = []
        for cid in unique_clusters:
            cluster_faces = cursor.execute(
                "SELECT id, image_id FROM face_embeddings WHERE cluster_id = ? ORDER BY id ASC",
                (cid,),
            ).fetchall()
            if not cluster_faces:
                continue

            person_id = None
            for face_row in cluster_faces:
                mapped = existing_face_person.get(face_row["id"])
                if mapped:
                    person_id = mapped
                    break

            if not person_id:
                legacy = legacy_by_cluster.get(cid)
                if legacy and legacy.get("id"):
                    person_id = str(legacy["id"])

            if not person_id:
                person_id = self._new_person_uuid()

            cluster_entries.append({
                "cluster_id": cid,
                "person_id": person_id,
                "faces": [dict(face_row) for face_row in cluster_faces],
            })

        grouped_entries: dict[str, list[dict]] = {}
        for entry in cluster_entries:
            grouped_entries.setdefault(entry["person_id"], []).append(entry)

        cursor.execute("DELETE FROM photos")
        cursor.execute("DELETE FROM persons")

        for person_id, entries in grouped_entries.items():
            old_person = person_by_id.get(person_id)
            representative_cluster = min(entry["cluster_id"] for entry in entries)
            all_faces = [face for entry in entries for face in entry["faces"]]
            faces_by_id = {face["id"]: face for face in all_faces}
            legacy = old_person or legacy_by_cluster.get(representative_cluster) or {}

            person_name = legacy.get("name") or f"人物_{representative_cluster:03d}"
            sort_order = int(legacy.get("sort_order") or 0)
            is_deleted = int(legacy.get("is_deleted") or 0)
            cover_face_id = legacy.get("cover_face_id")
            cover_image_id = legacy.get("cover_image_id")

            if cover_face_id not in faces_by_id:
                cover_face_id = all_faces[0]["id"] if all_faces else None
                cover_image_id = all_faces[0]["image_id"] if all_faces else None
            elif cover_image_id is None and cover_face_id in faces_by_id:
                cover_image_id = faces_by_id[cover_face_id]["image_id"]

            cursor.execute(
                """
                INSERT INTO persons (id, cluster_id, name, cover_image_id, cover_face_id, sort_order, is_deleted)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    representative_cluster,
                    person_name,
                    cover_image_id,
                    cover_face_id,
                    sort_order,
                    is_deleted,
                ),
            )

            for entry in entries:
                for face in entry["faces"]:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO photos (image_id, face_id, person_id, source_cluster_id)
                        VALUES (?, ?, ?, ?)
                        """,
                        (face["image_id"], face["id"], person_id, entry["cluster_id"]),
                    )

        self._conn.commit()

    def get_all_persons(self, include_deleted: bool = False) -> list[dict]:
        """获取所有人物列表（基于 UUID 标识与 photos 关联计数）。"""
        where_clause = "" if include_deleted else "WHERE p.is_deleted = 0"
        cursor = self._conn.execute(
            f"""
            SELECT p.id, p.cluster_id, p.name, p.cover_image_id, p.cover_face_id,
                   p.sort_order, p.is_deleted,
                   i.path, fe.bbox,
                   COALESCE(pc.photo_count, 0) AS face_count
            FROM persons p
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings fe ON p.cover_face_id = fe.id
            LEFT JOIN (
                SELECT person_id, COUNT(DISTINCT image_id) AS photo_count
                FROM photos
                GROUP BY person_id
            ) pc ON pc.person_id = p.id
            {where_clause}
            ORDER BY p.sort_order ASC, face_count DESC, p.name ASC, p.id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_deleted_persons(self) -> list[dict]:
        """获取回收站内的人物列表，按照片数倒序。"""
        cursor = self._conn.execute(
            """
            SELECT p.id, p.cluster_id, p.name, p.cover_image_id, p.cover_face_id,
                   i.path, fe.bbox,
                   COALESCE(pc.photo_count, 0) AS face_count
            FROM persons p
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings fe ON p.cover_face_id = fe.id
            LEFT JOIN (
                SELECT person_id, COUNT(DISTINCT image_id) AS photo_count
                FROM photos
                GROUP BY person_id
            ) pc ON pc.person_id = p.id
            WHERE p.is_deleted = 1
            ORDER BY face_count DESC, p.name ASC, p.id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_person_by_id(self, person_id: str) -> dict | None:
        row = self._conn.execute(
            """
            SELECT p.id, p.cluster_id, p.name, p.cover_image_id, p.cover_face_id,
                   p.sort_order, p.is_deleted, i.path, fe.bbox
            FROM persons p
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings fe ON p.cover_face_id = fe.id
            WHERE p.id = ?
            """,
            (person_id,),
        ).fetchone()
        return dict(row) if row else None

    def update_person_sort_order(self, person_ids_in_order: list[str]):
        """批量更新人物排序顺序。"""
        cursor = self._conn.cursor()
        for idx, person_id in enumerate(person_ids_in_order):
            cursor.execute("UPDATE persons SET sort_order = ? WHERE id = ?", (idx, person_id))
        self._conn.commit()

    def set_person_pinned(self, person_id: str, is_pinned: bool = True):
        """设置人物是否置顶。"""
        sort_order = -1 if is_pinned else 0
        self._conn.execute("UPDATE persons SET sort_order = ? WHERE id = ?", (sort_order, person_id))
        self._conn.commit()

    def update_person_cover(self, person_id: str, cover_image_id: int, cover_face_id: int):
        """设置某人物的封面头像。"""
        self._conn.execute(
            "UPDATE persons SET cover_image_id = ?, cover_face_id = ? WHERE id = ?",
            (cover_image_id, cover_face_id, person_id),
        )
        self._conn.commit()

    def set_person_deleted(self, person_id: str, is_deleted: int = 1):
        """将人物移入/移出回收站。"""
        self._conn.execute("UPDATE persons SET is_deleted = ? WHERE id = ?", (is_deleted, person_id))
        self._conn.commit()

    def get_images_by_person(self, person_id: str) -> list[dict]:
        """获取某个人物相关的图片。"""
        sql = """
        SELECT DISTINCT i.*
        FROM photos ph
        JOIN images i ON ph.image_id = i.id
        WHERE ph.person_id = ?
        ORDER BY i.modified_time DESC, i.id DESC
        """
        cursor = self._conn.execute(sql, (person_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_person_face_for_image(self, person_id: str, image_id: int) -> tuple[int, str] | None:
        """获取指定人物在某张图片中的 face_id 与 bbox。"""
        row = self._conn.execute(
            """
            SELECT fe.id, fe.bbox
            FROM photos ph
            JOIN face_embeddings fe ON ph.face_id = fe.id
            WHERE ph.person_id = ? AND ph.image_id = ?
            ORDER BY fe.id ASC
            LIMIT 1
            """,
            (person_id, image_id),
        ).fetchone()
        if not row:
            return None
        return int(row[0]), str(row[1])

    def update_person_name(self, person_id: str, new_name: str):
        """重命名人物。"""
        self._conn.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
        self._conn.commit()

    def merge_persons(self, target_person_id: str, source_person_id: str) -> dict:
        """将 source 人物合并到 target 人物，并持久化到当前数据库。"""
        if not target_person_id or not source_person_id:
            raise ValueError("人物 UUID 不能为空")
        if target_person_id == source_person_id:
            return {"merged": False, "reason": "same_uuid"}

        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            source_row = cursor.execute(
                "SELECT * FROM persons WHERE id = ?",
                (source_person_id,),
            ).fetchone()
            if not source_row:
                raise ValueError(f"未找到待合并人物: {source_person_id}")

            target_row = cursor.execute(
                "SELECT * FROM persons WHERE id = ?",
                (target_person_id,),
            ).fetchone()

            if target_row:
                cursor.execute(
                    "UPDATE photos SET person_id = ? WHERE person_id = ?",
                    (target_person_id, source_person_id),
                )

                merged_name = target_row["name"] or source_row["name"]
                merged_sort = min(int(target_row["sort_order"] or 0), int(source_row["sort_order"] or 0))
                merged_deleted = 1 if int(target_row["is_deleted"] or 0) and int(source_row["is_deleted"] or 0) else 0
                merged_cover_image = target_row["cover_image_id"] or source_row["cover_image_id"]
                merged_cover_face = target_row["cover_face_id"] or source_row["cover_face_id"]

                representative_cluster = cursor.execute(
                    "SELECT MIN(source_cluster_id) FROM photos WHERE person_id = ?",
                    (target_person_id,),
                ).fetchone()[0]

                cursor.execute(
                    """
                    UPDATE persons
                    SET name = ?, sort_order = ?, is_deleted = ?,
                        cover_image_id = ?, cover_face_id = ?, cluster_id = ?
                    WHERE id = ?
                    """,
                    (
                        merged_name,
                        merged_sort,
                        merged_deleted,
                        merged_cover_image,
                        merged_cover_face,
                        representative_cluster,
                        target_person_id,
                    ),
                )
                cursor.execute("DELETE FROM persons WHERE id = ?", (source_person_id,))
            else:
                cursor.execute(
                    "UPDATE persons SET id = ? WHERE id = ?",
                    (target_person_id, source_person_id),
                )
                representative_cluster = cursor.execute(
                    "SELECT MIN(source_cluster_id) FROM photos WHERE person_id = ?",
                    (target_person_id,),
                ).fetchone()[0]
                cursor.execute(
                    "UPDATE persons SET cluster_id = ? WHERE id = ?",
                    (representative_cluster, target_person_id),
                )

            self._conn.commit()
            return {"merged": True, "target_person_id": target_person_id, "source_person_id": source_person_id}
        except Exception:
            self._conn.rollback()
            raise

    def get_person_feature_vectors(self, include_deleted: bool = False) -> list[dict]:
        """返回人物的聚合向量，用于跨库相似人物推荐。"""
        where_clause = "" if include_deleted else "WHERE p.is_deleted = 0"
        rows = self._conn.execute(
            f"""
            SELECT p.id, p.name, p.cluster_id, p.cover_image_id, p.cover_face_id, p.is_deleted,
                   i.path, cover.bbox, ph.face_id, fe.embedding
            FROM persons p
            JOIN photos ph ON ph.person_id = p.id
            JOIN face_embeddings fe ON ph.face_id = fe.id
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings cover ON p.cover_face_id = cover.id
            {where_clause}
            ORDER BY p.sort_order ASC, p.name ASC, ph.face_id ASC
            """
        ).fetchall()

        grouped: dict[str, dict] = {}
        for row in rows:
            person_id = row["id"]
            item = grouped.setdefault(
                person_id,
                {
                    "id": person_id,
                    "name": row["name"],
                    "cluster_id": row["cluster_id"],
                    "path": row["path"],
                    "bbox": row["bbox"],
                    "face_count": 0,
                    "_vectors": [],
                },
            )
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size == 0:
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-8:
                continue
            item["_vectors"].append(vector / norm)
            item["face_count"] += 1

        results = []
        for person in grouped.values():
            vectors = person.pop("_vectors", [])
            if not vectors:
                continue
            mean_vec = np.mean(np.stack(vectors), axis=0)
            mean_norm = float(np.linalg.norm(mean_vec))
            if mean_norm <= 1e-8:
                continue
            person["embedding_vector"] = mean_vec / mean_norm
            results.append(person)
        return results

    def get_all_embeddings(self) -> list[tuple[int, bytes]]:
        """获取所有已入库图片的特征向量，用于语义搜索。

        Returns:
            list of (image_id, text_embedding_bytes)
        """
        cursor = self._conn.execute("SELECT id, text_embedding FROM images WHERE text_embedding IS NOT NULL")
        return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_images_by_ids(self, image_ids: list[int]) -> list[dict]:
        """根据图片 ID 列表批量获取图片信息并保持顺序。"""
        if not image_ids:
            return []

        placeholders = ",".join("?" for _ in image_ids)
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            f"SELECT * FROM images WHERE id IN ({placeholders})",
            image_ids,
        )

        results_map = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            row_dict["file_size"] = row_dict.get("size_bytes")
            row_dict["photo_date"] = row_dict.get("created_time")
            row_dict["updated_at"] = row_dict.get("modified_time")
            results_map[row_dict["id"]] = row_dict

        ordered_results = []
        for img_id in image_ids:
            if img_id in results_map:
                ordered_results.append(results_map[img_id])

        return ordered_results

    def get_images_paginated(self, offset: int = 0, limit: int = 50) -> list[dict]:
        """分页获取所有图片，按修改时间倒序排列。

        Args:
            offset: 跳过的记录数
            limit: 返回的最大记录数

        Returns:
            图片记录列表
        """
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            """
            SELECT * FROM images
            ORDER BY modified_time DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            row_dict["file_size"] = row_dict.get("size_bytes")
            row_dict["photo_date"] = row_dict.get("created_time")
            row_dict["updated_at"] = row_dict.get("modified_time")
            results.append(row_dict)
        return results

    def search_fts(self, query: str, limit: int = 50) -> list[dict]:
        """全文检索。

        引入 jieba 分词，把长句或未加空格的词拆分成细粒度词汇组合。
        例如：结婚照 -> 结婚 OR 照。
        同时，如果 query 包含已知人物名，优先将其照片排在首位。
        """
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()

        # 0. 尝试从查询词中匹配人物名称
        person_images = {}
        try:
            cursor.execute("SELECT id, name FROM persons WHERE is_deleted = 0")
            all_persons = cursor.fetchall()
            for p_row in all_persons:
                p_name = p_row["name"]
                if p_name in query:
                    cursor.execute(
                        "SELECT DISTINCT image_id FROM photos WHERE person_id = ?",
                        (p_row["id"],),
                    )
                    for img_row in cursor.fetchall():
                        person_images[img_row["image_id"]] = p_name
        except sqlite3.OperationalError:
            pass

        parts = query.split()
        match_terms = []
        for p in parts:
            cut_words = list(jieba.cut_for_search(p))
            if cut_words:
                group = " OR ".join([f'"{w}"*' for w in cut_words])
                match_terms.append(f"({group})")

        if not match_terms and not person_images:
            return []

        fts_results = {}
        if match_terms:
            match_query_and = " AND ".join(match_terms)
            print(f"  🔍 FTS 分词转换 (精确): \"{query}\" -> \"{match_query_and}\"")

            sql = """
            SELECT
                r.id,
                r.path,
                r.file_name,
                r.size_bytes AS file_size,
                r.created_time AS photo_date,
                r.scene,
                r.objects,
                r.style,
                r.modified_time AS updated_at,
                fts.rank as score
            FROM images_fts fts
            JOIN images r ON fts.rowid = r.id
            WHERE images_fts MATCH ?
            """

            and_count = 0
            try:
                cursor.execute(sql + f" LIMIT {limit * 5}", (match_query_and,))
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    row_dict["score"] = row_dict["score"] - 100.0
                    fts_results[row_dict["id"]] = row_dict
                    and_count += 1
            except sqlite3.OperationalError as e:
                import sys

                print(f"  ⚠️ FTS 精确查询语法有误: {match_query_and} ({e})", file=sys.stderr)

            if and_count < limit and len(match_terms) > 1:
                match_query_or = " OR ".join(match_terms)
                print(f"  🔍 FTS 分词转换 (降级): \"{query}\" -> \"{match_query_or}\"")
                try:
                    cursor.execute(sql + f" LIMIT {limit * 5}", (match_query_or,))
                    for row in cursor.fetchall():
                        row_dict = dict(row)
                        if row_dict["id"] not in fts_results:
                            fts_results[row_dict["id"]] = row_dict
                except sqlite3.OperationalError as e:
                    import sys

                    print(f"  ⚠️ FTS 降级查询语法有误: {match_query_or} ({e})", file=sys.stderr)

            if not fts_results:
                cursor.execute(
                    "SELECT * FROM images WHERE scene LIKE ? OR objects LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit * 5),
                )
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    row_dict["score"] = -0.5
                    fts_results[row_dict["id"]] = row_dict

        final_results = {}

        for img_id, _p_name in person_images.items():
            if img_id in fts_results:
                row = fts_results[img_id]
                row["score"] = row["score"] - 1000.0
                final_results[img_id] = row
            else:
                cursor.execute("SELECT * FROM images WHERE id = ?", (img_id,))
                row = cursor.fetchone()
                if row:
                    row_dict = dict(row)
                    row_dict["score"] = -500.0
                    row_dict["file_size"] = row_dict.get("size_bytes")
                    row_dict["photo_date"] = row_dict.get("created_time")
                    row_dict["updated_at"] = row_dict.get("modified_time")
                    final_results[img_id] = row_dict

        for img_id, row in fts_results.items():
            if img_id not in final_results:
                final_results[img_id] = row

        sorted_results = sorted(final_results.values(), key=lambda x: x.get("score", 0.0))
        return sorted_results[:limit]

    def _search_like(self, terms: list[str], limit: int = 20) -> list[dict]:
        """使用 LIKE 在多个文本字段中进行模糊搜索。

        每个关键词在 scene/objects/style/location_time/wallpaper_hint/file_name
        任一字段中匹配即算命中，多个关键词之间为 OR 关系。

        Args:
            terms: 搜索关键词列表。
            limit: 最大返回条数。

        Returns:
            匹配的图片记录列表。
        """
        fields = ["scene", "objects", "style", "location_time", "wallpaper_hint", "file_name"]
        conditions = []
        params = []
        for term in terms:
            field_conditions = [f"{f} LIKE ?" for f in fields]
            conditions.append(f"({' OR '.join(field_conditions)})")
            params.extend([f"%{term}%"] * len(fields))

        where_clause = " OR ".join(conditions)
        cursor = self._conn.execute(
            f"SELECT * FROM images WHERE {where_clause} LIMIT ?",
            params + [limit],
        )
        return [dict(row) for row in cursor.fetchall()]

    def search_by_filename(self, keyword: str, limit: int = 50) -> list[dict]:
        """按文件名模糊搜索，支持部分匹配。"""
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE file_name LIKE ? ORDER BY modified_time DESC LIMIT ?",
            (f"%{keyword}%", limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_description(self, image_id: int, fields: dict):
        """更新图片的描述字段（scene/objects/style/location_time/wallpaper_hint）"""
        allowed_fields = {"scene", "objects", "style", "location_time", "wallpaper_hint"}
        updates = {k: v for k, v in fields.items() if k in allowed_fields}
        if not updates:
            return

        # 构建 SET 子句
        set_parts = []
        values = []
        for k, v in updates.items():
            set_parts.append(f"{k} = ?")
            values.append(v)

        set_clause = ", ".join(set_parts)
        values.append(image_id)

        self._conn.execute(f"UPDATE images SET {set_clause} WHERE id = ?", values)
        self._conn.commit()

    def update_embedding(self, image_id: int, embedding_bytes: bytes | None):
        """更新图片的向量字段"""
        self._conn.execute("UPDATE images SET text_embedding = ? WHERE id = ?", (embedding_bytes, image_id))
        self._conn.commit()

    def count(self) -> int:
        """返回已入库的图片总数。

        Returns:
            图片记录数量。
        """
        cursor = self._conn.execute("SELECT COUNT(*) FROM images")
        return cursor.fetchone()[0]

    def all_records(self) -> list[dict]:
        """获取所有图片记录，用于导出。

        Returns:
            所有记录的字典列表。
        """
        cursor = self._conn.execute("SELECT * FROM images ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        """关闭数据库连接。"""
        self._conn.close()
