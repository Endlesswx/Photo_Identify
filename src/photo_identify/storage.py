"""SQLite 存储层，负责图片分析结果的持久化与全文搜索。

使用 SQLite + FTS5 实现：
- 增量写入（单条 INSERT，无需全量序列化）
- 按 MD5 内容哈希去重
- FTS5 全文搜索支持口语化检索
"""

import json
import sqlite3
import threading
import jieba
from pathlib import Path
# from photo_identify.image_utils import is_valid_image # This import was in the user's snippet but not used in the provided code. Keeping it commented out.


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
    face_scanned INTEGER DEFAULT 0
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
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER UNIQUE,
    name TEXT NOT NULL,
    cover_image_id INTEGER,
    cover_face_id INTEGER,
    sort_order INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
CREATE INDEX IF NOT EXISTS idx_face_image_id ON face_embeddings(image_id);
CREATE INDEX IF NOT EXISTS idx_face_cluster_id ON face_embeddings(cluster_id);
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
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        """创建数据库表和全文搜索索引。"""
        cursor = self._conn.cursor()
        cursor.executescript(_SCHEMA_SQL)
        # 增加 face_scanned 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN face_scanned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # 字段已存在
            
        # 增加 sort_order 字段
        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN sort_order INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # 字段已存在
            
        # 增加 is_deleted 字段
        try:
            cursor.execute("ALTER TABLE persons ADD COLUMN is_deleted INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # 字段已存在
            
        cursor.executescript(_FTS_SCHEMA_SQL)
        cursor.executescript(_FTS_TRIGGER_INSERT)
        cursor.executescript(_FTS_TRIGGER_DELETE)
        cursor.executescript(_FTS_TRIGGER_UPDATE)

        # 持续失败的文件记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skipped_files (
                path TEXT PRIMARY KEY,
                reason TEXT,
                skipped_at TEXT
            )
        """)

        self._conn.commit()

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
                # 执行更新
                cursor.execute(
                    """
                    UPDATE images SET
                        path = ?,
                        file_name = ?,
                        size_bytes = ?,
                        analyzed_at = ?,
                        face_scanned = 1
                    WHERE md5 = ?
                    """,
                    (
                        record.get("path", ""),
                        record.get("file_name", ""),
                        record.get("size_bytes"),
                        record.get("analyzed_at", ""),
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
                        analyzed_at = ?
                    WHERE md5 = ?
                    """,
                    (
                        record.get("path", ""),
                        record.get("file_name", ""),
                        record.get("size_bytes"),
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
                    llm_raw, analyzed_at, face_scanned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            image_id = cursor.lastrowid or 0
            
        self._conn.commit()
        return int(image_id)

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
                (image_id, bbox_str, embedding_blob)
            )
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
        """更新人脸聚类结果并同步至 persons 表。
        
        Args:
            cluster_mapping: {face_id: cluster_id}
        """
        if not cluster_mapping:
            return
            
        cursor = self._conn.cursor()
        
        # 1. 备份当前 persons 里的所有数据（以 cover_face_id 为锚点识别身份）
        cursor.execute("SELECT cover_face_id, name, sort_order, is_deleted FROM persons")
        named_faces = cursor.fetchall()
        # {face_id: (name, sort_order, is_deleted)}
        custom_names = {row[0]: (row[1], row[2], row[3]) for row in named_faces if row[0] is not None}
        
        # 清空现有的 persons 表（因为 DBSCAN 每次重新聚类的 cluster_id 顺序和意义都会完全改变）
        cursor.execute("DELETE FROM persons")
        
        # 2. 更新 face_embeddings 的 cluster_id
        for face_id, cluster_id in cluster_mapping.items():
            cursor.execute("UPDATE face_embeddings SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
            
        # 3. 找到所有独立且大于 0 的 cluster_id (排除 -1 噪点)
        unique_clusters = set(cid for cid in cluster_mapping.values() if cid >= 0)
        
        # 4. 为新的 cluster_id 在 persons 表中创建记录
        for cid in unique_clusters:
            # 找出该 cluster 下所有的 face_id
            cursor.execute("SELECT id, image_id FROM face_embeddings WHERE cluster_id = ?", (cid,))
            cluster_faces = cursor.fetchall()
            if not cluster_faces:
                continue
                
            # 尝试在这个类中找一个已经有自定义名字的人脸
            person_name = None
            sort_order = 0
            is_deleted = 0
            cover_face_id = cluster_faces[0][0]
            cover_image_id = cluster_faces[0][1]
            
            # 优先找被重命名过或者改过设定的
            for f_id, i_id in cluster_faces:
                if f_id in custom_names:
                    saved_name, saved_sort, saved_del = custom_names[f_id]
                    # 如果有多个历史匹配，优先保留 非“人物_”开头的
                    if person_name is None or (not saved_name.startswith("人物_")):
                        person_name = saved_name
                        sort_order = saved_sort
                        is_deleted = saved_del
                        cover_face_id = f_id
                        cover_image_id = i_id
                    
            if not person_name:
                person_name = f"人物_{cid:03d}"
                
            cursor.execute(
                "INSERT INTO persons (cluster_id, name, cover_image_id, cover_face_id, sort_order, is_deleted) VALUES (?, ?, ?, ?, ?, ?)",
                (cid, person_name, cover_image_id, cover_face_id, sort_order, is_deleted)
            )
        
        self._conn.commit()

    def get_all_persons(self, include_deleted: bool = False) -> list[dict]:
        """获取所有聚类人物列表（优先按 sort_order 升序，其次按该人照片数倒序）。"""
        where_clause = "" if include_deleted else "WHERE p.is_deleted = 0"
        cursor = self._conn.execute(
            f"""
            SELECT p.id, p.cluster_id, p.name, p.cover_image_id, p.cover_face_id,
                   p.sort_order, p.is_deleted,
                   i.path, fe.bbox,
                   (SELECT COUNT(DISTINCT image_id) FROM face_embeddings WHERE cluster_id = p.cluster_id) as face_count
            FROM persons p
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings fe ON p.cover_face_id = fe.id
            {where_clause}
            ORDER BY p.sort_order ASC, face_count DESC, p.id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_deleted_persons(self) -> list[dict]:
        """获取回收站内的人物列表，按照片数倒序。"""
        cursor = self._conn.execute(
            """
            SELECT p.id, p.cluster_id, p.name, p.cover_image_id, p.cover_face_id,
                   i.path, fe.bbox,
                   (SELECT COUNT(DISTINCT image_id) FROM face_embeddings WHERE cluster_id = p.cluster_id) as face_count
            FROM persons p
            LEFT JOIN images i ON p.cover_image_id = i.id
            LEFT JOIN face_embeddings fe ON p.cover_face_id = fe.id
            WHERE p.is_deleted = 1
            ORDER BY face_count DESC, p.id ASC
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_person_sort_order(self, person_ids_in_order: list[int]):
        """批量更新人物排序顺序"""
        cursor = self._conn.cursor()
        for idx, person_id in enumerate(person_ids_in_order):
            cursor.execute("UPDATE persons SET sort_order = ? WHERE id = ?", (idx, person_id))
        self._conn.commit()

    def update_person_cover(self, person_id: int, cover_image_id: int, cover_face_id: int):
        """设置某人物的封面头像"""
        self._conn.execute(
            "UPDATE persons SET cover_image_id = ?, cover_face_id = ? WHERE id = ?",
            (cover_image_id, cover_face_id, person_id)
        )
        self._conn.commit()

    def set_person_deleted(self, person_id: int, is_deleted: int = 1):
        """将人物移入/移出回收站"""
        self._conn.execute("UPDATE persons SET is_deleted = ? WHERE id = ?", (is_deleted, person_id))
        self._conn.commit()

        
    def get_images_by_person(self, cluster_id: int) -> list[dict]:
        """获取某个人物相关的图片。"""
        sql = """
        SELECT DISTINCT i.* 
        FROM face_embeddings fe
        JOIN images i ON fe.image_id = i.id
        WHERE fe.cluster_id = ?
        ORDER BY i.modified_time DESC
        """
        cursor = self._conn.execute(sql, (cluster_id,))
        return [dict(row) for row in cursor.fetchall()]

    def update_person_name(self, person_id: int, new_name: str):
        """重命名人物"""
        self._conn.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
        self._conn.commit()

    def search_fts(self, query: str, limit: int = 50) -> list[dict]:
        """全文检索
        引入 jieba 分词，把长句或未加空格的词拆分成细粒度词汇组合
        例如：结婚照 -> 结婚 OR 照
        并且对输入先用空格分开的关键词进行 OR 逻辑组合。
        """
        # 第一步：把用户输入用空白符分割开
        parts = query.split()
        match_terms = []
        for p in parts:
            # 第二步：对每一部分使用搜索引擎模式结巴分词
            cut_words = list(jieba.cut_for_search(p))
            if cut_words:
                # 给每个切出来的词加上 * 通配符，内部再打上 OR
                group = " OR ".join([f'"{w}"*' for w in cut_words])
                match_terms.append(f"({group})")

        if not match_terms:
            return []

        # 所有子分组的关键字组合之间采取 OR
        # 因为利用了 SQLite FTS5 的 BM25 score，越精准覆盖多的记录自然会排越前
        # 但如果是 AND 匹配大模型发散扩充的近义词就没有任何数据能全中
        match_query = " OR ".join(match_terms)
        
        # 调试观察最终在本地生成的匹配语法
        print(f"  🔍 FTS 分词转换: \"{query}\" -> \"{match_query}\"")

        # 为了获得最新的 file_name 等，采用 JOIN 形式
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
        ORDER BY score
        LIMIT ?
        """
        # The original code used self.get_connection(), which is not defined.
        # Assuming self._conn is the correct connection object.
        self._conn.row_factory = sqlite3.Row # Ensure row_factory is set for this connection
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, (match_query, limit))
            results = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError as e:
            import sys
            print(f"  ⚠️ FTS 查询语法有误: {match_query} ({e})", file=sys.stderr)
            # 后备方案：退化为最原始简单的 LIKE
            cursor.execute(
                "SELECT * FROM images WHERE scene LIKE ? OR objects LIKE ? LIMIT ?",
                (f"%{query}%", f"%{query}%", limit)
            )
            results = [dict(row) for row in cursor.fetchall()]
        return results

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
