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

        # 增加 is_favorite 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN is_favorite INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # 字段已存在

        # 增加 text_embedding 字段（如果是旧库）
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN text_embedding BLOB")
        except sqlite3.OperationalError:
            pass # 字段已存在

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
                # 执行更新
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
            (1 if is_favorite else 0, image_id)
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
                (image_id, bbox_str, embedding_blob)
            )
        self._conn.commit()

    def delete_face_embeddings_for_image(self, image_id: int) -> None:
        """删除指定图片已有的人脸特征记录，避免重复写入。"""

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
            image_ids
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

    def search_fts(self, query: str, limit: int = 50) -> list[dict]:
        """全文检索
        引入 jieba 分词，把长句或未加空格的词拆分成细粒度词汇组合
        例如：结婚照 -> 结婚 OR 照
        并且对输入先用空格分开的关键词进行 OR 逻辑组合。
        同时，如果 query 包含已知人物名，优先将其照片排在首位。
        """
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()
        
        # 0. 尝试从查询词中匹配人物名称
        person_images = {} # image_id -> person_name
        try:
            cursor.execute("SELECT id, name, cluster_id FROM persons WHERE is_deleted = 0")
            all_persons = cursor.fetchall()
            for p_row in all_persons:
                p_name = p_row["name"]
                if p_name in query:
                    # 获取该人物下的所有图片
                    cursor.execute(
                        "SELECT DISTINCT image_id FROM face_embeddings WHERE cluster_id = ?",
                        (p_row["cluster_id"],)
                    )
                    for img_row in cursor.fetchall():
                        person_images[img_row["image_id"]] = p_name
        except sqlite3.OperationalError:
            pass # 可能表不存在或结构有变
            
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
                
        # 第三步：如果没有任何分词并且也没有匹配到人物，直接返回空
        if not match_terms and not person_images:
            return []
            
        fts_results = {}
        if match_terms:
            # 严格匹配（各个输入分词之间为 AND）
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
                # 扩大召回范围以便和 person_images 融合
                cursor.execute(sql + f" LIMIT {limit * 5}", (match_query_and,))
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # AND 匹配的结果具有极大优势，分数给予奖励
                    row_dict["score"] = row_dict["score"] - 100.0
                    fts_results[row_dict["id"]] = row_dict
                    and_count += 1
            except sqlite3.OperationalError as e:
                import sys
                print(f"  ⚠️ FTS 精确查询语法有误: {match_query_and} ({e})", file=sys.stderr)
                
            # 如果精确匹配召回不足，且有多个关键词，则降级为 OR 查询进行补充
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

            # 兜底
            if not fts_results:
                cursor.execute(
                    "SELECT * FROM images WHERE scene LIKE ? OR objects LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", limit * 5)
                )
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    row_dict["score"] = -0.5
                    fts_results[row_dict["id"]] = row_dict
        
        # 融合 person_images 和 fts_results
        final_results = {}
        
        # 1. 既包含目标人物又命中 FTS 的图片（最高级：-1000 + 原始分数）
        for img_id, p_name in person_images.items():
            if img_id in fts_results:
                row = fts_results[img_id]
                row["score"] = row["score"] - 1000.0
                final_results[img_id] = row
            else:
                # 2. 属于目标人物但未命中 FTS，我们需从 images 表中拉取（次高级：-500.0）
                cursor.execute("SELECT * FROM images WHERE id = ?", (img_id,))
                row = cursor.fetchone()
                if row:
                    row_dict = dict(row)
                    row_dict["score"] = -500.0
                    row_dict["file_size"] = row_dict.get("size_bytes")
                    row_dict["photo_date"] = row_dict.get("created_time")
                    row_dict["updated_at"] = row_dict.get("modified_time")
                    final_results[img_id] = row_dict

        # 3. 剩下纯命中 FTS 的图片
        for img_id, row in fts_results.items():
            if img_id not in final_results:
                final_results[img_id] = row
                
        # 排序：按照 score 升序（分越小越靠前）
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
