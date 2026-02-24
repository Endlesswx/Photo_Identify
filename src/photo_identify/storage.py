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
    analyzed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_md5 ON images(md5);
CREATE INDEX IF NOT EXISTS idx_images_path ON images(path);
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
        cursor.executescript(_FTS_SCHEMA_SQL)
        cursor.executescript(_FTS_TRIGGER_INSERT)
        cursor.executescript(_FTS_TRIGGER_DELETE)
        cursor.executescript(_FTS_TRIGGER_UPDATE)
        self._conn.commit()

    def get_known_md5s(self) -> set[str]:
        """获取所有已入库图片的 MD5 集合，用于批量跳过检测。

        Returns:
            MD5 字符串的集合。
        """
        cursor = self._conn.execute("SELECT md5 FROM images")
        return {row[0] for row in cursor.fetchall()}

    def get_known_paths(self) -> dict[str, tuple[int, float]]:
        """获取所有已入库图片的路径及其 size+mtime 信息，用于快速跳过判断。

        Returns:
            字典，key 为路径，value 为 (size_bytes, modified_time 字符串)。
        """
        cursor = self._conn.execute("SELECT path, size_bytes, modified_time FROM images")
        return {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

    def has_md5(self, md5: str) -> bool:
        """检查指定 MD5 是否已入库。

        Args:
            md5: 图片内容的 MD5 哈希值。

        Returns:
            是否存在。
        """
        cursor = self._conn.execute("SELECT 1 FROM images WHERE md5 = ?", (md5,))
        return cursor.fetchone() is not None

    def upsert(self, record: dict):
        """插入或更新一条图片分析记录（按 MD5 去重）。

        Args:
            record: 包含所有字段的字典。
        """
        objects_str = json.dumps(record.get("objects", []), ensure_ascii=False) if isinstance(record.get("objects"), list) else record.get("objects", "")
        self._conn.execute(
            """
            INSERT INTO images (
                path, file_name, size_bytes, md5, sha256,
                width, height, image_mode, image_format, exif_json,
                created_time, modified_time,
                scene, objects, style, location_time, wallpaper_hint,
                llm_raw, analyzed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(md5) DO UPDATE SET
                path = excluded.path,
                file_name = excluded.file_name,
                size_bytes = excluded.size_bytes,
                analyzed_at = excluded.analyzed_at
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
            ),
        )
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
