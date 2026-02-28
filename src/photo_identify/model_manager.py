"""模型管理模块：管理 AI 模型配置的增删改查，以及 APIkey 状态检测。

使用独立的 SQLite 数据库 (models.db) 存储模型配置，
与图片数据库分离，便于独立管理。
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional


# 默认预设模型数据
_DEFAULT_MODELS = [
    {
        "type": "视觉模型",
        "name": "THUDM/GLM-4.1V-9B-Thinking",
        "model_id": "THUDM/GLM-4.1V-9B-Thinking",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_var": "SILICONFLOW_API_KEY",
    },
    {
        "type": "文本模型",
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key_var": "SILICONFLOW_API_KEY",
    },
    {
        "type": "文本模型",
        "name": "讯飞云Kimi-K2.5",
        "model_id": "xopkimik25",
        "base_url": "https://maas-api.cn-huabei-1.xf-yun.com/v2",
        "api_key_var": "XFYUN_API_KEY",
    },
]


class ModelManager:
    """模型配置管理器，封装 SQLite CRUD 操作。"""

    def __init__(self, db_path: str):
        """初始化模型管理器。

        Args:
            db_path: models.db 的路径。若文件不存在则自动创建并插入预设数据。
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_table()

    # ── 内部辅助 ──────────────────────────────────────────────

    def _init_table(self):
        """建表，并在数据库为空时插入预设数据。"""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,
                name        TEXT NOT NULL,
                model_id    TEXT NOT NULL,
                base_url    TEXT NOT NULL,
                api_key_var TEXT NOT NULL
            )
        """)
        self._conn.commit()

        # 若表为空，插入预设数据
        count = self._conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        if count == 0:
            for m in _DEFAULT_MODELS:
                self._conn.execute(
                    "INSERT INTO models (type, name, model_id, base_url, api_key_var) VALUES (?,?,?,?,?)",
                    (m["type"], m["name"], m["model_id"], m["base_url"], m["api_key_var"]),
                )
            self._conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """将数据库行转为字典，附加 api_key_status 字段。"""
        d = dict(row)
        d["api_key_status"] = self.check_api_key_status(d["api_key_var"])
        return d

    # ── 公共接口 ──────────────────────────────────────────────

    def get_all_models(self) -> list[dict]:
        """获取所有模型配置，按 id 排序。"""
        rows = self._conn.execute("SELECT * FROM models ORDER BY id").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_models_by_type(self, model_type: str) -> list[dict]:
        """获取指定类型的模型列表。

        Args:
            model_type: "视觉模型" 或 "文本模型"

        Returns:
            模型字典列表。
        """
        rows = self._conn.execute(
            "SELECT * FROM models WHERE type=? ORDER BY id", (model_type,)
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_model_by_id(self, model_db_id: int) -> Optional[dict]:
        """根据数据库 id 获取单条模型配置。"""
        row = self._conn.execute(
            "SELECT * FROM models WHERE id=?", (model_db_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_model_by_model_id(self, model_id: str) -> Optional[dict]:
        """根据 model_id 字段查找模型配置。"""
        row = self._conn.execute(
            "SELECT * FROM models WHERE model_id=?", (model_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def add_model(
        self,
        model_type: str,
        name: str,
        model_id: str,
        base_url: str,
        api_key_var: str,
    ) -> int:
        """添加新模型配置。

        Returns:
            新插入记录的 id。
        """
        cursor = self._conn.execute(
            "INSERT INTO models (type, name, model_id, base_url, api_key_var) VALUES (?,?,?,?,?)",
            (model_type, name, model_id, base_url, api_key_var),
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_model(
        self,
        model_db_id: int,
        model_type: str,
        name: str,
        model_id: str,
        base_url: str,
        api_key_var: str,
    ) -> None:
        """更新模型配置。"""
        self._conn.execute(
            "UPDATE models SET type=?, name=?, model_id=?, base_url=?, api_key_var=? WHERE id=?",
            (model_type, name, model_id, base_url, api_key_var, model_db_id),
        )
        self._conn.commit()

    def delete_model(self, model_db_id: int) -> None:
        """删除模型配置。"""
        self._conn.execute("DELETE FROM models WHERE id=?", (model_db_id,))
        self._conn.commit()

    @staticmethod
    def _get_env_from_registry(var_name: str) -> str:
        """在 Windows 上直接从注册表读取用户级和系统级环境变量。

        绕过 os.environ 快照限制，使运行中的进程能感知系统设置面板中新添加的变量。
        先查用户级 (HKCU)，再查系统级 (HKLM)，若均无则返回空字符串。
        非 Windows 平台直接返回空字符串（由调用方回退到 os.environ）。
        """
        try:
            import winreg
        except ImportError:
            return ""

        # 用户级环境变量
        reg_paths = [
            (winreg.HKEY_CURRENT_USER,  r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        ]
        for hive, path in reg_paths:
            try:
                with winreg.OpenKey(hive, path) as key:
                    value, _ = winreg.QueryValueEx(key, var_name)
                    if value:
                        return str(value)
            except (FileNotFoundError, OSError):
                continue
        return ""

    @classmethod
    def get_api_key_value(cls, api_key_var: str) -> str:
        """读取环境变量的实际值，优先从注册表（Windows），再回退到 os.environ。

        Args:
            api_key_var: 环境变量名称，如 "SILICONFLOW_API_KEY"。

        Returns:
            环境变量值字符串，找不到时返回空字符串。
        """
        # 优先从注册表读（可感知运行期间新添加的系统变量）
        reg_value = cls._get_env_from_registry(api_key_var)
        if reg_value.strip():
            return reg_value.strip()
        # 回退：使用进程继承的环境（非 Windows，或注册表未找到时）
        return os.environ.get(api_key_var, "").strip()

    @classmethod
    def check_api_key_status(cls, api_key_var: str) -> bool:
        """检查指定系统环境变量是否存在且非空。

        Args:
            api_key_var: 环境变量名称，如 "SILICONFLOW_API_KEY"。

        Returns:
            True 表示环境变量存在且非空，False 表示缺失或为空。
        """
        return bool(cls.get_api_key_value(api_key_var))

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()


def get_model_db_path(app_db_path: str) -> str:
    """根据主数据库路径推导出 models.db 的路径（同目录下）。

    Args:
        app_db_path: 主应用数据库路径（photo_identify.db）

    Returns:
        models.db 的绝对路径字符串。
    """
    return str(Path(app_db_path).resolve().parent / "models.db")
