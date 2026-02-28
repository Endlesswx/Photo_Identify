import configparser
import math
import os
import subprocess
import sys
import threading
import time
import io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from PIL import Image, ImageTk

from photo_identify.image_utils import get_image_frame_bytes
from photo_identify.config import (
    DEFAULT_DB_PATH,
    DEFAULT_VIDEO_FRAME_INTERVAL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TPM_LIMIT,
)
from photo_identify.model_manager import ModelManager, get_model_db_path
from photo_identify.search import search
from photo_identify.scanner import scan


class StdoutRedirector:
    """重定向标准输出到 tkinter Text UI 组件的包装器"""
    def __init__(self, text_widget: tk.scrolledtext.ScrolledText):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        if not string:
            return
            
        def _update_ui(s=string):
            self.text_widget.configure(state=tk.NORMAL)
            if '\r' in s:
                # 遇到 \r 表示要回到行首覆盖当前行，终端进度条常见做法
                lines = s.split('\r')
                # 取最后的有意义文本（因为 \r 之后的内容会覆盖前面的）
                final_text = lines[-1] if lines[-1] else (lines[-2] if len(lines) > 1 else "")
                
                # 删除最后一行（从前一个换行符到末尾）并替换
                self.text_widget.delete("end-1c linestart", "end-1c")
                self.text_widget.insert("end-1c", final_text)
            else:
                self.text_widget.insert(tk.END, s)
                
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)
            
        # 安全地将其放入 Tkinter 事件队列
        self.text_widget.after(0, _update_ui)

    def flush(self):
        pass


class ModelDialog(tk.Toplevel):
    """新增/编辑模型的对话框。"""

    MODEL_TYPES = ["视觉模型", "文本模型"]

    def __init__(self, parent, title: str, model_data: dict = None):
        """
        Args:
            parent: 父窗口。
            title: 对话框标题。
            model_data: 若为编辑模式，传入现有模型字典；新增时传 None。
        """
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.grab_set()  # 模态
        self.result = None  # 用户点保存后存变量值

        # ── 变量 ──
        self._type_var = tk.StringVar(value=model_data["type"] if model_data else self.MODEL_TYPES[0])
        self._name_var = tk.StringVar(value=model_data.get("name", "") if model_data else "")
        self._model_id_var = tk.StringVar(value=model_data.get("model_id", "") if model_data else "")
        self._base_url_var = tk.StringVar(value=model_data.get("base_url", "") if model_data else "")
        self._api_key_var_var = tk.StringVar(value=model_data.get("api_key_var", "") if model_data else "")

        # ── 布局 ──
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        labels = ["模型类型:", "模型名称:", "模型ID:", "接口地址:", "API变量名:"]
        row = 0
        for lbl in labels:
            ttk.Label(frame, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=6, padx=(0, 10))
            row += 1

        # 类型下拉
        ttk.Combobox(
            frame, textvariable=self._type_var,
            values=self.MODEL_TYPES, state="readonly", width=30
        ).grid(row=0, column=1, sticky=tk.W, pady=6)

        # 文本字段
        for i, var in enumerate([self._name_var, self._model_id_var, self._base_url_var, self._api_key_var_var], 1):
            ttk.Entry(frame, textvariable=var, width=40).grid(row=i, column=1, sticky=tk.W, pady=6)

        # 底部按钮
        btn_frame = ttk.Frame(self, padding=(20, 0, 20, 15))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="✔ 保存", command=self._on_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="✖ 取消", command=self.destroy).pack(side=tk.RIGHT)

        self._center(parent)

    def _center(self, parent):
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    def _on_save(self):
        name = self._name_var.get().strip()
        model_id = self._model_id_var.get().strip()
        base_url = self._base_url_var.get().strip()
        api_key_var = self._api_key_var_var.get().strip()

        if not all([name, model_id, base_url, api_key_var]):
            messagebox.showwarning("警告", "所有字段均不能为空！", parent=self)
            return

        self.result = {
            "type": self._type_var.get(),
            "name": name,
            "model_id": model_id,
            "base_url": base_url,
            "api_key_var": api_key_var,
        }
        self.destroy()


class PhotoIdentifyGUI(tk.Tk):
    def __init__(self, db_path: str = str(DEFAULT_DB_PATH)):
        super().__init__()
        self.title("AI 图片语义检索与扫描")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.db_path = db_path
        self.search_dbs = [db_path] if db_path else []
        
        # 初始化模型管理器
        self._model_mgr = ModelManager(get_model_db_path(db_path))
        
        # Initialize variables
        self.current_results = []
        self.current_index = 0
        self.photo_image = None
        self.original_image = None
        self._resize_timer = None

        # 检索页：文本模型选择（存 model_id）
        self.search_model_id_var = tk.StringVar(value="")
        self.query_var = tk.StringVar(value="")
        self.search_mode_var = tk.StringVar(value="llm")
        self.status_var = tk.StringVar(value="准备就绪。")
        self.info_var = tk.StringVar(value="暂无图片")
        self.desc_var = tk.StringVar(value="")
        self.page_var = tk.StringVar(value="0 / 0")

        # 扫描页：视觉模型选择（存 model_id）
        self.scan_model_id_var = tk.StringVar(value="")

        # Notebook (Tab) Container
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: 检索
        self.search_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.search_tab, text="图片检索")
        self._init_search_tab()
        
        # Tab 2: 扫描
        self.scan_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.scan_tab, text="信息扫描")
        self._init_scan_tab()

        # Tab 3: 模型管理
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="模型管理")
        self._init_model_tab()

        # 加载上次保存的参数
        self._load_settings()

        # 关闭窗口时自动保存参数
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
    @property
    def _settings_path(self) -> str:
        """INI 配置文件路径，与默认数据库同目录。"""
        return os.path.join(os.path.dirname(os.path.abspath(self.db_path)), "photo_identify_gui.ini")

    def _load_settings(self):
        """从 INI 文件加载上次保存的界面参数。"""
        cfg = configparser.ConfigParser()
        if not os.path.exists(self._settings_path):
            return
        try:
            cfg.read(self._settings_path, encoding="utf-8")
        except Exception:
            return

        # Search tab
        if cfg.has_option("search", "model_id"):
            self._set_search_model_by_id(cfg.get("search", "model_id"))
        if cfg.has_option("search", "mode"):
            self.search_mode_var.set(cfg.get("search", "mode"))
        if cfg.has_option("search", "databases"):
            raw = cfg.get("search", "databases").strip()
            if raw:
                dbs = [p.strip() for p in raw.split("|") if p.strip()]
                self.search_dbs = dbs
                self.db_listbox.delete(0, tk.END)
                for db in dbs:
                    self.db_listbox.insert(tk.END, db)

        # Scan tab
        if cfg.has_option("scan", "model_id"):
            self._set_scan_model_by_id(cfg.get("scan", "model_id"))
        if cfg.has_option("scan", "db_path"):
            self.scan_db_var.set(cfg.get("scan", "db_path"))
        if cfg.has_option("scan", "paths"):
            raw = cfg.get("scan", "paths").strip()
            if raw:
                paths = [p.strip() for p in raw.split("|") if p.strip()]
                self.scan_paths = paths
                self.paths_listbox.delete(0, tk.END)
                for p in paths:
                    self.paths_listbox.insert(tk.END, p)
        if cfg.has_option("scan", "frame_interval"):
            self.scan_frame_interval_var.set(cfg.get("scan", "frame_interval"))

    def _save_settings(self):
        """将当前界面参数保存到 INI 文件。"""
        cfg = configparser.ConfigParser()

        cfg["search"] = {
            "model_id": self.search_model_id_var.get(),
            "mode": self.search_mode_var.get(),
            "databases": "|".join(self.search_dbs),
        }
        cfg["scan"] = {
            "model_id": self.scan_model_id_var.get(),
            "db_path": self.scan_db_var.get(),
            "paths": "|".join(self.scan_paths),
            "frame_interval": self.scan_frame_interval_var.get(),
        }

        try:
            with open(self._settings_path, "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception:
            pass  # 静默失败，不影响主流程

    def _on_close(self):
        """窗口关闭时保存设置并退出。"""
        self._save_settings()
        self._model_mgr.close()
        self.destroy()

    def open_env_vars(self):
        """打开 Windows 环境变量编辑窗口"""
        if sys.platform == "win32":
            subprocess.Popen("rundll32 sysdm.cpl,EditEnvironmentVariables")
            messagebox.showinfo("提示", "修改环境变量后，可能需要重启本应用才能生效。")
        else:
            messagebox.showinfo("提示", "仅在 Windows 下支持快速打开环境变量编辑窗口。")

    # ── 工具方法：下拉列表数据填充 ──────────────────────────────

    def _get_text_models(self) -> list[dict]:
        """获取所有文本模型列表。"""
        return self._model_mgr.get_models_by_type("文本模型")

    def _get_vision_models(self) -> list[dict]:
        """获取所有视觉模型列表。"""
        return self._model_mgr.get_models_by_type("视觉模型")

    def _refresh_search_model_combo(self):
        """刷新检索页模型下拉列表。"""
        models = self._get_text_models()
        self._search_model_map = {m["name"]: m["model_id"] for m in models}
        names = list(self._search_model_map.keys())
        self.search_model_combo["values"] = names
        # 保持当前选中的 model_id
        current_id = self.search_model_id_var.get()
        matched_name = next((m["name"] for m in models if m["model_id"] == current_id), None)
        if matched_name:
            self.search_model_combo.set(matched_name)
        elif names:
            self.search_model_combo.set(names[0])
            self.search_model_id_var.set(self._search_model_map[names[0]])

    def _refresh_scan_model_combo(self):
        """刷新扫描页模型下拉列表。"""
        models = self._get_vision_models()
        self._scan_model_map = {m["name"]: m["model_id"] for m in models}
        names = list(self._scan_model_map.keys())
        self.scan_model_combo["values"] = names
        # 保持当前选中的 model_id
        current_id = self.scan_model_id_var.get()
        matched_name = next((m["name"] for m in models if m["model_id"] == current_id), None)
        if matched_name:
            self.scan_model_combo.set(matched_name)
        elif names:
            self.scan_model_combo.set(names[0])
            self.scan_model_id_var.set(self._scan_model_map[names[0]])

    def _set_search_model_by_id(self, model_id: str):
        """根据 model_id 设置检索页下拉选中项。"""
        self.search_model_id_var.set(model_id)
        self._refresh_search_model_combo()

    def _set_scan_model_by_id(self, model_id: str):
        """根据 model_id 设置扫描页下拉选中项。"""
        self.scan_model_id_var.set(model_id)
        self._refresh_scan_model_combo()

    def _on_search_model_selected(self, event=None):
        """检索页下拉选择变化时，更新 model_id 变量。"""
        name = self.search_model_combo.get()
        if name and hasattr(self, "_search_model_map"):
            self.search_model_id_var.set(self._search_model_map.get(name, ""))

    def _on_scan_model_selected(self, event=None):
        """扫描页下拉选择变化时，更新 model_id 变量。"""
        name = self.scan_model_combo.get()
        if name and hasattr(self, "_scan_model_map"):
            self.scan_model_id_var.set(self._scan_model_map.get(name, ""))

    # ── 获取当前模型的 API 参数 ──────────────────────────────────

    def _get_search_api_params(self) -> tuple[str, str, str] | None:
        """获取检索页当前选中模型的 (model_id, base_url, api_key)。

        Returns:
            (model_id, base_url, api_key) 三元组，或失败时返回 None。
        """
        model_id = self.search_model_id_var.get().strip()
        if not model_id:
            messagebox.showwarning("警告", "请在「模型管理」页添加文本模型，并在检索页选择一个模型！")
            return None
        
        model = self._model_mgr.get_model_by_model_id(model_id)
        if not model:
            messagebox.showwarning("警告", f"未找到模型 {model_id!r} 的配置，请检查「模型管理」页。")
            return None
        
        api_key = ModelManager.get_api_key_value(model["api_key_var"])
        if not api_key:
            messagebox.showwarning(
                "警告",
                f"未找到环境变量 {model['api_key_var']}！\n"
                f"请在「模型管理」页点击「⚙ 环境变量设置」进行配置。\n"
                f"设置后点击「🔄 刷新状态」即可，无需重启应用。"
            )
            return None
        
        return model_id, model["base_url"], api_key

    def _get_scan_api_params(self) -> tuple[str, str, str] | None:
        """获取扫描页当前选中模型的 (model_id, base_url, api_key)。

        Returns:
            (model_id, base_url, api_key) 三元组，或失败时返回 None。
        """
        model_id = self.scan_model_id_var.get().strip()
        if not model_id:
            messagebox.showwarning("警告", "请在「模型管理」页添加视觉模型，并在扫描页选择一个模型！")
            return None
        
        model = self._model_mgr.get_model_by_model_id(model_id)
        if not model:
            messagebox.showwarning("警告", f"未找到模型 {model_id!r} 的配置，请检查「模型管理」页。")
            return None
        
        api_key = ModelManager.get_api_key_value(model["api_key_var"])
        if not api_key:
            messagebox.showwarning(
                "警告",
                f"未找到环境变量 {model['api_key_var']}！\n"
                f"请在「模型管理」页点击「⚙ 环境变量设置」进行配置。\n"
                f"设置后点击「🔄 刷新状态」即可，无需重启应用。"
            )
            return None
        
        return model_id, model["base_url"], api_key

    # ── Tab 1: 图片检索 ──────────────────────────────────────────

    def _init_search_tab(self):
        self.top_frame = ttk.Frame(self.search_tab, padding="10")

        # Row 0: 模型选择下拉
        ttk.Label(self.top_frame, text="文本模型:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self._search_model_map = {}
        self.search_model_combo = ttk.Combobox(self.top_frame, state="readonly", width=40)
        self.search_model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.search_model_combo.bind("<<ComboboxSelected>>", self._on_search_model_selected)
        # 初始化下拉选项
        self._refresh_search_model_combo()

        # Row 1: 检索数据库列表
        ttk.Label(self.top_frame, text="检索数据库:").grid(row=1, column=0, sticky=tk.W+tk.N, padx=5, pady=5)
        
        db_frame = ttk.Frame(self.top_frame)
        db_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        
        self.db_listbox = tk.Listbox(db_frame, height=3, selectmode=tk.EXTENDED, width=65)
        self.db_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(db_frame, orient=tk.VERTICAL, command=self.db_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.db_listbox.config(yscrollcommand=scrollbar.set)
        
        for db in self.search_dbs:
            self.db_listbox.insert(tk.END, db)
            
        btn_frame = ttk.Frame(self.top_frame)
        btn_frame.grid(row=1, column=3, sticky=tk.N+tk.W, padx=5, pady=5)
        self.add_db_btn = ttk.Button(btn_frame, text="➕ 添 加", command=self.add_search_db)
        self.add_db_btn.pack(fill=tk.X, pady=2)
        self.rm_db_btn = ttk.Button(btn_frame, text="➖ 移 除", command=self.remove_search_db)
        self.rm_db_btn.pack(fill=tk.X, pady=2)

        # Row 2: Query Keyword
        ttk.Label(self.top_frame, text="查询关键字:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.query_entry = ttk.Entry(self.top_frame, textvariable=self.query_var, width=50)
        self.query_entry.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        self.query_entry.bind("<Return>", lambda e: self.do_search())

        # Row 3: Search Mode Radio Buttons
        mode_frame = ttk.Frame(self.top_frame)
        mode_frame.grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)
        ttk.Label(mode_frame, text="搜索方案:").pack(side=tk.LEFT, padx=(0, 10))
        self.local_radio = ttk.Radiobutton(mode_frame, text="本地算法", variable=self.search_mode_var, value="local")
        self.local_radio.pack(side=tk.LEFT)
        self.llm_radio = ttk.Radiobutton(mode_frame, text="调用大模型匹配", variable=self.search_mode_var, value="llm")
        self.llm_radio.pack(side=tk.LEFT, padx=10)

        # Row 3 (continued): Search Button
        self.search_btn = ttk.Button(self.top_frame, text="🔍 搜索", command=self.do_search)
        self.search_btn.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)

        # Row 4: Status Feedback
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)

        # 主体预览区域
        self.main_frame = ttk.Frame(self.search_tab, padding="10")
        
        # 预览提示文本或信息
        self.info_label = ttk.Label(self.main_frame, textvariable=self.info_var, font=("Arial", 11, "bold"))
        self.info_label.pack(side=tk.TOP, pady=5)

        self.desc_label = ttk.Label(self.main_frame, textvariable=self.desc_var, wraplength=800, justify=tk.LEFT)
        self.desc_label.pack(side=tk.TOP, pady=5)

        # 图片画板
        self.canvas = tk.Canvas(self.main_frame, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 底部控制区
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.prev_btn = ttk.Button(self.bottom_frame, text="◀ 上一张", command=self.show_prev, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=10)

        self.page_label = ttk.Label(self.bottom_frame, textvariable=self.page_var, font=("Arial", 11))
        self.page_label.pack(side=tk.LEFT, padx=20)

        self.next_btn = ttk.Button(self.bottom_frame, text="下一张 ▶", command=self.show_next, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.open_dir_btn = ttk.Button(self.bottom_frame, text="📂 打开文件位置", command=self.open_file_location, state=tk.DISABLED)
        self.open_dir_btn.pack(side=tk.RIGHT, padx=10)

        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # ── Tab 2: 信息扫描 ──────────────────────────────────────────

    def _init_scan_tab(self):
        self.scan_db_var = tk.StringVar(value=self.db_path)
        self.scan_paths = []
        
        form_frame = ttk.Frame(self.scan_tab, padding="20")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        # 视觉模型下拉
        ttk.Label(form_frame, text="视觉模型:").grid(row=0, column=0, sticky=tk.W, pady=10)
        self._scan_model_map = {}
        self.scan_model_combo = ttk.Combobox(form_frame, state="readonly", width=40)
        self.scan_model_combo.grid(row=0, column=1, sticky=tk.W, pady=10)
        self.scan_model_combo.bind("<<ComboboxSelected>>", self._on_scan_model_selected)
        self._refresh_scan_model_combo()
        
        ttk.Label(form_frame, text="数据库路径:").grid(row=1, column=0, sticky=tk.W, pady=10)
        ttk.Entry(form_frame, textvariable=self.scan_db_var, width=40).grid(row=1, column=1, sticky=tk.W, pady=10)
        ttk.Button(form_frame, text="📂 浏览", command=self._browse_scan_db).grid(row=1, column=2, padx=10, sticky=tk.W, pady=10)
        
        ttk.Label(form_frame, text="扫描目录:").grid(row=2, column=0, sticky=tk.NW, pady=10)
        
        self.paths_listbox = tk.Listbox(form_frame, width=50, height=8)
        self.paths_listbox.grid(row=2, column=1, sticky=tk.W, pady=10)
        
        btn_frame = ttk.Frame(form_frame)
        btn_frame.grid(row=2, column=2, sticky=tk.NW, pady=10)
        ttk.Button(btn_frame, text="➕ 添加目录", command=self.add_scan_path).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="➖ 移除选中", command=self.remove_scan_path).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🗑 清空列表", command=self.clear_scan_paths).pack(fill=tk.X, pady=2)

        self.scan_frame_interval_var = tk.StringVar(value=str(DEFAULT_VIDEO_FRAME_INTERVAL))
        ttk.Label(form_frame, text="视频抽帧间隔(秒):").grid(row=3, column=0, sticky=tk.W, pady=10)
        ttk.Entry(form_frame, textvariable=self.scan_frame_interval_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=10)
        self.scan_btn = ttk.Button(form_frame, text="▶ 开始扫描 (见下方日志)", command=self.start_scan)
        self.scan_btn.grid(row=4, column=1, sticky=tk.EW, pady=20, ipady=5)
        
        self.restart_btn = ttk.Button(form_frame, text="🔄 重启扫描", command=self.restart_scan, state=tk.DISABLED)
        self.restart_btn.grid(row=4, column=2, sticky=tk.W, padx=10, pady=20, ipady=5)
        
        self._scan_cancel_event = None
        self._scan_thread_ref = None
        
        self.scan_status_var = tk.StringVar(value="准备就绪")
        ttk.Label(form_frame, textvariable=self.scan_status_var, foreground="blue").grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 终端输出重定向区域
        log_frame = ttk.LabelFrame(self.scan_tab, text="扫描日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tools inside log_frame
        log_tools_frame = ttk.Frame(log_frame)
        log_tools_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        ttk.Button(log_tools_frame, text="💾 导出日志保存", command=self.export_scan_logs).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

    # ── Tab 3: 模型管理 ──────────────────────────────────────────

    def _init_model_tab(self):
        """初始化模型管理标签页。"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.model_tab, padding=(10, 8, 10, 4))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="➕ 新增", command=self._model_add).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="✏️ 编辑", command=self._model_edit).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🗑 删除", command=self._model_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🔄 刷新状态", command=self._model_refresh).pack(side=tk.LEFT, padx=10)

        # 将环境变量设置按钮放右侧
        ttk.Button(toolbar, text="⚙ 环境变量设置", command=self.open_env_vars).pack(side=tk.RIGHT, padx=4)

        # 状态提示
        self._model_status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self._model_status_var, foreground="gray").pack(side=tk.RIGHT, padx=10)

        # 模型列表表格
        tree_frame = ttk.Frame(self.model_tab, padding=(10, 0, 10, 10))
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("type", "name", "model_id", "base_url", "api_key_var", "status")
        self._model_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")

        col_settings = [
            ("type",        "模型类型",     80,  tk.CENTER),
            ("name",        "模型名称",     160, tk.W),
            ("model_id",    "模型ID",       160, tk.W),
            ("base_url",    "接口地址",     200, tk.W),
            ("api_key_var", "API变量名",    150, tk.W),
            ("status",      "API状态",      70,  tk.CENTER),
        ]
        for col, heading, width, anchor in col_settings:
            self._model_tree.heading(col, text=heading)
            self._model_tree.column(col, width=width, anchor=anchor, minwidth=50)

        # 配置不同状态的行标签（颜色）
        self._model_tree.tag_configure("ok", foreground="green")
        self._model_tree.tag_configure("fail", foreground="red")

        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._model_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self._model_tree.xview)
        self._model_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._model_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        # 双击编辑
        self._model_tree.bind("<Double-1>", lambda e: self._model_edit())

        # 存储 tree item id → model db id 的映射
        self._tree_item_to_db_id: dict[str, int] = {}

        # 填充数据
        self._model_refresh()

    def _model_refresh(self):
        """刷新模型列表（重新读数据库 + 检查环境变量）。"""
        self._model_tree.delete(*self._model_tree.get_children())
        self._tree_item_to_db_id.clear()

        models = self._model_mgr.get_all_models()
        for m in models:
            status_ok = m["api_key_status"]
            status_text = "✅" if status_ok else "❌"
            tag = "ok" if status_ok else "fail"
            iid = self._model_tree.insert(
                "", tk.END,
                values=(m["type"], m["name"], m["model_id"], m["base_url"], m["api_key_var"], status_text),
                tags=(tag,)
            )
            self._tree_item_to_db_id[iid] = m["id"]

        ok_count = sum(1 for m in models if m["api_key_status"])
        self._model_status_var.set(f"共 {len(models)} 个模型 · {ok_count} 个 APIkey 已配置")

        # 同步刷新两个下拉列表
        self._refresh_search_model_combo()
        self._refresh_scan_model_combo()

    def _get_selected_model_db_id(self) -> int | None:
        """获取当前在 Treeview 中选中的数据库 id，未选中返回 None。"""
        selected = self._model_tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先选中一个模型！")
            return None
        return self._tree_item_to_db_id.get(selected[0])

    def _model_add(self):
        """弹出新增模型对话框。"""
        dlg = ModelDialog(self, "新增模型")
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self._model_mgr.add_model(r["type"], r["name"], r["model_id"], r["base_url"], r["api_key_var"])
            self._model_refresh()

    def _model_edit(self):
        """弹出编辑模型对话框。"""
        db_id = self._get_selected_model_db_id()
        if db_id is None:
            return
        model = self._model_mgr.get_model_by_id(db_id)
        if not model:
            return
        dlg = ModelDialog(self, "编辑模型", model_data=model)
        self.wait_window(dlg)
        if dlg.result:
            r = dlg.result
            self._model_mgr.update_model(db_id, r["type"], r["name"], r["model_id"], r["base_url"], r["api_key_var"])
            self._model_refresh()

    def _model_delete(self):
        """删除选中的模型（弹确认框）。"""
        db_id = self._get_selected_model_db_id()
        if db_id is None:
            return
        model = self._model_mgr.get_model_by_id(db_id)
        if not model:
            return
        if not messagebox.askyesno("确认删除", f"确定要删除模型「{model['name']}」吗？"):
            return
        self._model_mgr.delete_model(db_id)
        self._model_refresh()

    # ── 信息扫描相关方法 ──────────────────────────────────────────

    def export_scan_logs(self):
        log_content = self.log_text.get(1.0, tk.END).strip()
        if not log_content:
            messagebox.showinfo("提示", "目前没有日志可以导出。")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="导出扫描日志",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("Log Files", "*.log"), ("All Files", "*.*")],
            initialfile="scan_logs.txt"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(log_content)
                messagebox.showinfo("成功", f"日志已成功导出并保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"无法保存日志文件:\n{e}")

    def add_scan_path(self):
        path = filedialog.askdirectory(title="选择要扫描的文件夹")
        if path and path not in self.scan_paths:
            self.scan_paths.append(path)
            self.paths_listbox.insert(tk.END, path)

    def remove_scan_path(self):
        selection = self.paths_listbox.curselection()
        if selection:
            index = selection[0]
            self.paths_listbox.delete(index)
            self.scan_paths.pop(index)

    def clear_scan_paths(self):
        self.paths_listbox.delete(0, tk.END)
        self.scan_paths.clear()

    def _browse_scan_db(self):
        """通过文件对话框选择数据库路径。"""
        path = filedialog.asksaveasfilename(
            title="选择或新建数据库文件",
            defaultextension=".db",
            filetypes=[("SQLite DB", "*.db"), ("All Files", "*.*")],
            confirmoverwrite=False,
        )
        if path:
            self.scan_db_var.set(path)

    def start_scan(self):
        if not self.scan_paths:
            messagebox.showwarning("警告", "请先添加要扫描的目录！")
            return
            
        self._save_settings()
        
        # 从选中的视觉模型获取 API 参数
        api_params = self._get_scan_api_params()
        if api_params is None:
            return
        model_id, base_url, api_key = api_params
            
        self.scan_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.NORMAL)
        self.scan_status_var.set("正在后台扫描，查看下方日志区...")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        self._scan_cancel_event = threading.Event()
        # 代际计数器：防止旧线程的回调覆盖新扫描的 GUI 状态
        if not hasattr(self, '_scan_generation'):
            self._scan_generation = 0
        self._scan_generation += 1
        current_gen = self._scan_generation
        
        def _scan_thread():
            import logging
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            redirector = StdoutRedirector(self.log_text)
            
            # Redirect stdout and stderr to our text widget
            sys.stdout = redirector
            sys.stderr = redirector
            
            # Configure logging to emit to the new stdout
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            old_handlers = root_logger.handlers[:]
            for h in old_handlers:
                root_logger.removeHandler(h)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            root_logger.addHandler(stream_handler)
            
            print("==================================================")
            print(f"[{time.strftime('%H:%M:%S')}] 扫描进程启动...")
            print(f"使用模型: {model_id}")
            print(f"接口地址: {base_url}")
            print("正在遍历与收集指定目录内所有符合条件的多媒体文件。")
            print("如果您选择了体积庞大的盘符或含有万级文件的目录，这可能需要数分钟的搜寻时间，请耐心等待...")
            print("==================================================\n")
            
            try:
                scan(
                    paths=self.scan_paths,
                    db_path=self.scan_db_var.get(),
                    api_key=api_key,
                    base_url=base_url,
                    model=model_id,
                    rpm_limit=DEFAULT_RPM_LIMIT,
                    tpm_limit=DEFAULT_TPM_LIMIT,
                    cancel_event=self._scan_cancel_event,
                    video_frame_interval=float(self.scan_frame_interval_var.get() or DEFAULT_VIDEO_FRAME_INTERVAL),
                )
                # 只有当前代际仍然有效时才更新状态
                if current_gen == self._scan_generation:
                    if self._scan_cancel_event and not self._scan_cancel_event.is_set():
                        self.after(0, lambda: self.scan_status_var.set("扫描完成！可以去搜索页检索了。"))
                    else:
                        self.after(0, lambda: self.scan_status_var.set("扫描已中止。"))
            except Exception as e:
                if current_gen == self._scan_generation:
                    self.after(0, lambda e=e: self.scan_status_var.set(f"扫描出错: {e}"))
                    self.after(0, lambda e=e: messagebox.showerror("错误", f"扫描异常:\n{e}"))
            finally:
                sys.stdout = original_stdout  # Restore
                sys.stderr = original_stderr
                root_logger.removeHandler(stream_handler)
                for h in old_handlers:
                    root_logger.addHandler(h)
                
                # 只有当前代际仍然有效时才恢复按钮状态
                if current_gen == self._scan_generation:
                    self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
                    self.after(0, lambda: self.restart_btn.config(state=tk.DISABLED))
                
        t = threading.Thread(target=_scan_thread, daemon=True)
        self._scan_thread_ref = t
        t.start()

    def restart_scan(self):
        """中止当前扫描并重新启动。"""
        if self._scan_cancel_event:
            self._scan_cancel_event.set()
        self.scan_status_var.set("正在停止当前扫描，请稍候...")
        self.restart_btn.config(state=tk.DISABLED)

        def _wait_and_restart():
            # 等待旧线程真正结束（最多 30 秒）
            if self._scan_thread_ref and self._scan_thread_ref.is_alive():
                self._scan_thread_ref.join(timeout=30)
            self.after(0, self.start_scan)

        threading.Thread(target=_wait_and_restart, daemon=True).start()

    # ── 图片检索相关方法 ──────────────────────────────────────────

    def add_search_db(self):
        files = filedialog.askopenfilenames(
            title="选择要添加的数据库文件",
            filetypes=[("SQLite DB", "*.db"), ("All Files", "*.*")]
        )
        if files:
            for f in files:
                if f not in self.search_dbs:
                    self.search_dbs.append(f)
                    self.db_listbox.insert(tk.END, f)

    def remove_search_db(self):
        selection = self.db_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先在列表中选中要移除的数据库。")
            return
        # 从后往前删防止索引变化
        for index in reversed(selection):
            db_path = self.db_listbox.get(index)
            self.db_listbox.delete(index)
            if db_path in self.search_dbs:
                self.search_dbs.remove(db_path)

    def toggle_state(self, state):
        """控制搜索时的组件状态"""
        self.search_btn.config(state=state)
        self.query_entry.config(state=state)
        if hasattr(self, 'add_db_btn'):
            self.add_db_btn.config(state=state)
        if hasattr(self, 'rm_db_btn'):
            self.rm_db_btn.config(state=state)
        # Also disable/enable radio buttons
        if hasattr(self, 'local_radio'):
            self.local_radio.config(state=state)
        if hasattr(self, 'llm_radio'):
            self.llm_radio.config(state=state)

    def do_search(self):
        if not self.search_dbs:
            messagebox.showwarning("警告", "请至少添加一个数据库用于检索！")
            return
            
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("警告", "请输入查询关键字！")
            return

        self._save_settings()

        is_llm_mode = (self.search_mode_var.get() == "llm")
        
        if is_llm_mode:
            # 从选中的文本模型获取 API 参数
            api_params = self._get_search_api_params()
            if api_params is None:
                return
            model_id, base_url, api_key = api_params
        else:
            model_id, base_url, api_key = "", "", ""

        self.toggle_state(tk.DISABLED)
        if is_llm_mode:
            self.status_var.set("正在使用大模型理解查询并召回数据，请稍候...")
        else:
            self.status_var.set("正在使用本地算法检索，请稍候...")
            
        self.update_idletasks()

        # 在后台线程执行搜索避免阻塞 GUI
        def _search_thread():
            try:
                results = search(
                    query=query,
                    db_paths=self.search_dbs,
                    limit=30,  # 截取前 30 个给 LLM 做重排
                    smart=is_llm_mode,
                    api_key=api_key,
                    base_url=base_url,
                    model=model_id,
                    rerank=is_llm_mode,
                )
                self.after(0, self._on_search_done, results, None)
            except Exception as e:
                self.after(0, self._on_search_done, None, str(e))

        threading.Thread(target=_search_thread, daemon=True).start()

    def _on_search_done(self, results, error):
        self.toggle_state(tk.NORMAL)
        if error is not None:
            self.status_var.set(f"搜索失败: {error}")
            messagebox.showerror("错误", f"搜索执行发生错误：{error}")
            return

        if not results:
            self.status_var.set("搜索完成：未找到匹配的图片。")
            self.current_results = []
            self.current_index = 0
            self._update_display()
            return

        self.status_var.set(f"搜索完成：共找到并重排序 {len(results)} 张图片。")
        self.current_results = results
        self.current_index = 0
        self._update_display()

    def show_prev(self):
        if self.current_results and self.current_index > 0:
            self.current_index -= 1
            self._update_display()

    def show_next(self):
        if self.current_results and self.current_index < len(self.current_results) - 1:
            self.current_index += 1
            self._update_display()

    def _update_display(self):
        total = len(self.current_results)
        if total == 0:
            self.info_var.set("暂无图片")
            self.desc_var.set("")
            self.page_var.set("0 / 0")
            self.canvas.delete("all")
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.open_dir_btn.config(state=tk.DISABLED)
            return

        self.page_var.set(f"{self.current_index + 1} / {total}")
        self.prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_index < total - 1 else tk.DISABLED)
        self.open_dir_btn.config(state=tk.NORMAL)

        record = self.current_results[self.current_index]
        file_path = record.get("path", "")
        file_name = record.get("file_name", os.path.basename(file_path))
        scene = record.get("scene", "")
        objects = record.get("objects", "")
        
        # 由于是从数据库读出，objects 可能是 JSON 字符串
        if isinstance(objects, str) and objects.startswith("["):
            import json
            try:
                obj_list = json.loads(objects)
                objects = ", ".join(obj_list)
            except:
                pass
        elif isinstance(objects, list):
            objects = ", ".join(objects)

        self.info_var.set(file_name)
        desc_text = ""
        if scene:
            desc_text += f"场景: {scene}\n"
        if objects:
            desc_text += f"包含: {objects}"
        self.desc_var.set(desc_text)

        self._load_and_draw_image(file_path)

    def _load_and_draw_image(self, file_path: str):
        # 处理带有 "#t=" 等后缀的视频文件路径
        actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path

        if not os.path.isfile(actual_path):
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                text="图片文件不存在", fill="red", font=("Arial", 16)
            )
            return

        try:
            frame_bytes = get_image_frame_bytes(actual_path)
            self.original_image = Image.open(io.BytesIO(frame_bytes))
            self._resize_and_display()
        except Exception as e:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                text=f"无法加载图片: {e}", fill="red", font=("Arial", 14)
            )

    def on_canvas_resize(self, event):
        if self.current_results and hasattr(self, 'original_image'):
            # 防抖：避免调整窗口大小时频繁重绘
            if hasattr(self, '_resize_timer') and self._resize_timer is not None:
                try:
                    self.after_cancel(self._resize_timer)
                except ValueError:
                    pass
            self._resize_timer = self.after(100, self._resize_and_display)

    def _resize_and_display(self):
        if not hasattr(self, 'original_image'):
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        img_width, img_height = self.original_image.size
        # 计算缩放比例，保持长宽比
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = max(1, int(img_width * ratio))
        new_height = max(1, int(img_height * ratio))

        # 使用 LANCZOS 进行高质量缩放
        resized_img = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        # 居中显示
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.photo_image, anchor=tk.CENTER)

    def open_file_location(self):
        if not self.current_results:
            return
        record = self.current_results[self.current_index]
        file_path = record.get("path", "")
        
        # 处理带有 "#t=" 等后缀的视频文件路径
        actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
        
        if not os.path.exists(actual_path):
            messagebox.showerror("错误", "文件不存在！")
            return

        # 在 Windows 上选中文件并在资源管理器打开
        try:
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{os.path.normpath(actual_path)}"')
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", "-R", actual_path])
            else:  # Linux (fallback)
                subprocess.Popen(["xdg-open", os.path.dirname(actual_path)])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件位置: {e}")


def launch_gui(db_path: str = str(DEFAULT_DB_PATH)):
    """启动图形界面主循环"""
    app = PhotoIdentifyGUI(db_path)
    app.mainloop()

if __name__ == "__main__":
    launch_gui()
