import configparser
import math
import os
import subprocess
import sys
import threading
import time
import io
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from PIL import Image, ImageTk, ImageOps

from photo_identify.image_utils import get_image_frame_bytes
from photo_identify.config import (
    DEFAULT_DB_PATH,
    DEFAULT_VIDEO_FRAME_INTERVAL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TPM_LIMIT,
    DEFAULT_WORKERS,
)
from photo_identify.model_manager import ModelManager, get_model_db_path, MODEL_TYPES
from photo_identify.search import search
from photo_identify.scanner import scan


def crop_and_circle_face(image: Image.Image, bbox_str: str, size: int = 80) -> Image.Image:
    """根据 bbox 从图片裁剪人脸，并生成圆形带透明背景的图。"""
    import json
    from PIL import ImageDraw
    
    image = ImageOps.exif_transpose(image)
    
    try:
        bbox = json.loads(bbox_str)
        x1, y1, x2, y2 = bbox
        
        # 寻找中心点
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 获取最大边长并扩大1.6倍以包含整个头部
        w = x2 - x1
        h = y2 - y1
        side = max(w, h) * 1.6
        
        # 计算正方形边界
        left = int(cx - side / 2)
        top = int(cy - side / 2)
        right = int(cx + side / 2)
        bottom = int(cy + side / 2)
        
        # 如果超出边界，进行安全裁剪并填充为正方形
        # 更好的方法是直接用 image.crop，Pillow 对于超出边界的坐标会自动填充黑色像素（这正符合我们保持居中和正方形的需求）
        face_img = image.crop((left, top, right, bottom))
    except Exception:
        # 如果解析失败，则将原图等比例缩放居中裁剪成正方形
        min_dim = min(image.width, image.height)
        left = (image.width - min_dim) // 2
        top = (image.height - min_dim) // 2
        face_img = image.crop((left, top, left + min_dim, top + min_dim))
    
    face_img = face_img.resize((size, size), Image.Resampling.LANCZOS)
    
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size, size), fill=255)
    
    result = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    if face_img.mode != 'RGBA':
        face_img = face_img.convert('RGBA')
    result.paste(face_img, (0, 0), mask)
    
    return result

class TkLineWriter:
    """tqdm 兼容的文件对象，将输出更新到 Text 组件的指定行。

    支持多线程：所有 Tkinter 操作通过 after() 调度到主线程。
    内置节流（100ms），避免高频刷新导致 GUI 卡顿。
    """

    def __init__(self, text_widget, line_num: int):
        self.text_widget = text_widget
        self.line_num = line_num
        self._latest_text = ""
        self._update_pending = False
        self._lock = threading.Lock()

    def write(self, s):
        if not s:
            return
        # 清除 tqdm 输出的 ANSI 转义序列（如 \033[A 光标上移）
        text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)
        text = text.replace('\r', '').replace('\n', '').strip()
        if not text:
            return
        with self._lock:
            self._latest_text = text
            if not self._update_pending:
                self._update_pending = True
                self.text_widget.after(100, self._do_update)

    def _do_update(self):
        with self._lock:
            text = self._latest_text
            self._update_pending = False
        try:
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.delete(f"{self.line_num}.0", f"{self.line_num}.0 lineend")
            self.text_widget.insert(f"{self.line_num}.0", text)
            self.text_widget.configure(state=tk.DISABLED)
        except Exception:
            pass

    def flush(self):
        pass

    def isatty(self):
        return False


class ModelDialog(tk.Toplevel):
    """新增/编辑模型的对话框。"""

    def __init__(self, parent, title: str, model_data: dict | None = None):
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
        self._type_var = tk.StringVar(value=model_data["type"] if model_data else MODEL_TYPES[0])
        self._name_var = tk.StringVar(value=model_data.get("name", "") if model_data else "")
        self._model_id_var = tk.StringVar(value=model_data.get("model_id", "") if model_data else "")
        self._base_url_var = tk.StringVar(value=model_data.get("base_url", "") if model_data else "")
        self._api_key_var_var = tk.StringVar(value=model_data.get("api_key_var", "") if model_data else "")
        self._workers_var = tk.StringVar(value=str(model_data.get("workers", 4)) if model_data else "4")

        # ── 布局 ──
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        labels = ["模型类型:", "模型名称:", "模型ID:", "接口地址:", "API变量名:", "并发线程数:"]
        row = 0
        for lbl in labels:
            ttk.Label(frame, text=lbl).grid(row=row, column=0, sticky=tk.W, pady=6, padx=(0, 10))
            row += 1

        # 类型下拉
        ttk.Combobox(
            frame, textvariable=self._type_var,
            values=MODEL_TYPES, state="readonly", width=30
        ).grid(row=0, column=1, sticky=tk.W, pady=6)

        # 文本字段
        for i, var in enumerate([self._name_var, self._model_id_var, self._base_url_var, self._api_key_var_var], 1):
            ttk.Entry(frame, textvariable=var, width=40).grid(row=i, column=1, sticky=tk.W, pady=6)

        # 并发线程数
        self._workers_entry = ttk.Spinbox(frame, from_=1, to=32, textvariable=self._workers_var, width=10)
        self._workers_entry.grid(row=5, column=1, sticky=tk.W, pady=6)

        # API变量名旁的提示
        ttk.Label(frame, text="(本地模型可留空)", foreground="gray").grid(row=4, column=2, sticky=tk.W, padx=5)
        ttk.Label(frame, text="(本地模型固定为1)", foreground="gray").grid(row=5, column=2, sticky=tk.W, padx=5)

        # 底部按钮
        btn_frame = ttk.Frame(self, padding=(20, 0, 20, 15))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="✔ 保存", command=self._on_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="✖ 取消", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(btn_frame, text="🦙 Ollama 预设", command=self._fill_ollama_preset).pack(side=tk.LEFT, padx=5)

        self._api_key_var_var.trace_add("write", self._on_api_key_changed)
        self._on_api_key_changed()

        self._center(parent)

    def _on_api_key_changed(self, *args):
        if not self._api_key_var_var.get().strip():
            self._workers_var.set("1")
            self._workers_entry.state(["disabled"])
        else:
            self._workers_entry.state(["!disabled"])

    def _center(self, parent):
        """将对话框居中于父窗口。"""
        self.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width() // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    def _fill_ollama_preset(self):
        """一键填充 Ollama 本地模型预设值。"""
        self._type_var.set("多模态模型")
        self._base_url_var.set("http://localhost:11434/v1")
        self._api_key_var_var.set("")

    def _on_save(self):
        """校验并保存表单数据。"""
        name = self._name_var.get().strip()
        model_id = self._model_id_var.get().strip()
        base_url = self._base_url_var.get().strip()
        api_key_var = self._api_key_var_var.get().strip()
        workers = self._workers_var.get().strip()

        # 名称、模型ID、接口地址为必填；API变量名可为空（本地模型无需 API Key）
        if not all([name, model_id, base_url]):
            messagebox.showwarning("警告", "模型名称、模型ID、接口地址不能为空！", parent=self)
            return
            
        try:
            workers_int = int(workers)
        except ValueError:
            workers_int = 1 if not api_key_var else 4

        self.result = {
            "type": self._type_var.get(),
            "name": name,
            "model_id": model_id,
            "base_url": base_url,
            "api_key_var": api_key_var,
            "workers": workers_int,
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
        self.current_rotation = 0

        # 检索页：文本模型选择（存 model_id）
        self.search_model_id_var = tk.StringVar(value="")
        self.query_var = tk.StringVar(value="")
        self.search_mode_var = tk.StringVar(value="llm")
        self.search_limit_var = tk.StringVar(value="30")
        self.status_var = tk.StringVar(value="准备就绪。")
        self.info_var = tk.StringVar(value="暂无图片")
        self.desc_var = tk.StringVar(value="")
        self.page_var = tk.StringVar(value="0 / 0")

        # 扫描页：视觉模型选择（存 model_id）
        self.scan_model_id_var = tk.StringVar(value="")
        self.enable_face_scan_var = tk.BooleanVar(value=True)

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

        # Tab 3: 人物管理
        self.person_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.person_tab, text="人物管理")
        self._init_person_tab()

        # Tab 3.5: 照片收藏
        self.favorite_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.favorite_tab, text="照片收藏")
        self._init_favorite_tab()

        # Tab 4: 模型管理
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="模型管理")
        self._init_model_tab()

        # 加载上次保存的参数
        self._load_settings()

        # 关闭窗口时自动保存参数
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 绑定 Tab 切换事件
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self._person_tab_loaded = False
        
    def _on_tab_changed(self, event):
        selected_tab = self.notebook.select()
        if selected_tab == str(self.person_tab):
            if not getattr(self, "_person_tab_loaded", False):
                self._person_tab_loaded = True
                self._refresh_persons()
        elif selected_tab == str(self.favorite_tab):
            self._refresh_favorites()

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
        if cfg.has_option("search", "limit"):
            self.search_limit_var.set(cfg.get("search", "limit"))
        if cfg.has_option("search", "databases"):
            raw = cfg.get("search", "databases").strip()
            if raw:
                dbs = [p.strip() for p in raw.split("|") if p.strip()]
                self.search_dbs = dbs
                self.db_listbox.delete(0, tk.END)
                for db in dbs:
                    self.db_listbox.insert(tk.END, db)
        if cfg.has_option("search", "last_query"):
            self.query_var.set(cfg.get("search", "last_query"))

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
        if cfg.has_option("scan", "enable_face_scan"):
            self.enable_face_scan_var.set(cfg.getboolean("scan", "enable_face_scan"))

    def _save_settings(self):
        """将当前界面参数保存到 INI 文件。"""
        cfg = configparser.ConfigParser()

        cfg["search"] = {
            "model_id": self.search_model_id_var.get(),
            "mode": self.search_mode_var.get(),
            "limit": self.search_limit_var.get(),
            "databases": "|".join(self.search_dbs),
            "last_query": self.query_var.get(),
        }
        cfg["scan"] = {
            "model_id": self.scan_model_id_var.get(),
            "db_path": self.scan_db_var.get(),
            "paths": "|".join(self.scan_paths),
            "frame_interval": self.scan_frame_interval_var.get(),
            "enable_face_scan": str(self.enable_face_scan_var.get()),
        }

        try:
            with open(self._settings_path, "w", encoding="utf-8") as f:
                cfg.write(f)
        except Exception:
            pass  # 静默失败，不影响主流程

    def _on_close(self):
        """窗口关闭时保存设置并强制退出所有后台进程。"""
        try:
            self._save_settings()
        except Exception:
            pass
        if hasattr(self, '_scan_cancel_event') and self._scan_cancel_event:
            self._scan_cancel_event.set()
        try:
            self._model_mgr.close()
        except Exception:
            pass
        self.destroy()
        os._exit(0)

    def open_env_vars(self):
        """打开 Windows 环境变量编辑窗口"""
        if sys.platform == "win32":
            subprocess.Popen("rundll32 sysdm.cpl,EditEnvironmentVariables")
            messagebox.showinfo("提示", "修改环境变量后，可能需要重启本应用才能生效。")
        else:
            messagebox.showinfo("提示", "仅在 Windows 下支持快速打开环境变量编辑窗口。")

    # ── 工具方法：下拉列表数据填充 ──────────────────────────────

    def _get_text_models(self) -> list[dict]:
        """获取可用于文本检索的模型列表（文本模型 + 多模态模型）。"""
        return self._model_mgr.get_models_for_usage("text")

    def _get_vision_models(self) -> list[dict]:
        """获取可用于视觉扫描的模型列表（视觉模型 + 多模态模型）。"""
        return self._model_mgr.get_models_for_usage("vision")

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
            
        self._update_scan_workers_display()

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
            self._update_scan_workers_display()
            
    def _update_scan_workers_display(self):
        """更新扫描页的并发线程数显示"""
        model_id = self.scan_model_id_var.get()
        if not model_id:
            self.scan_workers_var.set("并发: -")
            return
            
        model = self._model_mgr.get_model_by_model_id(model_id)
        if model:
            workers = model.get("workers", 4)
            self.scan_workers_var.set(f"并发: {workers} 线程")
        else:
            self.scan_workers_var.set("并发: -")

    # ── 获取当前模型的 API 参数 ──────────────────────────────────

    def _get_model_api_params(self, model_id_var: tk.StringVar, usage_label: str) -> tuple[str, str, str, int] | None:
        """获取指定模型变量对应的 (model_id, base_url, api_key, workers)。

        Args:
            model_id_var: 存储当前选中 model_id 的 StringVar。
            usage_label: 用于提示信息的描述，如 "文本模型" 或 "视觉模型"。

        Returns:
            (model_id, base_url, api_key, workers) 四元组，或失败时返回 None。
        """
        model_id = model_id_var.get().strip()
        if not model_id:
            messagebox.showwarning("警告", f"请在「模型管理」页添加{usage_label}，并选择一个模型！")
            return None

        model = self._model_mgr.get_model_by_model_id(model_id)
        if not model:
            messagebox.showwarning("警告", f"未找到模型 {model_id!r} 的配置，请检查「模型管理」页。")
            return None

        workers = model.get("workers", 4)

        # 本地模型（如 Ollama）无需 API Key
        if model.get("is_local"):
            return model_id, model["base_url"], "", workers

        api_key = ModelManager.get_api_key_value(model["api_key_var"])
        if not api_key:
            messagebox.showwarning(
                "警告",
                f"未找到环境变量 {model['api_key_var']}！\n"
                f"请在「模型管理」页点击「⚙ 环境变量设置」进行配置。\n"
                f"设置后点击「🔄 刷新状态」即可，无需重启应用。"
            )
            return None

        return model_id, model["base_url"], api_key, workers

    def _get_search_api_params(self) -> tuple[str, str, str, int] | None:
        """获取检索页当前选中模型的 (model_id, base_url, api_key, workers)。"""
        return self._get_model_api_params(self.search_model_id_var, "文本/多模态模型")

    def _get_scan_api_params(self) -> tuple[str, str, str, int] | None:
        """获取扫描页当前选中模型的 (model_id, base_url, api_key, workers)。"""
        return self._get_model_api_params(self.scan_model_id_var, "视觉/多模态模型")

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

        ttk.Label(mode_frame, text="查询张数:").pack(side=tk.LEFT, padx=(20, 5))
        self.limit_entry = ttk.Spinbox(mode_frame, from_=1, to=1000, textvariable=self.search_limit_var, width=5)
        self.limit_entry.pack(side=tk.LEFT)

        # Row 3 (continued): Search Button
        self.search_btn = ttk.Button(self.top_frame, text="🔍 搜索", command=self.do_search)
        self.search_btn.grid(row=3, column=3, sticky=tk.W, padx=5, pady=5)

        # Row 4: Status Feedback
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=4, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)

        # 主体内容区域（用于切换列表视图和预览视图）
        self.content_frame = ttk.Frame(self.search_tab, padding="10")
        
        # 尝试获取主题背景色，用于 Text 控件
        try:
            self._bg_color = self.tk.call('ttk::style', 'lookup', 'TFrame', '-background')
            if not self._bg_color:
                self._bg_color = "#f0f0f0"
        except Exception:
            self._bg_color = "#f0f0f0"
            
        # ── 视图 1：缩略图列表视图 ──
        self.gallery_frame = ttk.Frame(self.content_frame)
        self.gallery_text = tk.Text(
            self.gallery_frame, 
            wrap="char",
            state="disabled", 
            cursor="arrow", 
            bg=self._bg_color,
            bd=0,
            highlightthickness=0,
        )
        self.gallery_scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.gallery_text.yview)
        self.gallery_text.configure(yscrollcommand=self.gallery_scrollbar.set)
        
        self.gallery_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ── 视图 2：详情预览视图 ──
        self.preview_frame = ttk.Frame(self.content_frame)
        
        # 预览提示文本或信息
        self.info_label = ttk.Label(self.preview_frame, textvariable=self.info_var, font=("Arial", 11, "bold"))
        self.info_label.pack(side=tk.TOP, pady=5)

        self.desc_label = ttk.Label(self.preview_frame, textvariable=self.desc_var, wraplength=800, justify=tk.LEFT)
        self.desc_label.pack(side=tk.TOP, pady=5)

        # 图片画板
        self.canvas = tk.Canvas(self.preview_frame, bg="gray", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 绑定画板右键菜单（支持大图预览时收藏操作）
        self.canvas.bind("<Button-3>", self._show_preview_context_menu)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 底部控制区
        self.bottom_frame = ttk.Frame(self.preview_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.back_list_btn = ttk.Button(self.bottom_frame, text="🔙 返回列表", command=self.show_gallery_view)
        self.back_list_btn.pack(side=tk.LEFT, padx=10)

        self.prev_btn = ttk.Button(self.bottom_frame, text="◀ 上一张", command=self.show_prev, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=10)

        self.page_label = ttk.Label(self.bottom_frame, textvariable=self.page_var, font=("Arial", 11))
        self.page_label.pack(side=tk.LEFT, padx=20)

        self.next_btn = ttk.Button(self.bottom_frame, text="下一张 ▶", command=self.show_next, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=10)

        self.open_dir_btn = ttk.Button(self.bottom_frame, text="📂 打开文件位置", command=self.open_file_location, state=tk.DISABLED)
        self.open_dir_btn.pack(side=tk.RIGHT, padx=10)

        self.top_frame.pack(side=tk.TOP, fill=tk.X)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 初始显示列表视图
        self.gallery_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.thumbnail_images = []
        self._thumbnail_gen = 0

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
        
        self.scan_workers_var = tk.StringVar(value="并发: -")
        ttk.Label(form_frame, textvariable=self.scan_workers_var, foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=10, pady=10)
        
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

        # 人脸扫描相关配置
        face_frame = ttk.Frame(form_frame)
        face_frame.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=10)
        ttk.Checkbutton(face_frame, text="同时进行人物扫描 (初次加载可能较慢)", variable=self.enable_face_scan_var).pack(side=tk.LEFT)
        
        self.face_device_status_var = tk.StringVar(value="人脸引擎状态: 检测中...")
        face_status_label = ttk.Label(face_frame, textvariable=self.face_device_status_var, foreground="green")
        face_status_label.pack(side=tk.LEFT, padx=10)
        
        # 异步检测人脸设备状态，避免阻塞 UI
        def _check_face_device():
            try:
                from photo_identify.face_manager import get_face_app, get_device_mode
                get_face_app() # 触发懒加载
                mode = get_device_mode()
                color = "green" if "CUDA" in mode else "orange"
                self.after(0, lambda: face_status_label.configure(foreground=color))
                self.after(0, lambda: self.face_device_status_var.set(f"人脸引擎状态: {mode}"))
            except Exception as e:
                self.after(0, lambda: face_status_label.configure(foreground="red"))
                self.after(0, lambda: self.face_device_status_var.set("人脸引擎状态: 初始化失败"))
        
        threading.Thread(target=_check_face_device, daemon=True).start()

        self.scan_btn = ttk.Button(form_frame, text="▶ 开始扫描 (见下方日志)", command=self.start_scan)
        self.scan_btn.grid(row=5, column=1, sticky=tk.EW, pady=20, ipady=5)

        scan_ctrl_frame = ttk.Frame(form_frame)
        scan_ctrl_frame.grid(row=5, column=2, sticky=tk.W, padx=10, pady=20)

        self.restart_btn = ttk.Button(scan_ctrl_frame, text="🔄 重启扫描", command=self.restart_scan, state=tk.DISABLED)
        self.restart_btn.pack(side=tk.LEFT, padx=(0, 5), ipady=5)

        self.stop_btn = ttk.Button(scan_ctrl_frame, text="⏹ 停止扫描", command=self.stop_scan, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, ipady=5)
        
        self._scan_cancel_event = None
        self._scan_thread_ref = None
        
        self.scan_status_var = tk.StringVar(value="准备就绪")
        ttk.Label(form_frame, textvariable=self.scan_status_var, foreground="blue").grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 终端输出重定向区域
        log_frame = ttk.LabelFrame(self.scan_tab, text="扫描日志", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tools inside log_frame
        log_tools_frame = ttk.Frame(log_frame)
        log_tools_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        ttk.Button(log_tools_frame, text="💾 导出日志保存", command=self.export_scan_logs).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

    # ── Tab 3: 人物管理 ──────────────────────────────────────────

    def _init_person_tab(self):
        """初始化人物管理标签页。"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.person_tab, padding=(10, 8, 10, 4))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="🔄 刷新列表", command=self._refresh_persons).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🗑 人物回收站", command=self._show_trash_bin).pack(side=tk.LEFT, padx=4)
        
        self.person_status_var = tk.StringVar(value="准备就绪")
        ttk.Label(toolbar, textvariable=self.person_status_var, foreground="blue").pack(side=tk.LEFT, padx=20)
        
        # 左右分割面板
        paned = ttk.PanedWindow(self.person_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧：人物列表视图 (使用 Canvas 实现可滚动的 Frame 列表)
        left_container = ttk.Frame(paned)
        paned.add(left_container, weight=0) # weight=0 意味着保持请求的宽度不被拉伸
        
        self.person_canvas = tk.Canvas(left_container, bg=getattr(self, "_bg_color", "#f0f0f0"), highlightthickness=0, width=190)
        vsb_left = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=self.person_canvas.yview)
        self.person_canvas.configure(yscrollcommand=vsb_left.set)
        
        self.person_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_left.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.person_list_inner_frame = tk.Frame(self.person_canvas, bg=getattr(self, "_bg_color", "#f0f0f0"))
        self._person_canvas_window = self.person_canvas.create_window((0, 0), window=self.person_list_inner_frame, anchor="nw")
        
        def _on_canvas_configure(event):
            self.person_canvas.itemconfig(self._person_canvas_window, width=event.width)
            
        def _on_frame_configure(event):
            self.person_canvas.configure(scrollregion=self.person_canvas.bbox("all"))
            
        self.person_canvas.bind('<Configure>', _on_canvas_configure)
        self.person_list_inner_frame.bind('<Configure>', _on_frame_configure)
        
        # 绑定鼠标滚轮
        def _on_mousewheel(event):
            if str(self.person_canvas.cget("state")) != tk.DISABLED:
                self.person_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                
        # Windows / macOS binding
        self.person_canvas.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e) if self._is_in_widget(e, self.person_canvas) else None)
        
        # 绑定事件相关的状态
        self.person_thumbnail_images = [] # 保存列表里的头像防止被回收
        self.gallery_thumbnail_images = [] # 保存右侧画廊的头像
        self._person_list_gen = 0
        self._gallery_gen = 0
        self.person_list = []
        self._selected_person_id = None
        self._person_frames = {} # {person_id: tk.Frame}
        
        # 右侧：选中人物包含的图片画廊
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        right_toolbar = ttk.Frame(right_frame, padding=(0, 0, 0, 5))
        right_toolbar.pack(fill=tk.X)
        self.person_image_count_var = tk.StringVar(value="")
        ttk.Label(right_toolbar, textvariable=self.person_image_count_var).pack(side=tk.LEFT)
        
        self.person_gallery_text = tk.Text(
            right_frame, 
            wrap="char",
            state="disabled", 
            cursor="arrow", 
            bg=getattr(self, "_bg_color", "#f0f0f0"),
            bd=0,
            highlightthickness=0,
        )
        vsb_right = ttk.Scrollbar(right_frame, orient="vertical", command=self.person_gallery_text.yview)
        self.person_gallery_text.configure(yscrollcommand=vsb_right.set)
        
        self.person_gallery_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb_right.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.person_thumbnail_images = []
        self._person_thumbnail_gen = 0
        self.person_list = []
        
    def _is_in_widget(self, event, widget):
        if not widget.winfo_exists():
            return False
        x, y = widget.winfo_pointerxy()
        widget_x, widget_y = widget.winfo_rootx(), widget.winfo_rooty()
        widget_w, widget_h = widget.winfo_width(), widget.winfo_height()
        return (widget_x <= x <= widget_x + widget_w) and (widget_y <= y <= widget_y + widget_h)

    def _refresh_persons(self, keep_selected_id=None, refresh_gallery=False):
        for widget in self.person_list_inner_frame.winfo_children():
            widget.destroy()
        self.person_thumbnail_images.clear()
        self._person_frames.clear()
        self._selected_person_id = None
        self._person_frame_order = None  # 重置显示顺序跟踪
        
        # 只有当没有要求保留选择，或者右侧还没有被渲染时，才清空右侧画廊
        if keep_selected_id is None:
            self.person_gallery_text.configure(state="normal")
            self.person_gallery_text.delete(1.0, tk.END)
            self.person_gallery_text.configure(state="disabled")
            self.person_image_count_var.set("")
        
        try:
            from photo_identify.storage import Storage
            storage = Storage(self.scan_db_var.get())
            self.person_list = storage.get_all_persons(include_deleted=False)
            storage.close()
            
            self._person_list_gen += 1
            current_gen = self._person_list_gen
            
            def _thread_load():
                for idx, p in enumerate(self.person_list):
                    if current_gen != self._person_list_gen:
                        break
                        
                    img = None
                    try:
                        file_path = p.get("path", "")
                        actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
                        if actual_path and os.path.isfile(actual_path):
                            frame_bytes = get_image_frame_bytes(actual_path)
                            pil_img = Image.open(io.BytesIO(frame_bytes))
                            img = crop_and_circle_face(pil_img, p.get("bbox", "[]"), size=40)
                    except Exception as e:
                        print(f"Error loading face for {p.get('name')}: {e}")
                        
                    self.after(0, self._add_person_list_item, idx, p, img, current_gen)
                    
                # 加载完毕后，恢复选中状态或默认选中第一个人物
                if keep_selected_id is not None:
                    self.after(100, lambda: self._select_person(keep_selected_id, refresh_gallery=refresh_gallery))
                elif self.person_list:
                    first_id = self.person_list[0]["id"]
                    self.after(100, lambda pid=first_id: self._select_person(pid))
                    
            threading.Thread(target=_thread_load, daemon=True).start()
            
            self.person_status_var.set(f"共显示 {len(self.person_list)} 个人物。")
        except Exception as e:
            self.person_status_var.set(f"加载人物失败: {e}")

    def _add_person_list_item(self, idx, p_data, img, gen):
        if gen != self._person_list_gen:
            return
            
        bg_color = getattr(self, "_bg_color", "#f0f0f0")
        item_frame = tk.Frame(self.person_list_inner_frame, bg=bg_color, cursor="hand2", pady=2, padx=5)
        item_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
        
        person_id = p_data["id"]
        self._person_frames[person_id] = item_frame
        
        if img:
            photo = ImageTk.PhotoImage(img)
            self.person_thumbnail_images.append(photo)
            img_lbl = tk.Label(item_frame, image=photo, bg=bg_color)
        else:
            img_lbl = tk.Label(item_frame, text="无头像", bg=bg_color, width=5, height=2)
            
        img_lbl.pack(side=tk.LEFT, padx=(0, 10))
        item_frame.img_lbl = img_lbl
        
        mid_frame = tk.Frame(item_frame, bg=bg_color)
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        name_var = tk.StringVar(value=p_data["name"])
        name_lbl = tk.Label(mid_frame, textvariable=name_var, bg=bg_color, anchor="sw", font=("Arial", 10))
        name_lbl.pack(side=tk.TOP, fill=tk.X, expand=True)
        
        count_lbl = tk.Label(mid_frame, text=f"{p_data.get('face_count', 0)} 张照片", bg=bg_color, anchor="nw", font=("Arial", 8), fg="gray")
        count_lbl.pack(side=tk.TOP, fill=tk.X, expand=True)
        
        # 保存 name_var 到 frame 上方便后续双击修改
        item_frame.name_var = name_var
        item_frame.person_data = p_data
        
        # 控制按钮
        btn_frame = tk.Frame(item_frame, bg=bg_color)
        btn_frame.pack(side=tk.RIGHT)
        
        def move_up(e, pid=person_id):
            e.stopPropagation = True
            self._move_person(pid, -1)
            
        def move_down(e, pid=person_id):
            e.stopPropagation = True
            self._move_person(pid, 1)
            
        btn_up = tk.Label(btn_frame, text="▲", bg=bg_color, cursor="hand2")
        btn_up.pack(side=tk.LEFT, padx=2)
        btn_up.bind("<Button-1>", move_up)
        
        btn_down = tk.Label(btn_frame, text="▼", bg=bg_color, cursor="hand2")
        btn_down.pack(side=tk.LEFT, padx=2)
        btn_down.bind("<Button-1>", move_down)
        
        # 右键菜单
        menu = tk.Menu(item_frame, tearoff=0)
        menu.add_command(label="重命名", command=lambda pid=person_id: self._rename_person(pid))
        menu.add_command(label="移入回收站", command=lambda pid=person_id: self._delete_person(pid))
        if p_data.get("path"):
            def _open_loc_person():
                self._open_file_location_by_path(p_data.get("path", ""))
            menu.add_command(label="打开文件位置", command=_open_loc_person)
            
        def show_menu(e):
            menu.tk_popup(e.x_root, e.y_root)
            
        item_frame.bind("<Button-3>", show_menu)
        img_lbl.bind("<Button-3>", show_menu)
        name_lbl.bind("<Button-3>", show_menu)
        
        def on_click(e, pid=person_id):
            if hasattr(e, 'stopPropagation') and e.stopPropagation:
                return
            self._select_person(pid)
            
        item_frame.bind("<Button-1>", on_click)
        img_lbl.bind("<Button-1>", on_click)
        name_lbl.bind("<Button-1>", on_click)
        
        # 双击重命名
        def on_double_click(e, pid=person_id):
            self._rename_person(pid)
            
        item_frame.bind("<Double-1>", on_double_click)
        name_lbl.bind("<Double-1>", on_double_click)

    def _select_person(self, person_id, refresh_gallery=True):
        # 恢复旧的选中状态
        if self._selected_person_id in self._person_frames:
            old_frame = self._person_frames[self._selected_person_id]
            old_frame.configure(bg=getattr(self, "_bg_color", "#f0f0f0"))
            for child in old_frame.winfo_children():
                child.configure(bg=getattr(self, "_bg_color", "#f0f0f0"))
                if hasattr(child, "winfo_children"):
                    for subchild in child.winfo_children():
                        subchild.configure(bg=getattr(self, "_bg_color", "#f0f0f0"))
                
        self._selected_person_id = person_id
        if person_id in self._person_frames:
            new_frame = self._person_frames[person_id]
            sel_color = "#cce8ff"
            new_frame.configure(bg=sel_color)
            for child in new_frame.winfo_children():
                child.configure(bg=sel_color)
                if hasattr(child, "winfo_children"):
                    for subchild in child.winfo_children():
                        subchild.configure(bg=sel_color)
                
            p_data = new_frame.person_data
            if refresh_gallery:
                try:
                    from photo_identify.storage import Storage
                    storage = Storage(self.scan_db_var.get())
                    images = storage.get_images_by_person(p_data["cluster_id"])
                    for img in images:
                        img["db_path"] = self.scan_db_var.get()
                    storage.close()
                    
                    self.person_image_count_var.set(f"包含此人的照片总数: {len(images)}")
                    self._load_person_thumbnails(images, p_data)
                    
                except Exception as e:
                    self.person_status_var.set(f"加载照片失败: {e}")

    def _show_trash_bin(self):
        dlg = tk.Toplevel(self)
        dlg.title("人物回收站")
        dlg.geometry("400x500")
        dlg.grab_set()
        
        try:
            from photo_identify.storage import Storage
            storage = Storage(self.scan_db_var.get())
            deleted_list = storage.get_deleted_persons()
            storage.close()
        except Exception as e:
            messagebox.showerror("错误", f"加载回收站失败: {e}", parent=dlg)
            return

        lbl = ttk.Label(dlg, text="在列表右键点击或双击可以恢复人物:", padding=10)
        lbl.pack(side=tk.TOP, fill=tk.X)
        
        list_container = ttk.Frame(dlg)
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        canvas = tk.Canvas(list_container, bg=getattr(self, "_bg_color", "#f0f0f0"), highlightthickness=0)
        vsb = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        inner_frame = tk.Frame(canvas, bg=getattr(self, "_bg_color", "#f0f0f0"))
        canvas_window = canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
            
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        canvas.bind('<Configure>', _on_canvas_configure)
        inner_frame.bind('<Configure>', _on_frame_configure)
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", lambda e: _on_mousewheel(e) if self._is_in_widget(e, canvas) else None)
        
        # 保持引用
        dlg.thumbnails = []
        
        def _restore_person(person_id, name, item_frame):
            if messagebox.askyesno("恢复", f"确定要恢复人物 {name} 吗？", parent=dlg):
                try:
                    from photo_identify.storage import Storage
                    storage = Storage(self.scan_db_var.get())
                    storage.set_person_deleted(person_id, 0)
                    storage.close()
                    item_frame.destroy()
                    self._refresh_persons()
                except Exception as e:
                    messagebox.showerror("错误", f"恢复失败: {e}", parent=dlg)
                    
        def _load_trash_items():
            for p in deleted_list:
                img = None
                try:
                    file_path = p.get("path", "")
                    actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
                    if actual_path and os.path.isfile(actual_path):
                        frame_bytes = get_image_frame_bytes(actual_path)
                        pil_img = Image.open(io.BytesIO(frame_bytes))
                        img = crop_and_circle_face(pil_img, p.get("bbox", "[]"), size=40)
                except Exception:
                    pass
                    
                self.after(0, _add_trash_item, p, img)
                
        def _add_trash_item(p_data, img):
            bg_color = getattr(self, "_bg_color", "#f0f0f0")
            item_frame = tk.Frame(inner_frame, bg=bg_color, cursor="hand2", pady=2, padx=5)
            item_frame.pack(side=tk.TOP, fill=tk.X, expand=True)
            
            if img:
                photo = ImageTk.PhotoImage(img)
                dlg.thumbnails.append(photo)
                img_lbl = tk.Label(item_frame, image=photo, bg=bg_color)
            else:
                img_lbl = tk.Label(item_frame, text="无头像", bg=bg_color, width=5, height=2)
                
            img_lbl.pack(side=tk.LEFT, padx=(0, 10))
            
            mid_frame = tk.Frame(item_frame, bg=bg_color)
            mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            name_lbl = tk.Label(mid_frame, text=p_data["name"], bg=bg_color, anchor="sw", font=("Arial", 10))
            name_lbl.pack(side=tk.TOP, fill=tk.X, expand=True)
            
            count_lbl = tk.Label(mid_frame, text=f"{p_data.get('face_count', 0)} 张照片", bg=bg_color, anchor="nw", font=("Arial", 8), fg="gray")
            count_lbl.pack(side=tk.TOP, fill=tk.X, expand=True)
            
            # 绑定事件
            person_id = p_data["id"]
            name = p_data["name"]
            
            menu = tk.Menu(item_frame, tearoff=0)
            menu.add_command(label="恢复显示", command=lambda pid=person_id, n=name, f=item_frame: _restore_person(pid, n, f))
            
            def show_menu(e):
                menu.tk_popup(e.x_root, e.y_root)
                
            item_frame.bind("<Button-3>", show_menu)
            img_lbl.bind("<Button-3>", show_menu)
            name_lbl.bind("<Button-3>", show_menu)
            
            def on_double_click(e, pid=person_id, n=name, f=item_frame):
                _restore_person(pid, n, f)
                
            item_frame.bind("<Double-1>", on_double_click)
            name_lbl.bind("<Double-1>", on_double_click)
            
        threading.Thread(target=_load_trash_items, daemon=True).start()
        
        btn_frame = ttk.Frame(dlg, padding=10)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(btn_frame, text="关闭", command=dlg.destroy).pack(side=tk.RIGHT)
        
        dlg.update_idletasks()
        pw = self.winfo_rootx() + self.winfo_width() // 2
        ph = self.winfo_rooty() + self.winfo_height() // 2
        w, h = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f"+{pw - w // 2}+{ph - h // 2}")
    def _rename_person(self, person_id):
        if person_id not in self._person_frames:
            return
        frame = self._person_frames[person_id]
        name_var = getattr(frame, "name_var", None)
        if not name_var:
            return
        old_name = name_var.get()
        import tkinter.simpledialog as simpledialog
        new_name = simpledialog.askstring("重命名人物", "请输入新的姓名:", initialvalue=old_name, parent=self)
        if new_name and new_name.strip() and new_name != old_name:
            try:
                from photo_identify.storage import Storage
                storage = Storage(self.scan_db_var.get())
                storage.update_person_name(person_id, new_name.strip())
                storage.close()
                name_var.set(new_name.strip())
                p_data = getattr(frame, "person_data", {})
                if p_data:
                    p_data["name"] = new_name.strip()
            except Exception as e:
                messagebox.showerror("重命名失败", f"数据库更新错误:\n{e}")

    def _delete_person(self, person_id):
        if person_id not in self._person_frames:
            return
        frame = self._person_frames[person_id]
        name_var = getattr(frame, "name_var", None)
        name = name_var.get() if name_var else "未知"
        if messagebox.askyesno("确认删除", f"确定要将 {name} 移入回收站吗？\n下次扫描该人物也不会恢复显示。"):
            try:
                from photo_identify.storage import Storage
                storage = Storage(self.scan_db_var.get())
                storage.set_person_deleted(person_id, 1)
                storage.close()
                self._refresh_persons() # 刷新列表
            except Exception as e:
                messagebox.showerror("删除失败", f"数据库更新错误:\n{e}")

    def _move_person(self, person_id, direction):
        # 使用跟踪的显示顺序，而非 winfo_children()（后者始终返回创建顺序，排序后会失效）
        if self._person_frame_order is None:
            self._person_frame_order = list(self.person_list_inner_frame.winfo_children())

        frames_list = list(self._person_frame_order)
        idx = -1
        for i, f in enumerate(frames_list):
            p_data = getattr(f, "person_data", None)
            if p_data and p_data["id"] == person_id:
                idx = i
                break

        if idx == -1:
            return

        new_idx = idx + direction
        if 0 <= new_idx < len(frames_list):
            # 交换位置
            frames_list.insert(new_idx, frames_list.pop(idx))
            # 重新 pack
            for f in frames_list:
                f.pack_forget()
            for f in frames_list:
                f.pack(side=tk.TOP, fill=tk.X, expand=True)

            # 更新跟踪的显示顺序
            self._person_frame_order = frames_list
                
            # 保存到数据库
            try:
                from photo_identify.storage import Storage
                storage = Storage(self.scan_db_var.get())
                person_ids_in_order = [getattr(f, "person_data")["id"] for f in frames_list if hasattr(f, "person_data")]
                storage.update_person_sort_order(person_ids_in_order)
                storage.close()
            except Exception as e:
                print(f"Error saving sort order: {e}")


    def _load_person_thumbnails(self, images, person_data):
        self.person_gallery_text.configure(state="normal")
        self.person_gallery_text.delete(1.0, tk.END)
        self.person_gallery_text.configure(state="disabled")
        self.gallery_thumbnail_images.clear()
        
        self._gallery_gen += 1
        current_gen = self._gallery_gen
        
        # 先在第一张强行插入该人物当前的圆角头像，然后再启动其它图片的加载线程
        def _add_cover():
            img = None
            try:
                if person_data.get("path") and os.path.isfile(person_data["path"]):
                    frame_bytes = get_image_frame_bytes(person_data["path"])
                    pil_img = Image.open(io.BytesIO(frame_bytes))
                    img = crop_and_circle_face(pil_img, person_data.get("bbox", "[]"), size=150)
            except Exception:
                pass
            self.after(0, lambda: self._add_gallery_item(None, "【当前头像】", img, current_gen, is_cover=True, p_data=person_data))
            self.after(0, lambda: threading.Thread(target=_thread, daemon=True).start())
            
        def _thread():
            for i, record in enumerate(images):
                if current_gen != self._gallery_gen:
                    break
                
                file_path = record.get("path", "")
                actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
                
                img = None
                try:
                    if os.path.isfile(actual_path):
                        frame_bytes = get_image_frame_bytes(actual_path)
                        pil_img = Image.open(io.BytesIO(frame_bytes))
                        pil_img = ImageOps.exif_transpose(pil_img)
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                            
                        # 如果需要显示普通的照片缩略图
                        pil_img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        img = pil_img
                except Exception:
                    pass
                
                self.after(0, lambda r=record, image=img: self._add_gallery_item(r, None, image, current_gen, is_cover=False, p_data=person_data))

        threading.Thread(target=_add_cover, daemon=True).start()

    def _add_gallery_item(self, record, custom_name, img, gen, is_cover, p_data):
        if gen != self._gallery_gen:
            return
            
        bg_color = getattr(self, "_bg_color", "#f0f0f0")
        item_frame = tk.Frame(self.person_gallery_text, bg=bg_color, cursor="hand2")
        
        if img:
            photo = ImageTk.PhotoImage(img)
            self.gallery_thumbnail_images.append(photo)
        else:
            photo = None
            
        kwargs = {}
        if photo:
            kwargs['image'] = photo
        else:
            kwargs['text'] = "无图片"
            kwargs['width'] = 20
            kwargs['height'] = 10
        
        img_lbl = tk.Label(item_frame, bg=bg_color, **kwargs)
        img_lbl.pack(side="top")
        
        if custom_name:
            name = custom_name
        else:
            name = record.get("file_name", os.path.basename(record.get("path", "")))
            if len(name) > 12:
                name = name[:10] + "..."
                
        txt_lbl = tk.Label(item_frame, text=name, bg=bg_color, width=18)
        txt_lbl.pack(side="top")
        
        if is_cover:
            txt_lbl.configure(fg="blue", font=("Arial", 10, "bold"))
            if p_data.get("path"):
                menu = tk.Menu(item_frame, tearoff=0)
                def _open_loc_cover():
                    self._open_file_location_by_path(p_data.get("path", ""))
                menu.add_command(label="打开文件位置", command=_open_loc_cover)
                def show_menu_cover(e):
                    menu.tk_popup(e.x_root, e.y_root)
                item_frame.bind("<Button-3>", show_menu_cover)
                img_lbl.bind("<Button-3>", show_menu_cover)
                txt_lbl.bind("<Button-3>", show_menu_cover)
            
        if not is_cover and record:
            # 点击打开大图预览
            def on_click(event, r=record):
                self.current_results = [r]
                self.current_index = 0
                self.notebook.select(self.search_tab)
                self._update_display()
                self.show_preview_view()
                
            item_frame.bind("<Button-1>", lambda e, r=record: on_click(e, r))
            img_lbl.bind("<Button-1>", lambda e, r=record: on_click(e, r))
            txt_lbl.bind("<Button-1>", lambda e, r=record: on_click(e, r))
            
            # 右键菜单
            menu = tk.Menu(item_frame, tearoff=0)
            def _copy_filename(r=record):
                file_name = r.get("file_name", os.path.basename(r.get("path", "")))
                self.clipboard_clear()
                self.clipboard_append(file_name)
            menu.add_command(label="复制文件名", command=lambda r=record: _copy_filename(r))
            def _set_cover(r=record):
                person_id = p_data["id"]
                image_id = r["id"]
                # 由于这整个列表是基于 images 返回的，但是 cover 必须是 face_id
                # 为了简便起见，在这个 record 里我们需要得到 face_id。但是 storage 里的查询只有 images 数据。
                # 修改 storage 的 get_images_by_person 返回包含 face_id。
                # 为了不改动过大，我们单独查一下该图像对应的当前人的 face_id
                self._set_person_cover_action(person_id, p_data["cluster_id"], image_id)
                
            menu.add_command(label="设为头像", command=lambda r=record: _set_cover(r))
            
            def _open_loc(r=record):
                self._open_file_location_by_path(r.get("path", ""))
            menu.add_command(label="打开文件位置", command=lambda r=record: _open_loc(r))
            
            # --- 增加画廊照片右上角的收藏星角标 ---
            is_fav = record.get("is_favorite", 0)
            star_label = tk.Label(img_lbl, text="★", fg="#e3b341", bg="#333333", font=("Segoe UI", 12))
            
            # 只有当原本就被收藏时，才把它 place 出来
            def _update_star_badge(state):
                if state:
                    star_label.place(relx=1.0, x=-4, y=4, anchor="ne")
                else:
                    star_label.place_forget()
                    
            _update_star_badge(is_fav)
            # -----------------------------------------------
            
            def _toggle_fav(r=record):
                from photo_identify.storage import Storage # Import here to avoid circular dependency if Storage imports GUI
                # 取得来源于记录本身的 db_path，否则退回默认的主库
                target_db = r.get("db_path", self.db_path)
                storage = Storage(target_db)
                new_fav = not bool(r.get("is_favorite", 0))
                storage.toggle_favorite(r["id"], new_fav)
                storage.close()
                r["is_favorite"] = 1 if new_fav else 0
                
                # 动态更新缩略图上的星标可见性
                _update_star_badge(new_fav)
                
                # In _add_gallery_item, we don't refresh the favorites tab immediately,
                # as this item might be in the search results, not the favorites tab.
                # The menu text will be updated dynamically by show_menu.
            menu.add_command(label="取消收藏" if record.get("is_favorite") else "加入收藏", command=lambda r=record: _toggle_fav(r))
            
            def show_menu(e, r=record):
                is_fav = r.get("is_favorite", 0)
                # The index for "加入收藏" / "取消收藏" is 3 (0:复制文件名, 1:设为头像, 2:打开文件位置, 3:收藏)
                menu.entryconfigure(3, label="取消收藏" if is_fav else "加入收藏")
                menu.tk_popup(e.x_root, e.y_root)
                
            item_frame.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
            img_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
            txt_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        
        self.person_gallery_text.configure(state="normal")
        self.person_gallery_text.window_create("end", window=item_frame, padx=10, pady=10)
        self.person_gallery_text.configure(state="disabled")

    def _set_person_cover_action(self, person_id, cluster_id, image_id):
        try:
            from photo_identify.storage import Storage
            storage = Storage(self.scan_db_var.get())
            # 找到这张图属于这个人的 face_id 和 bbox
            cursor = storage._conn.cursor()
            cursor.execute("SELECT id, bbox FROM face_embeddings WHERE image_id = ? AND cluster_id = ? LIMIT 1", (image_id, cluster_id))
            row = cursor.fetchone()
            if not row:
                messagebox.showerror("错误", "该照片不包含此人脸，无法设为头像。")
                storage.close()
                return
            face_id = row[0]
            face_bbox = row[1]
            storage.update_person_cover(person_id, image_id, face_id)
            
            # 获取新的 path 用于局部更新
            cursor.execute("SELECT path FROM images WHERE id = ?", (image_id,))
            img_row = cursor.fetchone()
            new_path = img_row[0] if img_row else ""
            storage.close()
            
            # 局部更新逻辑
            if person_id in self._person_frames:
                frame = self._person_frames[person_id]
                p_data = getattr(frame, "person_data", {})
                if p_data:
                    p_data["cover_image_id"] = image_id
                    p_data["cover_face_id"] = face_id
                    p_data["path"] = new_path
                    p_data["bbox"] = face_bbox
                    
                    # 重新生成左侧小头像
                    try:
                        actual_path = new_path.split("#t=")[0] if "#t=" in new_path else new_path
                        if os.path.isfile(actual_path):
                            frame_bytes = get_image_frame_bytes(actual_path)
                            pil_img = Image.open(io.BytesIO(frame_bytes))
                            img = crop_and_circle_face(pil_img, face_bbox, size=40)
                            photo = ImageTk.PhotoImage(img)
                            self.person_thumbnail_images.append(photo)
                            img_lbl = getattr(frame, "img_lbl", None)
                            if img_lbl:
                                img_lbl.configure(image=photo, text="")
                    except Exception as e:
                        pass
                        
                    # 重新选中，触发右侧画廊的重新渲染
                    self._select_person(person_id, refresh_gallery=True)
                
        except Exception as e:
            messagebox.showerror("错误", f"更新头像失败:\n{e}")

    # =========================================================================
    # 照片收藏 Tab
    # =========================================================================
    def _init_favorite_tab(self):
        # 顶部控制栏
        top_frame = tk.Frame(self.favorite_tab, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.favorite_count_var = tk.StringVar(value="共 0 张")
        tk.Label(top_frame, textvariable=self.favorite_count_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="导出CSV", command=self._export_favorites_csv).pack(side=tk.RIGHT, padx=10)
        tk.Button(top_frame, text="刷新", command=self._refresh_favorites).pack(side=tk.RIGHT, padx=10)
        
        # 缩略图滚动区域 (使用 Text + Scrollbar 方案)
        gallery_container = ttk.Frame(self.favorite_tab)
        gallery_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self._favorite_gen = 0
        self.favorite_thumbnail_images = []
        
        bg_color = getattr(self, "_bg_color", "#f0f0f0")
        self.favorite_gallery_text = tk.Text(gallery_container, wrap="char", state="disabled", bg=bg_color, bd=0, highlightthickness=0)
        self.favorite_scrollbar = ttk.Scrollbar(gallery_container, orient="vertical", command=self.favorite_gallery_text.yview)
        self.favorite_gallery_text.configure(yscrollcommand=self.favorite_scrollbar.set)
        
        self.favorite_gallery_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.favorite_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮绑定与 _init_search_tab 中类似
        self.favorite_gallery_text.bind_all("<MouseWheel>", self._on_favorite_mousewheel)

    def _on_favorite_mousewheel(self, event):
        if self._is_in_widget(event, self.favorite_gallery_text):
            self.favorite_gallery_text.yview_scroll(int(-1*(event.delta/120)), "units")

    def _export_favorites_csv(self):
        import csv
        from photo_identify.storage import Storage
        
        favs = []
        try:
            storage = Storage(self.db_path)
            main_favs = storage.get_favorites()
            for f in main_favs:
                f["db_path"] = self.db_path
            favs.extend(main_favs)
            storage.close()
        except Exception:
            pass
            
        # 若配置了独立的扫描库，同时也加载扫描库里的收藏图片
        scan_db = self.scan_db_var.get()
        if scan_db and scan_db != self.db_path and os.path.exists(scan_db):
            try:
                storage_scan = Storage(scan_db)
                scan_favs = storage_scan.get_favorites()
                for f in scan_favs:
                    f["db_path"] = scan_db
                favs.extend(scan_favs)
                storage_scan.close()
            except Exception:
                pass
                
        if not favs:
            messagebox.showinfo("提示", "当前没有收藏的照片")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="导出收藏记录",
            initialfile="favorites_export.csv"
        )
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["图片名称", "绝对地址", "所属数据库"])
                for r in favs:
                    fname = r.get("file_name", os.path.basename(r.get("path", "")))
                    writer.writerow([fname, r.get("path", ""), r.get("db_path", self.db_path)])
            messagebox.showinfo("成功", f"成功导出 {len(favs)} 条记录 (含扫描库)")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

    def _refresh_favorites(self):
        if not self.db_path or not os.path.exists(self.db_path):
            return
            
        from photo_identify.storage import Storage
        favs = []
        try:
            storage = Storage(self.db_path)
            main_favs = storage.get_favorites()
            for f in main_favs:
                f["db_path"] = self.db_path
            favs.extend(main_favs)
            storage.close()
        except:
            pass

        # 若配置了独立的扫描库，同时也加载扫描库里的收藏图片
        scan_db = self.scan_db_var.get()
        if scan_db and scan_db != self.db_path and os.path.exists(scan_db):
            try:
                storage_scan = Storage(scan_db)
                scan_favs = storage_scan.get_favorites()
                for f in scan_favs:
                    f["db_path"] = scan_db
                favs.extend(scan_favs)
                storage_scan.close()
            except:
                pass

        # 重新按照 modified_time 降序排序所有合并的记录
        favs.sort(key=lambda x: x.get("modified_time", ""), reverse=True)
        
        self.favorite_count_var.set(f"共 {len(favs)} 张")
        
        self.favorite_gallery_text.configure(state="normal")
        self.favorite_gallery_text.delete(1.0, tk.END)
        self.favorite_gallery_text.configure(state="disabled")
        self.favorite_thumbnail_images.clear()
        
        self._favorite_gen += 1
        current_gen = self._favorite_gen
        
        def _thread():
            for i, record in enumerate(favs):
                if current_gen != self._favorite_gen:
                    break
                    
                file_path = record.get("path", "")
                actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
                img = None
                try:
                    if os.path.isfile(actual_path):
                        frame_bytes = get_image_frame_bytes(actual_path)
                        pil_img = Image.open(io.BytesIO(frame_bytes))
                        pil_img = ImageOps.exif_transpose(pil_img)
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        pil_img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                        img = pil_img
                except:
                    pass
                
                self.after(0, lambda idx=i, r=record, image=img: self._add_favorite_thumbnail(idx, r, image, current_gen))
                
        threading.Thread(target=_thread, daemon=True).start()

    def _add_favorite_thumbnail(self, index, record, img, gen):
        if gen != self._favorite_gen:
            return
            
        bg_color = getattr(self, "_bg_color", "#f0f0f0")
        item_frame = tk.Frame(self.favorite_gallery_text, bg=bg_color, cursor="hand2")
        
        if img:
            photo = ImageTk.PhotoImage(img)
            self.favorite_thumbnail_images.append(photo)
        else:
            photo = None
            
        kwargs = {}
        if photo:
            kwargs['image'] = photo
        else:
            kwargs['text'] = "无图片"
            kwargs['width'] = 20
            kwargs['height'] = 10
        
        img_lbl = tk.Label(item_frame, bg=bg_color, **kwargs)
        img_lbl.pack(side="top")
        
        name = record.get("file_name", os.path.basename(record.get("path", "")))
        if len(name) > 12:
            name = name[:10] + "..."
        txt_lbl = tk.Label(item_frame, text=name, bg=bg_color, width=18)
        txt_lbl.pack(side="top")
        
        def on_click(event, r=record):
            # 将当前列表作为预览内容传入
            self.current_results = [r]
            self.current_index = 0
            # 临时借用检索的预览组件，但可能需要返回，因此更好的做法是先跳去检索页
            self.notebook.select(self.search_tab)
            self._update_display()
            self.show_preview_view()
            
        item_frame.bind("<Button-1>", lambda e, r=record: on_click(e, r))
        img_lbl.bind("<Button-1>", lambda e, r=record: on_click(e, r))
        txt_lbl.bind("<Button-1>", lambda e, r=record: on_click(e, r))
        
        # 右键菜单
        menu = tk.Menu(item_frame, tearoff=0)
        def _copy_filename(r=record):
            file_name = r.get("file_name", os.path.basename(r.get("path", "")))
            self.clipboard_clear()
            self.clipboard_append(file_name)
        menu.add_command(label="复制文件名", command=lambda r=record: _copy_filename(r))
        
        def _open_loc(r=record):
            self._open_file_location_by_path(r.get("path", ""))
        menu.add_command(label="打开文件位置", command=lambda r=record: _open_loc(r))
        
        def _toggle_fav(r=record):
            from photo_identify.storage import Storage
            target_db = r.get("db_path", self.db_path)
            storage = Storage(target_db)
            new_fav = not bool(r.get("is_favorite", 0))
            storage.toggle_favorite(r["id"], new_fav)
            storage.close()
            # 由于在收藏页，点击取消收藏后需要刷新
            self._refresh_favorites()
        menu.add_command(label="取消收藏" if record.get("is_favorite") else "加入收藏", command=lambda r=record: _toggle_fav(r))
        
        def show_menu(e, r=record):
            menu.tk_popup(e.x_root, e.y_root)
            
        item_frame.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        img_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        txt_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        
        self.favorite_gallery_text.configure(state="normal")
        self.favorite_gallery_text.window_create("end", window=item_frame, padx=10, pady=10)
        self.favorite_gallery_text.configure(state="disabled")

    # ── Tab 4: 模型管理 ──────────────────────────────────────────

    def _init_model_tab(self):
        """初始化模型管理标签页。"""
        # 顶部工具栏
        toolbar = ttk.Frame(self.model_tab, padding=(10, 8, 10, 4))
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="➕ 新增", command=self._model_add).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="✏️ 编辑", command=self._model_edit).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🗑 删除", command=self._model_delete).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="🔄 刷新状态", command=self._model_refresh).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="💾 导出CSV", command=self._model_export_csv).pack(side=tk.LEFT, padx=4)

        # 将环境变量设置按钮放右侧
        ttk.Button(toolbar, text="⚙ 环境变量设置", command=self.open_env_vars).pack(side=tk.RIGHT, padx=4)

        # 状态提示
        self._model_status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self._model_status_var, foreground="gray").pack(side=tk.RIGHT, padx=10)

        # 模型列表表格
        tree_frame = ttk.Frame(self.model_tab, padding=(10, 0, 10, 10))
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("type", "name", "model_id", "base_url", "api_key_var", "workers", "status")
        self._model_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")

        col_settings = [
            ("type",        "模型类型",     80,  tk.CENTER),
            ("name",        "模型名称",     160, tk.W),
            ("model_id",    "模型ID",       160, tk.W),
            ("base_url",    "接口地址",     200, tk.W),
            ("api_key_var", "API变量名",    150, tk.W),
            ("workers",     "并发数",       60,  tk.CENTER),
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
        # 右键菜单
        self._model_tree.bind("<Button-3>", self._show_model_context_menu)

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
            if m.get("is_local"):
                status_text = "🏠 本地"
                tag = "ok"
            elif m["api_key_status"]:
                status_text = "✅"
                tag = "ok"
            else:
                status_text = "❌"
                tag = "fail"
            iid = self._model_tree.insert(
                "", tk.END,
                values=(m["type"], m["name"], m["model_id"], m["base_url"], m["api_key_var"] or "(无需)", m.get("workers", 4), status_text),
                tags=(tag,)
            )
            self._tree_item_to_db_id[iid] = m["id"]

        local_count = sum(1 for m in models if m.get("is_local"))
        remote_ok = sum(1 for m in models if not m.get("is_local") and m["api_key_status"])
        self._model_status_var.set(f"共 {len(models)} 个模型 · {local_count} 个本地 · {remote_ok} 个远程APIkey已配置")

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
            self._model_mgr.add_model(r["type"], r["name"], r["model_id"], r["base_url"], r["api_key_var"], r.get("workers", 4))
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
            self._model_mgr.update_model(db_id, r["type"], r["name"], r["model_id"], r["base_url"], r["api_key_var"], r.get("workers", 4))
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

    def _show_model_context_menu(self, event):
        """模型列表右键菜单。"""
        iid = self._model_tree.identify_row(event.y)
        if not iid:
            return
        self._model_tree.selection_set(iid)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="✏️ 编辑", command=self._model_edit)
        menu.add_command(label="📋 复制模型", command=self._model_copy)
        menu.add_separator()
        menu.add_command(label="🗑 删除", command=self._model_delete)
        menu.tk_popup(event.x_root, event.y_root)

    def _model_copy(self):
        """复制当前选中的模型，名称添加 -Copy 后缀。"""
        db_id = self._get_selected_model_db_id()
        if db_id is None:
            return
        model = self._model_mgr.get_model_by_id(db_id)
        if not model:
            return
        self._model_mgr.add_model(
            model["type"],
            model["name"] + "-Copy",
            model["model_id"],
            model["base_url"],
            model["api_key_var"],
            model.get("workers", 4),
        )
        self._model_refresh()

    def _model_export_csv(self):
        """导出模型配置到 CSV"""
        import csv
        models = self._model_mgr.get_all_models()
        if not models:
            messagebox.showinfo("提示", "当前没有可导出的模型配置。")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="导出模型配置",
            initialfile="models_export.csv"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["模型类型", "模型名称", "模型ID", "接口地址", "API变量名", "并发数", "是否本地"])
                for m in models:
                    writer.writerow([
                        m.get("type", ""),
                        m.get("name", ""),
                        m.get("model_id", ""),
                        m.get("base_url", ""),
                        m.get("api_key_var", ""),
                        m.get("workers", 4),
                        "是" if m.get("is_local") else "否"
                    ])
            messagebox.showinfo("成功", f"成功导出 {len(models)} 个模型配置")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

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
        model_id, base_url, api_key, current_workers = api_params
            
        self.scan_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.scan_status_var.set("正在后台扫描，查看下方日志区...")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        face_hint = "准备中..." if self.enable_face_scan_var.get() else "未启用"
        self.log_text.insert(tk.END, f"[信息扫描] 准备中...\n[人物识别] {face_hint}")
        self.log_text.configure(state=tk.DISABLED)
        
        self._scan_cancel_event = threading.Event()
        # 代际计数器：防止旧线程的回调覆盖新扫描的 GUI 状态
        if not hasattr(self, '_scan_generation'):
            self._scan_generation = 0
        self._scan_generation += 1
        current_gen = self._scan_generation
        
        def _scan_thread():
            import logging
            from logging.handlers import RotatingFileHandler

            # ── 配置 Logging：控制台 + 文件（1MB 轮转） ──
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            old_handlers = root_logger.handlers[:]
            for h in old_handlers:
                root_logger.removeHandler(h)

            console_handler = logging.StreamHandler(sys.__stdout__)
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            root_logger.addHandler(console_handler)

            log_path = os.path.splitext(self.scan_db_var.get())[0] + ".log"
            file_handler = RotatingFileHandler(
                log_path, maxBytes=1024 * 1024, backupCount=1, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            ))
            root_logger.addHandler(file_handler)

            # ── GUI 进度条写入器（tqdm → Text 组件） ──
            llm_writer = TkLineWriter(self.log_text, 1)
            face_writer = TkLineWriter(self.log_text, 2)

            logging.info("扫描启动 — 模型: %s, 接口: %s, 并发: %d", model_id, base_url, current_workers)

            try:
                # 不再使用固定 DEFAULT_WORKERS，而是从模型配置传入
                # if "127.0.0.1" in base_url or "localhost" in base_url: 这段不再需要，因为如果是本地模型则必定 workers=1 已经控制了
                
                result_stats = scan(
                    paths=self.scan_paths,
                    db_path=self.scan_db_var.get(),
                    api_key=api_key,
                    base_url=base_url,
                    model=model_id,
                    rpm_limit=DEFAULT_RPM_LIMIT,
                    tpm_limit=DEFAULT_TPM_LIMIT,
                    workers=current_workers,
                    cancel_event=self._scan_cancel_event,
                    video_frame_interval=float(self.scan_frame_interval_var.get() or DEFAULT_VIDEO_FRAME_INTERVAL),
                    enable_face_scan=self.enable_face_scan_var.get(),
                    progress_writers=(llm_writer, face_writer),
                )
                # 只有当前代际仍然有效时才更新状态
                if current_gen == self._scan_generation:
                    if self._scan_cancel_event and not self._scan_cancel_event.is_set():
                        self.after(0, lambda: self.scan_status_var.set("扫描完成！可以去搜索页检索了。"))
                        def _append_summary(stats=result_stats):
                            self.log_text.configure(state=tk.NORMAL)
                            self.log_text.insert(tk.END, f"\n\n✨ 所有流水线处理完毕！\n"
                                f"   - [信息扫描] 耗时 {stats.get('llm_cost', 0):.1f}s, 共扫描 {stats.get('total', 0)} 张图像\n"
                                f"   - [人物识别] 耗时 {stats.get('face_cost', 0):.1f}s, 共识别出 {stats.get('face_found', 0)} 张含人脸\n"
                                f"   - 扫描与聚类总耗时: {stats.get('total_cost', 0):.1f}s\n\n")
                            self.log_text.configure(state=tk.DISABLED)
                            self.log_text.see(tk.END)
                        self.after(0, _append_summary)
                    else:
                        self.after(0, lambda: self.scan_status_var.set("扫描已中止。"))
            except Exception as e:
                if current_gen == self._scan_generation:
                    self.after(0, lambda e=e: self.scan_status_var.set(f"扫描出错: {e}"))
                    self.after(0, lambda e=e: messagebox.showerror("错误", f"扫描异常:\n{e}"))
            finally:
                root_logger.removeHandler(console_handler)
                root_logger.removeHandler(file_handler)
                file_handler.close()
                for h in old_handlers:
                    root_logger.addHandler(h)
                
                # 只有当前代际仍然有效时才恢复按钮状态
                if current_gen == self._scan_generation:
                    self.after(0, lambda: self.scan_btn.config(state=tk.NORMAL))
                    self.after(0, lambda: self.restart_btn.config(state=tk.DISABLED))
                    self.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                
        t = threading.Thread(target=_scan_thread, daemon=True)
        self._scan_thread_ref = t
        t.start()

    def restart_scan(self):
        """中止当前扫描并重新启动。"""
        if self._scan_cancel_event:
            self._scan_cancel_event.set()
        self.scan_status_var.set("正在停止当前扫描，请稍候...")
        self.restart_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

        def _wait_and_restart():
            # 等待旧线程真正结束（最多 30 秒）
            if self._scan_thread_ref and self._scan_thread_ref.is_alive():
                self._scan_thread_ref.join(timeout=30)
            self.after(0, self.start_scan)

        threading.Thread(target=_wait_and_restart, daemon=True).start()

    def stop_scan(self):
        """停止当前扫描（不重启）。"""
        if self._scan_cancel_event:
            self._scan_cancel_event.set()
        self.scan_status_var.set("正在停止扫描...")
        self.stop_btn.config(state=tk.DISABLED)
        self.restart_btn.config(state=tk.DISABLED)

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
        if hasattr(self, 'limit_entry'):
            self.limit_entry.config(state=state)

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
            model_id, base_url, api_key, _ = api_params
        else:
            model_id, base_url, api_key = "", "", ""

        self.toggle_state(tk.DISABLED)
        if is_llm_mode:
            self.status_var.set("正在使用大模型理解查询并召回数据，请稍候...")
        else:
            self.status_var.set("正在使用本地算法检索，请稍候...")
            
        self.update_idletasks()

        try:
            limit_val = int(self.search_limit_var.get())
            if limit_val <= 0:
                limit_val = 30
        except ValueError:
            limit_val = 30

        # 在后台线程执行搜索避免阻塞 GUI
        import time as _time
        def _search_thread():
            t0 = _time.perf_counter()
            try:
                results, warnings = search(
                    query=query,
                    db_paths=self.search_dbs,
                    limit=limit_val,
                    smart=is_llm_mode,
                    api_key=api_key,
                    base_url=base_url,
                    model=model_id,
                    rerank=is_llm_mode,
                )
                elapsed = _time.perf_counter() - t0
                self.after(0, self._on_search_done, results, None, elapsed, warnings)
            except Exception as e:
                elapsed = _time.perf_counter() - t0
                self.after(0, self._on_search_done, None, str(e), elapsed, [])

        threading.Thread(target=_search_thread, daemon=True).start()

    def _on_search_done(self, results, error, elapsed=0.0, warnings=None):
        """搜索完成回调，显示结果、耗时和警告。"""
        self.toggle_state(tk.NORMAL)
        time_str = f"（耗时 {elapsed:.1f}s）"

        if error is not None:
            self.status_var.set(f"搜索失败: {error} {time_str}")
            self.status_label.configure(foreground="red")
            messagebox.showerror("错误", f"搜索执行发生错误：{error}")
            return

        if not results:
            self.status_var.set(f"搜索完成：未找到匹配的图片。{time_str}")
            self.status_label.configure(foreground="blue")
            self.current_results = []
            self.current_index = 0
            self._update_display()
            self.show_gallery_view()
            self._clear_gallery()
            return

        # 检查是否有警告（如 LLM 排序失败）
        if warnings:
            warn_text = "; ".join(warnings)
            self.status_var.set(f"⚠️ {warn_text} —— 共找到 {len(results)} 张图片 {time_str}")
            self.status_label.configure(foreground="red")
        else:
            self.status_var.set(f"搜索完成：共找到并重排序 {len(results)} 张图片。{time_str}")
            self.status_label.configure(foreground="blue")

        self.current_results = results
        self.current_index = 0
        self.show_gallery_view()
        self._load_thumbnails()

    def show_gallery_view(self):
        self.preview_frame.pack_forget()
        self.gallery_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_preview_view(self):
        self.gallery_frame.pack_forget()
        self.preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _clear_gallery(self):
        self.gallery_text.configure(state="normal")
        self.gallery_text.delete(1.0, tk.END)
        self.gallery_text.configure(state="disabled")
        self.thumbnail_images.clear()

    def _load_thumbnails(self):
        self._clear_gallery()
        self._thumbnail_gen += 1
        current_gen = self._thumbnail_gen
        
        def _thread():
            for i, record in enumerate(self.current_results):
                if current_gen != self._thumbnail_gen:
                    break
                
                file_path = record.get("path", "")
                actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
                
                img = None
                try:
                    if os.path.isfile(actual_path):
                        frame_bytes = get_image_frame_bytes(actual_path)
                        img = Image.open(io.BytesIO(frame_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.thumbnail((150, 150), Image.Resampling.LANCZOS)
                except Exception:
                    pass
                
                if current_gen == self._thumbnail_gen:
                    self.after(0, self._add_thumbnail, i, record, img, current_gen)

        threading.Thread(target=_thread, daemon=True).start()

    def _add_thumbnail(self, index, record, img, gen):
        if gen != self._thumbnail_gen:
            return
            
        bg_color = getattr(self, "_bg_color", "#f0f0f0")
        item_frame = tk.Frame(self.gallery_text, bg=bg_color, cursor="hand2")
        
        if img:
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_images.append(photo)
        else:
            photo = None
            
        kwargs = {}
        if photo:
            kwargs['image'] = photo
        else:
            kwargs['text'] = "无图片"
            kwargs['width'] = 20
            kwargs['height'] = 10
        
        img_lbl = tk.Label(item_frame, bg=bg_color, **kwargs)
        img_lbl.pack(side="top")
        
        name = record.get("file_name", os.path.basename(record.get("path", "")))
        if len(name) > 12:
            name = name[:10] + "..."
        txt_lbl = tk.Label(item_frame, text=name, bg=bg_color, width=18)
        txt_lbl.pack(side="top")
        
        def on_click(event, idx=index):
            self.current_index = idx
            self._update_display()
            self.show_preview_view()
            
        item_frame.bind("<Button-1>", lambda e, idx=index: on_click(e, idx))
        img_lbl.bind("<Button-1>", lambda e, idx=index: on_click(e, idx))
        txt_lbl.bind("<Button-1>", lambda e, idx=index: on_click(e, idx))

        # 右键菜单
        menu = tk.Menu(item_frame, tearoff=0)
        def _copy_filename(r=record):
            file_name = r.get("file_name", os.path.basename(r.get("path", "")))
            self.clipboard_clear()
            self.clipboard_append(file_name)
        menu.add_command(label="复制文件名", command=lambda r=record: _copy_filename(r))
        
        def _open_loc(r=record):
            self._open_file_location_by_path(r.get("path", ""))
        menu.add_command(label="打开文件位置", command=lambda r=record: _open_loc(r))
        
        def _toggle_fav(r=record):
            from photo_identify.storage import Storage # Import Storage
            target_db = r.get("db_path", self.db_path)
            storage = Storage(target_db)
            new_fav = not bool(r.get("is_favorite", 0))
            storage.toggle_favorite(r["id"], new_fav)
            storage.close()
            # 本地更新当前记录的字段使状态正确
            r["is_favorite"] = 1 if new_fav else 0
            # If the favorites tab is open, refresh it.
            try:
                if self.notebook.tab(self.notebook.select(), "text") == "照片收藏":
                    self._refresh_favorites()
            except:
                pass
        menu.add_command(label="取消收藏" if record.get("is_favorite") else "加入收藏", command=lambda r=record: _toggle_fav(r))

        def show_menu(e, r=record):
            # 每次弹出前先根据当前记录更新菜单项文字
            is_fav = r.get("is_favorite", 0)
            # The index for "加入收藏" / "取消收藏" is 2 (0:复制文件名, 1:打开文件位置, 2:收藏)
            menu.entryconfigure(2, label="取消收藏" if is_fav else "加入收藏")
            menu.tk_popup(e.x_root, e.y_root)
            
        item_frame.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        img_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        txt_lbl.bind("<Button-3>", lambda e, r=record: show_menu(e, r))
        
        self.gallery_text.configure(state="normal")
        self.gallery_text.window_create("end", window=item_frame, padx=10, pady=10)
        self.gallery_text.configure(state="disabled")

    def show_prev(self):
        if self.current_results and self.current_index > 0:
            self.current_index -= 1
            self.current_rotation = 0
            self._update_display()

    def show_next(self):
        if self.current_results and self.current_index < len(self.current_results) - 1:
            self.current_index += 1
            self.current_rotation = 0
            self._update_display()

    def _update_display(self):
        total = len(self.current_results)
        if total == 0:
            self.info_var.set("暂无图片")
            self.desc_var.set("")
            self.page_var.set("0 / 0")
            self.canvas.delete("all")
            if hasattr(self, 'back_list_btn'):
                self.back_list_btn.config(state=tk.DISABLED)
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            self.open_dir_btn.config(state=tk.DISABLED)
            return

        self.page_var.set(f"{self.current_index + 1} / {total}")
        if hasattr(self, 'back_list_btn'):
            self.back_list_btn.config(state=tk.NORMAL)
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

    def _show_preview_context_menu(self, event):
        if not self.current_results or self.current_index >= len(self.current_results):
            return
            
        record = self.current_results[self.current_index]
        
        menu = tk.Menu(self.canvas, tearoff=0)
        
        def _copy_filename(r=record):
            file_name = r.get("file_name", os.path.basename(r.get("path", "")))
            self.clipboard_clear()
            self.clipboard_append(file_name)
        menu.add_command(label="复制文件名", command=_copy_filename)
        
        def _open_loc(r=record):
            self._open_file_location_by_path(r.get("path", ""))
        menu.add_command(label="打开文件位置", command=_open_loc)
        
        def _toggle_fav(r=record):
            from photo_identify.storage import Storage
            target_db = r.get("db_path", self.db_path)
            storage = Storage(target_db)
            new_fav = not bool(r.get("is_favorite", 0))
            storage.toggle_favorite(r["id"], new_fav)
            storage.close()
            r["is_favorite"] = 1 if new_fav else 0
            
            # 如果当前恰好是有这个星标的单张预览界面，手动同步更新星标状态
            # 我们通过修改 fav_btn 和 fav_btn_shadow 这两个 tag 来实现
            if new_fav:
                self.canvas.itemconfigure("fav_btn_shadow", text="★", state="normal")
                self.canvas.itemconfigure("fav_btn", text="★", fill="#e3b341", state="normal")
            else:
                self.canvas.itemconfigure("fav_btn_shadow", text="☆")
                self.canvas.itemconfigure("fav_btn", text="☆", fill="#d1d5da")
                # 不强制隐藏，由鼠标移动事件去接管
            
            # 如果从任意地方取消收藏/加入收藏之后，都应该去刷新一下 favorite_tab (如果已经加载)
            self._refresh_favorites()
            
        is_fav = record.get("is_favorite", 0)
        menu.add_command(label="取消收藏" if is_fav else "加入收藏", command=_toggle_fav)
        
        menu.tk_popup(event.x_root, event.y_root)

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
            self.original_image = ImageOps.exif_transpose(self.original_image)
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
        if not hasattr(self, 'original_image') or self.original_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        if self.current_rotation != 0:
            # 采用负数来实现顺时针(Pillow rotate 为逆时针), expand=True 保证旋转后四角不被裁切
            rotated_image = self.original_image.rotate(-self.current_rotation, expand=True)
        else:
            rotated_image = self.original_image

        img_width, img_height = rotated_image.size
        # 计算缩放比例，保持长宽比
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = max(1, int(img_width * ratio))
        new_height = max(1, int(img_height * ratio))

        # 使用 LANCZOS 进行高质量缩放
        resized_img = rotated_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_img)

        self.canvas.delete("all")
        # 居中显示
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, image=self.photo_image, anchor=tk.CENTER)
        
        # --- 收藏星星按钮逻辑 (GitHub 风格) ---
        record = self.current_results[self.current_index]
        is_fav = record.get("is_favorite", 0)
        
        # 统一使用 ☆ 和 ★ 
        # GitHub 空心星颜色为浅灰，实心星为金黄色
        star_text = "★" if is_fav else "☆"
        star_color = "#e3b341" if is_fav else "#d1d5da"
        
        # 放置在右上角
        margin_x, margin_y = 35, 35
        btn_x = canvas_width - margin_x
        btn_y = margin_y
        
        # --- 最大化/还原 按钮逻辑 ---
        is_max = getattr(self, 'is_maximized', False)
        max_text = "⤡" if is_max else "⤢"
        max_color = "#d1d5da"
        max_btn_x = btn_x - 35  # 在星标左侧
        max_btn_y = btn_y
        
        max_shadow_id = self.canvas.create_text(
            max_btn_x + 1, max_btn_y + 1, text=max_text, fill="black", font=("Segoe UI", 20),
            state="hidden", tags="max_btn_shadow"
        )
        max_star_id = self.canvas.create_text(
            max_btn_x, max_btn_y, text=max_text, fill=max_color, font=("Segoe UI", 20),
            state="hidden", tags="max_btn"
        )
        
        # --- 旋转 按钮逻辑 ---
        rot_text = "↻"
        rot_color = "#d1d5da"
        rot_btn_x = max_btn_x - 35  # 在最大化标识左侧
        rot_btn_y = btn_y
        
        rot_shadow_id = self.canvas.create_text(
            rot_btn_x + 1, rot_btn_y + 1, text=rot_text, fill="black", font=("Segoe UI", 20),
            state="hidden", tags="rot_btn_shadow"
        )
        rot_star_id = self.canvas.create_text(
            rot_btn_x, rot_btn_y, text=rot_text, fill=rot_color, font=("Segoe UI", 20),
            state="hidden", tags="rot_btn"
        )
        
        # 加微弱的黑色阴影，防止在白底图片上不可见 (字体缩小到原来约60%: font size 20)
        shadow_id = self.canvas.create_text(
            btn_x + 1, btn_y + 1, text=star_text, fill="black", font=("Segoe UI", 20),
            state="normal" if is_fav else "hidden", tags="fav_btn_shadow"
        )
        
        star_id = self.canvas.create_text(
            btn_x, btn_y, text=star_text, fill=star_color, font=("Segoe UI", 20),
            state="normal" if is_fav else "hidden", tags="fav_btn"
        )
        
        # 鼠标移动显示逻辑
        def on_canvas_motion(event):
            # 收藏按钮逻辑
            if not record.get("is_favorite", 0):
                if event.x > canvas_width - 80 and event.y < 80:
                    self.canvas.itemconfigure(shadow_id, state="normal")
                    self.canvas.itemconfigure(star_id, state="normal")
                else:
                    self.canvas.itemconfigure(shadow_id, state="hidden")
                    self.canvas.itemconfigure(star_id, state="hidden")
                    
            # 最大化/看图 按钮逻辑
            if event.x > canvas_width - 120 and event.y < 80:
                self.canvas.itemconfigure(max_shadow_id, state="normal")
                self.canvas.itemconfigure(max_star_id, state="normal")
                self.canvas.itemconfigure(rot_shadow_id, state="normal")
                self.canvas.itemconfigure(rot_star_id, state="normal")
            else:
                self.canvas.itemconfigure(max_shadow_id, state="hidden")
                self.canvas.itemconfigure(max_star_id, state="hidden")
                self.canvas.itemconfigure(rot_shadow_id, state="hidden")
                self.canvas.itemconfigure(rot_star_id, state="hidden")
                
        def on_canvas_leave(event):
            if not record.get("is_favorite", 0):
                self.canvas.itemconfigure(shadow_id, state="hidden")
                self.canvas.itemconfigure(star_id, state="hidden")
            self.canvas.itemconfigure(max_shadow_id, state="hidden")
            self.canvas.itemconfigure(max_star_id, state="hidden")
            self.canvas.itemconfigure(rot_shadow_id, state="hidden")
            self.canvas.itemconfigure(rot_star_id, state="hidden")
            
        self.canvas.bind("<Motion>", on_canvas_motion)
        self.canvas.bind("<Leave>", on_canvas_leave)
        
        # 点击切换逻辑
        def on_star_click(event):
            from photo_identify.storage import Storage
            target_db = record.get("db_path", self.db_path)
            storage = Storage(target_db)
            
            new_fav = not bool(record.get("is_favorite", 0))
            storage.toggle_favorite(record["id"], new_fav)
            storage.close()
            
            record["is_favorite"] = 1 if new_fav else 0
            
            # 更新视觉
            if new_fav:
                self.canvas.itemconfigure(shadow_id, text="★", state="normal")
                self.canvas.itemconfigure(star_id, text="★", fill="#e3b341", state="normal")
            else:
                self.canvas.itemconfigure(shadow_id, text="☆")
                self.canvas.itemconfigure(star_id, text="☆", fill="#d1d5da")
                # 因为此时取消了收藏，应当立刻根据鼠标位置来决定是否该隐藏
                self.canvas.itemconfigure(shadow_id, state="normal") # 既然刚点完说明鼠标还在按钮上
                self.canvas.itemconfigure(star_id, state="normal")
                
            # 若收藏页已经加载过，刷新一下
            try:
                if self.notebook.tab(self.notebook.select(), "text") == "照片收藏":
                    self._refresh_favorites()
            except:
                pass

        self.canvas.tag_bind(star_id, "<Button-1>", on_star_click)
        self.canvas.tag_bind(max_star_id, "<Button-1>", self._open_in_system_viewer)
        self.canvas.tag_bind(rot_star_id, "<Button-1>", self._rotate_image)

    def _rotate_image(self, event=None):
        self.current_rotation = (self.current_rotation + 90) % 360
        self._resize_and_display()

    def _open_in_system_viewer(self, event=None):
        if not self.current_results:
            return
        record = self.current_results[self.current_index]
        file_path = record.get("path", "")
        if not file_path:
            return
            
        actual_path = file_path.split("#t=")[0] if "#t=" in file_path else file_path
        if not os.path.exists(actual_path):
            messagebox.showerror("错误", "文件不存在！")
            return
            
        try:
            if sys.platform == "win32":
                os.startfile(actual_path)
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", actual_path])
            else:  # Linux
                subprocess.Popen(["xdg-open", actual_path])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开图片: {e}")

    def open_file_location(self):
        if not self.current_results:
            return
        record = self.current_results[self.current_index]
        self._open_file_location_by_path(record.get("path", ""))

    def _open_file_location_by_path(self, file_path):
        if not file_path:
            return
            
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
