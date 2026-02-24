import configparser
import math
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import scrolledtext

from PIL import Image, ImageTk

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_DB_PATH,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VIDEO_FRAME_INTERVAL,
    DEFAULT_VISION_MODEL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_TPM_LIMIT,
    load_api_key,
)
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


class PhotoIdentifyGUI(tk.Tk):
    def __init__(self, db_path: str = str(DEFAULT_DB_PATH)):
        super().__init__()
        self.title("AI 图片语义检索与扫描")
        self.geometry("900x700")
        self.minsize(800, 600)

        self.db_path = db_path
        self.search_dbs = [db_path] if db_path else []
        
        # Initialize variables
        self.current_results = []
        self.current_index = 0
        self.photo_image = None
        self.original_image = None
        self._resize_timer = None

        self.model_var = tk.StringVar(value=DEFAULT_TEXT_MODEL)
        self.query_var = tk.StringVar(value="")
        self.search_mode_var = tk.StringVar(value="llm")
        self.status_var = tk.StringVar(value="准备就绪。")
        self.info_var = tk.StringVar(value="暂无图片")
        self.desc_var = tk.StringVar(value="")
        self.page_var = tk.StringVar(value="0 / 0")

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
        if cfg.has_option("search", "model"):
            self.model_var.set(cfg.get("search", "model"))
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
        if cfg.has_option("scan", "model"):
            self.scan_model_var.set(cfg.get("scan", "model"))
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
            "model": self.model_var.get(),
            "mode": self.search_mode_var.get(),
            "databases": "|".join(self.search_dbs),
        }
        cfg["scan"] = {
            "model": self.scan_model_var.get(),
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
        self.destroy()
    def open_env_vars(self):
        """打开 Windows 环境变量编辑窗口"""
        if sys.platform == "win32":
            subprocess.Popen("rundll32 sysdm.cpl,EditEnvironmentVariables")
            messagebox.showinfo("提示", "修改环境变量后，可能需要重启本应用才能生效。")
        else:
            messagebox.showinfo("提示", "仅在 Windows 下支持快速打开环境变量编辑窗口。")

    def _init_search_tab(self):
        self.top_frame = ttk.Frame(self.search_tab, padding="10")

        # Row 0: Model Name and Edit Env Button
        ttk.Label(self.top_frame, text="模型名称:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_entry = ttk.Entry(self.top_frame, textvariable=self.model_var, width=30)
        self.model_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(self.top_frame, text="⚙ 环境变量设置", command=self.open_env_vars).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)

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

    def _init_scan_tab(self):
        self.scan_model_var = tk.StringVar(value=DEFAULT_VISION_MODEL)
        self.scan_db_var = tk.StringVar(value=self.db_path)
        self.scan_paths = []
        
        form_frame = ttk.Frame(self.scan_tab, padding="20")
        form_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(form_frame, text="视觉模型名称:").grid(row=0, column=0, sticky=tk.W, pady=10)
        ttk.Entry(form_frame, textvariable=self.scan_model_var, width=40).grid(row=0, column=1, sticky=tk.W, pady=10)
        
        ttk.Button(form_frame, text="⚙ 环境变量设置", command=self.open_env_vars).grid(row=0, column=2, padx=10, sticky=tk.W, pady=10)
        
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
        
        api_key = load_api_key()
        if not api_key:
            messagebox.showwarning("警告", "未找到 API Key，请在环境变量配置 SILICONFLOW_API_KEY")
            return
            
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
            print("正在遍历与收集指定目录内所有符合条件的多媒体文件。")
            print("如果您选择了体积庞大的盘符或含有万级文件的目录，这可能需要数分钟的搜寻时间，请耐心等待...")
            print("==================================================\n")
            
            try:
                scan(
                    paths=self.scan_paths,
                    db_path=self.scan_db_var.get(),
                    api_key=api_key,
                    model=self.scan_model_var.get(),
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

        api_key = load_api_key()
        model = self.model_var.get().strip()
        is_llm_mode = (self.search_mode_var.get() == "llm")

        if is_llm_mode and not api_key:
            messagebox.showwarning("警告", "未在环境变量中找到有效的 API Key！请点击环境变量设置配置 SILICONFLOW_API_KEY。")
            return

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
                    smart=is_llm_mode, # 这里我们需要保证 smart 不仅做扩充，还要做重排
                    api_key=api_key,
                    base_url=DEFAULT_BASE_URL,
                    model=model,  # 新增传递给 search 的参数
                    rerank=is_llm_mode,  # 新增标志
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
        if not os.path.isfile(file_path):
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width()//2, self.canvas.winfo_height()//2,
                text="图片文件不存在", fill="red", font=("Arial", 16)
            )
            return

        try:
            self.original_image = Image.open(file_path)
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
        if not os.path.exists(file_path):
            messagebox.showerror("错误", "文件不存在！")
            return

        # 在 Windows 上选中文件并在资源管理器打开
        try:
            if sys.platform == "win32":
                subprocess.Popen(f'explorer /select,"{os.path.normpath(file_path)}"')
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", "-R", file_path])
            else:  # Linux (fallback)
                subprocess.Popen(["xdg-open", os.path.dirname(file_path)])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件位置: {e}")


def launch_gui(db_path: str = str(DEFAULT_DB_PATH)):
    """启动图形界面主循环"""
    app = PhotoIdentifyGUI(db_path)
    app.mainloop()

if __name__ == "__main__":
    launch_gui()
