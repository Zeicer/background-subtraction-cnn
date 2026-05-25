import os
import queue
import json
import shutil
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
# import finalsecond_zichen
import finalsecond
from function_monitor import PerformanceMonitor
import main_run
import realtime_run
import test_new_HyRGB


APP_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(APP_DIR, "settings.json")
SETTINGS_VERSION = 1

EVENT_LABELS = {
    "running": "奔跑",
    "loiter": "徘徊",
    "faint": "跌倒",
    "collision": "碰撞",
    "litter": "丟垃圾"
}

EVENT_DIR_NAMES = {
    "running": "RUNNING",
    "loiter": "LOITERING",
    "faint": "FAINT",
    "collision": "COLLISION",
    "litter": "LITTERING"
}


class AppLog:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        message = str(message)
        self.text_widget.after(0, self._append, message)

    def _append(self, message):
        self.text_widget.insert(tk.END, message + "\n")
        self.text_widget.see(tk.END)


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title, width=320, height=240):
        super().__init__(parent)
        self.title = ttk.Label(self, text=title)
        self.title.pack(anchor="w")
        self.label = ttk.Label(self)
        self.label.pack(fill="both", expand=True)
        self.width = width
        self.height = height
        self.photo = None
        self.last_frame = None
        self.fill_window = False

        self.label.bind("<Configure>", self._on_resize)

    def set_fill_window(self, enabled):
        self.fill_window = enabled

        if self.last_frame is not None:
            self.set_image(self.last_frame)

    def _on_resize(self, event):
        if self.fill_window and self.last_frame is not None:
            self.set_image(self.last_frame)

    def set_image(self, frame):
        if frame is None:
            return

        self.last_frame = frame

        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        if self.fill_window:
            target_width = max(1, self.label.winfo_width())
            target_height = max(1, self.label.winfo_height())
            image = image.resize(
                (target_width, target_height),
                Image.LANCZOS
            )
        else:
            pass

        self.photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self.photo)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f5f7fb")
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )
        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.inner,
            anchor="nw"
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_inner)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_inner(self, event):
        self.canvas.itemconfigure(self.window_id, width=max(event.width, 1))

    def _bind_mousewheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class RunApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("HyRGB 行為監控系統")
        self.geometry("1320x860")
        self.minsize(1100, 720)
        self._setup_style()

        self.stop_event = threading.Event()
        self.realtime_thread = None
        self.offline_thread = None
        self.monitor = None
        self.result_queue = queue.Queue(maxsize=8)
        self.output_buffer = deque(maxlen=8)
        self.realtime_history = deque(maxlen=10000)
        self.realtime_seek_mode = False
        self.timeline_updating = False
        self.playback_results = []
        self.playback_index = 0
        self.offline_processing = False
        self.last_result = None
        self.last_realtime_result = None
        self.last_offline_result = None
        self.last_display_time = 0.0
        self.large_window = None
        self.large_panel = None
        self.large_view_key = None
        self.event_alert = False
        self.display_frame_count = 0
        self.fps_last_time = time.perf_counter()
        self.current_display_fps = tk.StringVar(value="顯示 FPS: 0.0")
        self.event_buttons = {}
        self.event_button_alerts = {}
        self.event_items = {event: [] for event in EVENT_LABELS}
        self.event_seen = set()
        self.event_last_accept_time = {}
        self.event_pages = {event: 0 for event in EVENT_LABELS}
        self.event_frame_buffer = deque(maxlen=300)
        self.event_cover_refs = []
        self.selected_event = None
        self.event_cover_frame = None
        self.realtime_panels = {}
        self.offline_panels = {}
        self.show_new_only = tk.BooleanVar(value=False)
        self.event_search = tk.StringVar(value="")
        self.stopping = False
        self.offline_progress_counts = (0, 0)

        self.image_refs = {}

        self._build_vars()
        self._load_settings()
        self._build_ui()
        self._refresh_cameras()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(30, self._poll_results)

    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabelframe", background="#f5f7fb")
        style.configure("TLabelframe.Label", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("TLabel", background="#f5f7fb", font=("Microsoft JhengHei UI", 10))
        style.configure("TButton", font=("Microsoft JhengHei UI", 10), padding=6)
        style.configure("TNotebook", background="#eef2f7")
        style.configure("TNotebook.Tab", font=("Microsoft JhengHei UI", 10), padding=(14, 7))

    def _build_vars(self):
        self.base_dir = tk.StringVar(value=APP_DIR)
        self.roi_model_dir = tk.StringVar(
            value=os.path.join(
                os.path.dirname(APP_DIR),
                "dataset",
                "batch_path"
            )
        )
        self.dataset_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="app_output/combined_results")
        self.background_path = tk.StringVar(
            value=os.path.join(APP_DIR, "background.jpg")
        )
        self.camera_id = tk.StringVar(value="")
        self.save_mode = tk.StringVar(value="event")
        self.offline_save_mode = tk.StringVar(value="all")

        self.varThreshold = tk.DoubleVar(value=16)
        self.active_ratio_threshold = tk.DoubleVar(value=0.35)
        self.active_indices_threshold = tk.IntVar(value=200)
        self.active_indices_limit = tk.IntVar(value=8000)
        self.learningRate = tk.DoubleVar(value=0.001)
        self.update_interval = tk.IntVar(value=1)
        self.batch_size = tk.IntVar(value=2048)
        self.foreground_threshold = tk.DoubleVar(value=0.9)

        self.auto_varThreshold = tk.BooleanVar(value=True)
        self.varThreshold_min = tk.DoubleVar(value=8)
        self.varThreshold_max = tk.DoubleVar(value=64)
        self.varThreshold_step = tk.DoubleVar(value=2)
        self.active_ratio_spike = tk.DoubleVar(value=0.20)
        self.active_ratio_drop = tk.DoubleVar(value=0.08)
        self.active_ratio_smooth = tk.DoubleVar(value=0.4)

        self.crop_margin = tk.IntVar(value=50)
        self.display_fps = tk.DoubleVar(value=10)
        self.display_delay_seconds = tk.DoubleVar(value=0.5)
        self.retention_days = tk.IntVar(value=5)
        self.playback_paused = tk.BooleanVar(value=False)
        self.playback_speed = tk.DoubleVar(value=1.0)
        self.large_view = tk.StringVar(value="view2_skeleton")
        self.large_view_fill = tk.BooleanVar(value=False)
        self.offline_progress = tk.DoubleVar(value=0.0)
        self.offline_progress_text = tk.StringVar(value="尚未開始")
        self.realtime_timeline = tk.DoubleVar(value=180.0)
        self.realtime_timeline_text = tk.StringVar(value="即時回顧: 最新")
        self.offline_timeline = tk.DoubleVar(value=0.0)
        self.offline_timeline_text = tk.StringVar(value="圖集回顧: 0/0")
        self.realtime_settings_collapsed = tk.BooleanVar(value=False)
        self.offline_settings_collapsed = tk.BooleanVar(value=False)

    def _settings_vars(self):
        return {
            "base_dir": self.base_dir,
            "roi_model_dir": self.roi_model_dir,
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
            "background_path": self.background_path,
            "camera_id": self.camera_id,
            "save_mode": self.save_mode,
            "offline_save_mode": self.offline_save_mode,
            "varThreshold": self.varThreshold,
            "active_ratio_threshold": self.active_ratio_threshold,
            "active_indices_threshold": self.active_indices_threshold,
            "active_indices_limit": self.active_indices_limit,
            "learningRate": self.learningRate,
            "update_interval": self.update_interval,
            "batch_size": self.batch_size,
            "foreground_threshold": self.foreground_threshold,
            "auto_varThreshold": self.auto_varThreshold,
            "varThreshold_min": self.varThreshold_min,
            "varThreshold_max": self.varThreshold_max,
            "varThreshold_step": self.varThreshold_step,
            "active_ratio_spike": self.active_ratio_spike,
            "active_ratio_drop": self.active_ratio_drop,
            "active_ratio_smooth": self.active_ratio_smooth,
            "crop_margin": self.crop_margin,
            "display_fps": self.display_fps,
            "display_delay_seconds": self.display_delay_seconds,
            "retention_days": self.retention_days,
            "playback_speed": self.playback_speed,
            "large_view": self.large_view,
            "large_view_fill": self.large_view_fill,
            "realtime_settings_collapsed": self.realtime_settings_collapsed,
            "offline_settings_collapsed": self.offline_settings_collapsed,
        }

    def _load_settings(self):
        if not os.path.exists(SETTINGS_PATH):
            return

        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as exc:
            broken_path = os.path.join(
                APP_DIR,
                f"settings.broken.{int(time.time())}.json"
            )

            try:
                shutil.copy2(SETTINGS_PATH, broken_path)
                print(f"設定檔讀取失敗，已備份: {broken_path} ({exc})")
            except OSError:
                print(f"設定檔讀取失敗: {exc}")

            return

        if data.get("version") not in (None, SETTINGS_VERSION):
            print(f"設定檔版本不同，仍嘗試讀取: {data.get('version')}")

        for key, var in self._settings_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except tk.TclError:
                    pass

    def _save_settings(self):
        data = {
            "version": SETTINGS_VERSION,
            **{key: var.get() for key, var in self._settings_vars().items()}
        }

        with open(SETTINGS_PATH, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

        self._log(f"設定已儲存: {SETTINGS_PATH}")

    def export_settings(self):
        path = filedialog.asksaveasfilename(
            initialdir=APP_DIR,
            initialfile="settings_export.json",
            defaultextension=".json",
            filetypes=(("JSON 設定檔", "*.json"), ("所有檔案", "*.*"))
        )

        if not path:
            return

        self._save_settings()
        shutil.copy2(SETTINGS_PATH, path)
        self._log(f"設定已匯出: {path}")

    def import_settings(self):
        path = filedialog.askopenfilename(
            initialdir=APP_DIR,
            filetypes=(("JSON 設定檔", "*.json"), ("所有檔案", "*.*"))
        )

        if not path:
            return

        shutil.copy2(path, SETTINGS_PATH)
        self._load_settings()
        self._log(f"設定已匯入: {path}")
        messagebox.showinfo("設定檔", "設定已匯入，部分介面狀態重開 app 後會完整套用。")

    def _on_close(self):
        try:
            self._save_settings()
        except Exception as exc:
            self._log(f"設定儲存失敗: {exc}")

        self.stop_event.set()
        self.destroy()

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.realtime_tab = ttk.Frame(self.tabs, padding=8)
        self.offline_tab = ttk.Frame(self.tabs, padding=8)
        self.params_tab = ttk.Frame(self.tabs, padding=8)
        self.events_tab = ttk.Frame(self.tabs, padding=8)
        self.log_tab = ttk.Frame(self.tabs, padding=8)

        self.tabs.add(self.realtime_tab, text="即時模式")
        self.tabs.add(self.offline_tab, text="資料夾模式")
        self.tabs.add(self.params_tab, text="參數設定")
        self.tabs.add(self.events_tab, text="事件資料夾")
        self.tabs.add(self.log_tab, text="執行紀錄")

        self._build_realtime_tab()
        self._build_offline_tab()
        self._build_params_tab()
        self._build_events_tab()
        self._build_log_tab()
        self._build_status_bar(root)

    def _build_status_bar(self, parent):
        self.status_text = tk.StringVar(value="")
        status = ttk.Label(
            parent,
            textvariable=self.status_text,
            anchor="w",
            padding=(6, 4)
        )
        status.pack(fill="x", pady=(6, 0))
        self._update_status_bar()

    def _update_status_bar(self):
        if self.tabs.select() == str(self.realtime_tab):
            mode = "即時模式"
        elif self.tabs.select() == str(self.offline_tab):
            mode = "資料夾模式"
        elif self.tabs.select() == str(self.events_tab):
            mode = "事件資料夾"
        elif self.tabs.select() == str(self.params_tab):
            mode = "參數設定"
        else:
            mode = "執行紀錄"

        event_count = sum(len(items) for items in self.event_items.values())
        new_count = sum(
            1
            for items in self.event_items.values()
            for item in items
            if item.get("is_new")
        )
        saving = "儲存中" if self.monitor is not None else "未儲存"
        self.status_text.set(
            f"目前模式: {mode} | {self.current_display_fps.get()} | "
            f"{saving} | 事件: {event_count} / 新事件: {new_count} | "
            f"保存天數: {self.retention_days.get()}"
        )

        self.after(1000, self._update_status_bar)

    def _build_path_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="選擇", command=command).grid(
            row=row,
            column=2,
            sticky="ew",
            pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _build_collapsible_settings(self, parent, title, collapsed_var):
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill="x")

        header = ttk.Frame(wrapper)
        header.pack(fill="x")

        button_text = tk.StringVar()
        content = ttk.LabelFrame(wrapper, text=title, padding=8)

        def sync():
            if collapsed_var.get():
                content.pack_forget()
                button_text.set(f"展開{title}")
            else:
                content.pack(fill="x", pady=(4, 0))
                button_text.set(f"收起{title}")

        def toggle():
            collapsed_var.set(not collapsed_var.get())
            sync()

        ttk.Button(
            header,
            textvariable=button_text,
            command=toggle
        ).pack(side="left")

        sync()
        return content

    def _build_realtime_tab(self):
        top = self._build_collapsible_settings(
            self.realtime_tab,
            "即時模式設定",
            self.realtime_settings_collapsed
        )

        self._build_path_row(
            top,
            0,
            "基礎資料夾",
            self.base_dir,
            lambda: self._choose_folder(self.base_dir)
        )
        self._build_path_row(
            top,
            1,
            "ROI 模型資料夾",
            self.roi_model_dir,
            lambda: self._choose_folder(self.roi_model_dir)
        )
        self._build_path_row(
            top,
            2,
            "背景圖路徑",
            self.background_path,
            lambda: self._choose_file(self.background_path)
        )

        ttk.Label(top, text="攝影機").grid(row=3, column=0, sticky="w", pady=3)
        self.camera_combo = ttk.Combobox(
            top,
            textvariable=self.camera_id,
            state="readonly",
            width=20
        )
        self.camera_combo.grid(row=3, column=1, sticky="w", padx=6, pady=3)
        ttk.Button(top, text="重新掃描", command=self._refresh_cameras).grid(
            row=3,
            column=2,
            sticky="ew",
            pady=3
        )

        ttk.Label(top, text="儲存模式").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(
            top,
            textvariable=self.save_mode,
            state="readonly",
            values=("none", "event", "all"),
            width=12
        ).grid(row=4, column=1, sticky="w", padx=6, pady=3)

        controls = ttk.Frame(top)
        controls.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        run_row = ttk.LabelFrame(controls, text="執行", padding=4)
        play_row = ttk.LabelFrame(controls, text="播放", padding=4)
        view_row = ttk.LabelFrame(controls, text="視窗", padding=4)
        run_row.pack(fill="x", pady=2)
        play_row.pack(fill="x", pady=2)
        view_row.pack(fill="x", pady=2)
        self.start_realtime_button = ttk.Button(
            run_row,
            text="開始即時",
            command=self.start_realtime
        )
        self.start_realtime_button.pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(run_row, text="停止", command=self.stop_current).pack(
            side="left",
            padx=6
        )
        ttk.Button(
            run_row,
            text="拍攝背景",
            command=self.capture_background
        ).pack(side="left", padx=6)

        ttk.Button(
            play_row,
            text="播放/暫停",
            command=self.toggle_playback
        ).pack(side="left", padx=6)

        ttk.Label(play_row, text="倍速").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            play_row,
            textvariable=self.playback_speed,
            state="readonly",
            values=(0.25, 0.5, 1.0, 1.5, 2.0, 4.0),
            width=6
        ).pack(side="left", padx=4)

        ttk.Button(
            play_row,
            text="從頭播放",
            command=self.restart_playback
        ).pack(side="left", padx=6)

        ttk.Label(view_row, text="大視窗").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            view_row,
            textvariable=self.large_view,
            state="readonly",
            values=(
                "view1_box_only",
                "view2_skeleton",
                "view3_mask_overlay",
                "view4_mask",
                "view5_original"
            ),
            width=20
        ).pack(side="left", padx=4)
        ttk.Button(
            view_row,
            text="開啟大視窗",
            command=self.open_large_view
        ).pack(side="left", padx=6)

        ttk.Checkbutton(
            view_row,
            text="填滿視窗",
            variable=self.large_view_fill,
            command=self.update_large_view_fill
        ).pack(side="left", padx=6)

        image_scroll = ScrollableFrame(self.realtime_tab)
        image_scroll.pack(fill="both", expand=True, pady=(8, 0))
        images = image_scroll.inner

        self.realtime_panels = {
            "view1_box_only": ImagePanel(images, "1. 只顯示框線"),
            "view2_skeleton": ImagePanel(images, "2. 框線 + 骨架"),
            "view3_mask_overlay": ImagePanel(images, "3. 遮罩疊圖"),
            "view4_mask": ImagePanel(images, "4. 純遮罩"),
            "view5_original": ImagePanel(images, "5. 原始畫面"),
        }
        self.panels = self.realtime_panels

        for idx, panel in enumerate(self.realtime_panels.values()):
            panel.grid(
                row=idx // 3,
                column=idx % 3,
                sticky="nsew",
                padx=6,
                pady=6
            )

        for col in range(3):
            images.columnconfigure(col, weight=1)
        for row in range(2):
            images.rowconfigure(row, weight=1)

        fps_bar = ttk.Frame(self.realtime_tab)
        fps_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(
            fps_bar,
            textvariable=self.current_display_fps
        ).pack(side="left")

        timeline = ttk.Frame(self.realtime_tab)
        timeline.pack(fill="x", pady=(6, 0))
        ttk.Label(timeline, text="3 分鐘回顧").pack(side="left", padx=(0, 8))
        self.realtime_timeline_scale = ttk.Scale(
            timeline,
            from_=0,
            to=180,
            orient="horizontal",
            variable=self.realtime_timeline,
            command=self.seek_realtime_timeline
        )
        self.realtime_timeline_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(
            timeline,
            textvariable=self.realtime_timeline_text,
            width=18
        ).pack(side="left", padx=8)
        ttk.Button(
            timeline,
            text="回到最新",
            command=self.jump_realtime_latest
        ).pack(side="left")

    def _build_offline_tab(self):
        top = self._build_collapsible_settings(
            self.offline_tab,
            "資料夾模式設定",
            self.offline_settings_collapsed
        )

        self._build_path_row(
            top,
            0,
            "基礎資料夾",
            self.base_dir,
            lambda: self._choose_folder(self.base_dir)
        )
        self._build_path_row(
            top,
            1,
            "影像資料夾",
            self.dataset_dir,
            lambda: self._choose_folder(self.dataset_dir)
        )
        self._build_path_row(
            top,
            2,
            "ROI 模型資料夾",
            self.roi_model_dir,
            lambda: self._choose_folder(self.roi_model_dir)
        )
        self._build_path_row(
            top,
            3,
            "輸出資料夾",
            self.output_dir,
            lambda: self._choose_folder(self.output_dir)
        )
        self._build_path_row(
            top,
            4,
            "背景圖路徑",
            self.background_path,
            lambda: self._choose_file(self.background_path)
        )

        ttk.Label(top, text="儲存模式").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(
            top,
            textvariable=self.offline_save_mode,
            state="readonly",
            values=("none", "event", "all"),
            width=12
        ).grid(row=5, column=1, sticky="w", padx=6, pady=3)

        controls = ttk.Frame(top)
        controls.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        run_row = ttk.LabelFrame(controls, text="執行", padding=4)
        play_row = ttk.LabelFrame(controls, text="播放", padding=4)
        view_row = ttk.LabelFrame(controls, text="視窗", padding=4)
        run_row.pack(fill="x", pady=2)
        play_row.pack(fill="x", pady=2)
        view_row.pack(fill="x", pady=2)
        self.start_offline_button = ttk.Button(
            run_row,
            text="開始資料夾分析",
            command=self.start_offline
        )
        self.start_offline_button.pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(run_row, text="停止", command=self.stop_current).pack(
            side="left",
            padx=6
        )
        ttk.Button(
            play_row,
            text="播放/暫停",
            command=self.toggle_playback
        ).pack(side="left", padx=6)
        ttk.Label(play_row, text="倍速").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            play_row,
            textvariable=self.playback_speed,
            state="readonly",
            values=(0.25, 0.5, 1.0, 1.5, 2.0, 4.0),
            width=6
        ).pack(side="left", padx=4)
        ttk.Button(play_row, text="從頭播放", command=self.restart_playback).pack(
            side="left",
            padx=6
        )
        ttk.Label(view_row, text="大視窗").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            view_row,
            textvariable=self.large_view,
            state="readonly",
            values=(
                "view1_box_only",
                "view2_skeleton",
                "view3_mask_overlay",
                "view4_mask",
                "view5_original"
            ),
            width=20
        ).pack(side="left", padx=4)
        ttk.Button(
            view_row,
            text="開啟大視窗",
            command=self.open_large_view
        ).pack(side="left", padx=6)
        ttk.Checkbutton(
            view_row,
            text="填滿視窗",
            variable=self.large_view_fill,
            command=self.update_large_view_fill
        ).pack(side="left", padx=6)

        progress_frame = ttk.Frame(top)
        progress_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 2))
        ttk.Label(progress_frame, text="處理進度").pack(side="left", padx=(0, 8))
        ttk.Progressbar(
            progress_frame,
            variable=self.offline_progress,
            maximum=100
        ).pack(side="left", fill="x", expand=True)
        ttk.Label(
            progress_frame,
            textvariable=self.offline_progress_text,
            width=18
        ).pack(side="left", padx=(8, 0))

        info = ttk.Label(
            self.offline_tab,
            text="選擇影像資料夾後，程式會逐張分析並把結果即時顯示在五格畫面中。背景圖可獨立指定，不必放在基礎資料夾內。",
            wraplength=900
        )
        info.pack(anchor="w", pady=10)

        timeline = ttk.LabelFrame(self.offline_tab, text="圖集回顧時間軸", padding=8)
        timeline.pack(fill="x", pady=(0, 8))
        self.offline_timeline_scale = ttk.Scale(
            timeline,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.offline_timeline,
            command=self.seek_offline_timeline
        )
        self.offline_timeline_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(
            timeline,
            textvariable=self.offline_timeline_text,
            width=18
        ).pack(side="left", padx=8)

        image_scroll = ScrollableFrame(self.offline_tab)
        image_scroll.pack(fill="both", expand=True, pady=(8, 0))
        images = image_scroll.inner
        self.offline_panels = {
            "view1_box_only": ImagePanel(images, "1. 只顯示框線"),
            "view2_skeleton": ImagePanel(images, "2. 框線 + 骨架"),
            "view3_mask_overlay": ImagePanel(images, "3. 遮罩疊圖"),
            "view4_mask": ImagePanel(images, "4. 純遮罩"),
            "view5_original": ImagePanel(images, "5. 原始畫面"),
        }

        for idx, panel in enumerate(self.offline_panels.values()):
            panel.grid(
                row=idx // 3,
                column=idx % 3,
                sticky="nsew",
                padx=6,
                pady=6
            )

        for col in range(3):
            images.columnconfigure(col, weight=1)
        for row in range(2):
            images.rowconfigure(row, weight=1)

    def _build_params_tab(self):
        params_scroll = ScrollableFrame(self.params_tab)
        params_scroll.pack(fill="both", expand=True)
        params_root = params_scroll.inner

        groups = [
            (
                "HyRGB 偵測參數",
                [
                    ("varThreshold", self.varThreshold),
                    ("active_ratio_threshold", self.active_ratio_threshold),
                    ("active_indices_threshold", self.active_indices_threshold),
                    ("active_indices_limit", self.active_indices_limit),
                    ("learningRate", self.learningRate),
                    ("update_interval", self.update_interval),
                    ("batch_size", self.batch_size),
                    ("foreground_threshold", self.foreground_threshold),
                ],
            ),
            (
                "自動敏感度參數",
                [
                    ("varThreshold_min", self.varThreshold_min),
                    ("varThreshold_max", self.varThreshold_max),
                    ("varThreshold_step", self.varThreshold_step),
                    ("active_ratio_spike", self.active_ratio_spike),
                    ("active_ratio_drop", self.active_ratio_drop),
                    ("active_ratio_smooth", self.active_ratio_smooth),
                ],
            ),
            (
                "GUI 播放參數",
                [
                    ("crop_margin", self.crop_margin),
                    ("display_fps", self.display_fps),
                    ("display_delay_seconds", self.display_delay_seconds),
                    ("playback_speed", self.playback_speed),
                ],
            ),
            (
                "輸出 / 儲存參數",
                [
                    ("即時儲存模式", self.save_mode),
                    ("資料夾儲存模式", self.offline_save_mode),
                    ("輸出資料夾", self.output_dir),
                    ("背景圖路徑", self.background_path),
                    ("資料保存天數", self.retention_days),
                ],
            ),
        ]

        for group_index, (title, params) in enumerate(groups):
            frame = ttk.LabelFrame(params_root, text=title, padding=8)
            frame.grid(
                row=group_index // 2,
                column=group_index % 2,
                sticky="nsew",
                padx=6,
                pady=6
            )

            for idx, (label, var) in enumerate(params):
                ttk.Label(frame, text=label).grid(row=idx, column=0, sticky="w", pady=3)
                ttk.Entry(frame, textvariable=var, width=24).grid(
                    row=idx,
                    column=1,
                    sticky="ew",
                    padx=8,
                    pady=3
                )

            frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            params_root,
            text="啟用自動調整 varThreshold",
            variable=self.auto_varThreshold
        ).grid(row=2, column=0, sticky="w", padx=8, pady=8)

        ttk.Button(
            params_root,
            text="儲存設定",
            command=self._save_settings
        ).grid(row=2, column=1, sticky="e", padx=8, pady=8)

        settings_tools = ttk.LabelFrame(params_root, text="設定檔管理", padding=8)
        settings_tools.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        ttk.Button(
            settings_tools,
            text="匯出設定檔",
            command=self.export_settings
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            settings_tools,
            text="匯入設定檔",
            command=self.import_settings
        ).pack(side="left")

        storage_tools = ttk.LabelFrame(params_root, text="儲存資料夾管理", padding=8)
        storage_tools.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        ttk.Button(
            storage_tools,
            text="清除超過保存天數的舊資料",
            command=self.cleanup_old_storage
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            storage_tools,
            text="清空儲存資料夾",
            command=self.clear_storage_folders
        ).pack(side="left")

        params_root.columnconfigure(0, weight=1)
        params_root.columnconfigure(1, weight=1)

    def _build_log_tab(self):
        self.log_text = tk.Text(self.log_tab, height=20, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.logger = AppLog(self.log_text)

    def _build_events_tab(self):
        frame = ttk.LabelFrame(self.events_tab, text="事件與結果資料夾", padding=10)
        frame.pack(fill="x")

        items = [
            ("即時事件資料夾", self._realtime_event_dir),
            ("資料夾分析事件圖", self._offline_event_dir),
            ("1111 四視窗影片", lambda: os.path.join(self.base_dir.get(), "1111")),
            (
                "11111 行為結果",
                lambda: os.path.join(self.base_dir.get(), "11111", "combined_results")
            ),
            (
                "五種事件影片資料夾",
                lambda: os.path.join(self.base_dir.get(), "11111", "fivebehavior_videos")
            ),
            (
                "HyRGB maskpicture",
                lambda: os.path.join(self.base_dir.get(), "maskpicture")
            ),
        ]

        for row, (label, path_getter) in enumerate(items):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Button(
                frame,
                text="開啟資料夾",
                command=lambda getter=path_getter: self.open_folder(getter())
            ).grid(row=row, column=1, sticky="w", padx=8, pady=4)

    def _realtime_event_dir(self):
        return os.path.join(self.base_dir.get(), "realtime_event_output", "event")

    def _offline_event_dir(self):
        output_dir = self.output_dir.get()

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.base_dir.get(), output_dir)

        return os.path.join(output_dir, "event")

    def _resolved_output_dir(self):
        output_dir = self.output_dir.get()

        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.base_dir.get(), output_dir)

        return output_dir

    def _storage_roots(self):
        base_dir = self.base_dir.get()
        roots = [
            self._resolved_output_dir(),
            os.path.join(base_dir, "realtime_event_output"),
            os.path.join(base_dir, "maskpicture"),
            os.path.join(base_dir, "1111"),
            os.path.join(base_dir, "11111"),
            os.path.join(base_dir, "logs"),
        ]

        cleaned = []
        blocked = {
            os.path.abspath(base_dir),
            os.path.abspath(self.dataset_dir.get()) if self.dataset_dir.get() else "",
            os.path.abspath(self.roi_model_dir.get()) if self.roi_model_dir.get() else "",
        }

        for path in roots:
            path = os.path.abspath(path)
            if path not in blocked and path not in cleaned:
                cleaned.append(path)

        return cleaned

    def _delete_path_contents(self, path):
        deleted = 0

        if not os.path.isdir(path):
            return deleted

        for name in os.listdir(path):
            target = os.path.join(path, name)

            try:
                if os.path.isdir(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)
                deleted += 1
            except OSError as exc:
                self._log(f"無法刪除 {target}: {exc}")

        return deleted

    def _cleanup_old_storage_files(self):
        retention_days = max(1, int(self.retention_days.get()))
        cutoff = time.time() - retention_days * 86400
        deleted = 0

        for root in self._storage_roots():
            if not os.path.isdir(root):
                continue

            for current_root, dirs, files in os.walk(root, topdown=False):
                for name in files:
                    path = os.path.join(current_root, name)

                    try:
                        if os.path.getmtime(path) < cutoff:
                            os.remove(path)
                            deleted += 1
                    except OSError as exc:
                        self._log(f"無法刪除舊檔 {path}: {exc}")

                for name in dirs:
                    path = os.path.join(current_root, name)

                    try:
                        if not os.listdir(path) and os.path.getmtime(path) < cutoff:
                            os.rmdir(path)
                            deleted += 1
                    except OSError:
                        pass

        return deleted

    def cleanup_old_storage(self):
        deleted = self._cleanup_old_storage_files()
        self._log(f"已清除超過 {self.retention_days.get()} 天的舊資料，共 {deleted} 個項目。")
        messagebox.showinfo("儲存資料夾", f"已清除舊資料，共 {deleted} 個項目。")

    def clear_storage_folders(self):
        roots = self._storage_roots()
        message = (
            "危險操作：這會清空程式輸出的事件影片、mask、效能紀錄與分析結果。\n"
            "原始圖集與模型資料夾不會被清除。\n\n"
            "將清空以下資料夾內容：\n\n"
            + "\n".join(roots)
            + "\n\n確定要繼續嗎？"
        )

        if not messagebox.askyesno("清空儲存資料夾", message):
            return

        deleted = 0
        for root in roots:
            deleted += self._delete_path_contents(root)

        self.event_items = {event: [] for event in EVENT_LABELS}
        self.event_seen.clear()
        self.event_last_accept_time.clear()
        self._clear_event_alert()
        self.refresh_event_covers()
        self._log(f"已清空儲存資料夾，共刪除 {deleted} 個項目。")
        messagebox.showinfo("儲存資料夾", f"已清空儲存資料夾，共刪除 {deleted} 個項目。")

    def open_folder(self, path):
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    def _set_event_alert(self):
        if not self.event_alert:
            self.event_alert = True
            self.tabs.tab(self.events_tab, text="事件資料夾 ●")

    def _clear_event_alert(self):
        self.event_alert = False
        self.tabs.tab(self.events_tab, text="事件資料夾")

    def _on_tab_changed(self, event):
        if self.tabs.select() == str(self.events_tab):
            self._clear_event_alert()

    def _build_events_tab(self):
        root = ttk.Frame(self.events_tab)
        root.pack(fill="both", expand=True)

        top = ttk.LabelFrame(root, text="事件分類", padding=10)
        top.pack(fill="x")

        tools = ttk.Frame(top)
        tools.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 8))
        ttk.Label(tools, text="搜尋").pack(side="left")
        search_entry = ttk.Entry(tools, textvariable=self.event_search, width=24)
        search_entry.pack(side="left", padx=(6, 10))
        search_entry.bind("<Return>", lambda _event: self.refresh_event_covers())
        ttk.Checkbutton(
            tools,
            text="只看新事件",
            variable=self.show_new_only,
            command=self.refresh_event_covers
        ).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="清除紅點",
            command=self.clear_event_red_dots
        ).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="清除封面列表",
            command=self.clear_event_covers
        ).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="開啟目前事件資料夾",
            command=self.open_selected_event_folder
        ).pack(side="left", padx=6)
        ttk.Button(
            tools,
            text="開啟全部事件資料夾",
            command=self.open_all_event_folders
        ).pack(side="left", padx=6)

        for col, (event_key, event_name) in enumerate(EVENT_LABELS.items()):
            box = ttk.Frame(top)
            box.grid(row=1, column=col, padx=6, pady=4, sticky="n")

            button = ttk.Button(
                box,
                text=event_name,
                command=lambda key=event_key: self.show_event_covers(key)
            )
            button.pack(side="left")

            alert = ttk.Label(box, text="", foreground="red")
            alert.pack(side="left", padx=(3, 0))

            self.event_buttons[event_key] = button
            self.event_button_alerts[event_key] = alert

        folders = ttk.LabelFrame(root, text="事件資料夾捷徑", padding=10)
        folders.pack(fill="x", pady=(8, 0))

        items = [
            ("即時事件資料夾", self._realtime_event_dir),
            ("資料夾分析事件圖", self._offline_event_dir),
            ("1111 四視窗影片", lambda: os.path.join(self.base_dir.get(), "1111")),
            (
                "11111 行為結果",
                lambda: os.path.join(self.base_dir.get(), "11111", "combined_results")
            ),
            (
                "五種事件影片資料夾",
                lambda: os.path.join(self.base_dir.get(), "11111", "fivebehavior_videos")
            ),
            (
                "HyRGB maskpicture",
                lambda: os.path.join(self.base_dir.get(), "maskpicture")
            ),
        ]

        for row, (label, path_getter) in enumerate(items):
            ttk.Label(folders, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Button(
                folders,
                text="開啟資料夾",
                command=lambda getter=path_getter: self.open_folder(getter())
            ).grid(row=row, column=1, sticky="w", padx=8, pady=4)

        covers = ttk.LabelFrame(root, text="事件封面", padding=10)
        covers.pack(fill="both", expand=True, pady=(8, 0))

        self.event_cover_frame = ttk.Frame(covers)
        self.event_cover_frame.pack(fill="both", expand=True)

    def _set_event_alert(self):
        if not self.event_alert:
            self.event_alert = True
            self.tabs.tab(self.events_tab, text="事件資料夾 ●")

    def _clear_event_alert(self):
        self.event_alert = False
        self.tabs.tab(self.events_tab, text="事件資料夾")

    def _on_tab_changed(self, event):
        if self.tabs.select() == str(self.events_tab):
            self._clear_event_alert()

    def refresh_event_covers(self):
        if self.selected_event is not None:
            self.show_event_covers(self.selected_event)

    def clear_event_red_dots(self):
        self.event_seen.clear()
        for event_key, items in self.event_items.items():
            for item in items:
                item["is_new"] = False
            self.event_button_alerts[event_key].configure(text="")
        self._clear_event_alert()
        self.refresh_event_covers()
        self._log("已清除事件紅點。")

    def clear_event_covers(self):
        if self.selected_event is None:
            for event_key in self.event_items:
                self.event_items[event_key].clear()
                self.event_pages[event_key] = 0
                self.event_button_alerts[event_key].configure(text="")
        else:
            self.event_items[self.selected_event].clear()
            self.event_pages[self.selected_event] = 0
            self.event_button_alerts[self.selected_event].configure(text="")

        self.refresh_event_covers()
        self._log("已清除事件封面列表。")

    def open_selected_event_folder(self):
        if self.selected_event is None:
            messagebox.showinfo("事件資料夾", "請先選擇一種事件。")
            return

        self.open_folder(self._event_folder_for(self.selected_event))

    def open_all_event_folders(self):
        for event_key in EVENT_LABELS:
            self.open_folder(self._event_folder_for(event_key))

    def _event_folder_for(self, event_key):
        event_dir = EVENT_DIR_NAMES.get(event_key, event_key.upper())
        return os.path.join(
            self.base_dir.get(),
            "11111",
            "fivebehavior_videos",
            event_dir
        )

    def _latest_event_video(self, event_key):
        folder = self._event_folder_for(event_key)

        if not os.path.isdir(folder):
            return None

        videos = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        if not videos:
            return None

        return max(videos, key=os.path.getmtime)

    def _remove_event_folder_images(self, folder):
        if not os.path.isdir(folder):
            return

        for name in os.listdir(folder):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                path = os.path.join(folder, name)

                try:
                    os.remove(path)
                except OSError as exc:
                    self._log(f"無法刪除事件資料夾圖片 {path}: {exc}")

    def _remember_event_frame(self, result):
        frame = result.get("view5_original")

        if frame is None:
            return

        self.event_frame_buffer.append(
            (
                time.time(),
                result.get("frame_idx"),
                frame.copy()
            )
        )

    def _create_event_video(self, event_key, result, event_time=None):
        image = result.get("view5_original")

        if image is None:
            return None

        folder = self._event_folder_for(event_key)
        os.makedirs(folder, exist_ok=True)
        self._remove_event_folder_images(folder)

        frame_idx = result.get("frame_idx", 0)
        filename = f"gui_event_{frame_idx}_{int(time.time() * 1000)}.mp4"
        video_path = os.path.join(folder, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 10.0
        writer = cv2.VideoWriter(video_path, fourcc, fps, (320, 240))

        event_frame_idx = result.get("frame_idx")

        if result.get("source_mode") == "offline" and event_frame_idx is not None:
            frame_radius = int(fps * 5)
            frames = [
                frame
                for _, frame_idx, frame in self.event_frame_buffer
                if (
                    frame_idx is not None
                    and event_frame_idx - frame_radius <= frame_idx <= event_frame_idx + frame_radius
                )
            ]
        else:
            if event_time is None:
                start_time = time.time() - 10
                end_time = time.time()
            else:
                start_time = event_time - 5
                end_time = event_time + 5

            frames = [
                frame
                for ts, _, frame in self.event_frame_buffer
                if start_time <= ts <= end_time
            ]

        if not frames:
            frames = [image]

        while len(frames) < int(fps * 10):
            frames.append(frames[-1])

        for frame in frames[-int(fps * 10):]:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = frame.copy()

            frame = cv2.resize(frame, (320, 240))
            writer.write(frame)

        writer.release()

        return video_path

    def _finalize_event_item(self, event_key, item, result, event_time):
        item["video_path"] = self._create_event_video(
            event_key,
            result,
            event_time=event_time
        )
        self.event_items[event_key].append(item)

        if self.selected_event == event_key:
            self.show_event_covers(event_key)

    def _handle_result_events(self, result):
        triggers = result.get("frame_triggers", {})

        for event_key, triggered in triggers.items():
            if not triggered or event_key not in EVENT_LABELS:
                continue

            now = time.time()
            last_accept = self.event_last_accept_time.get(event_key)

            if last_accept is not None and now - last_accept < 30:
                continue

            event_id = (
                event_key,
                result.get("frame_idx"),
                result.get("image_path")
            )

            if event_id in self.event_seen:
                continue

            self.event_seen.add(event_id)
            self.event_last_accept_time[event_key] = now

            self._set_event_alert()
            self.event_button_alerts[event_key].configure(text="●")

            item = {
                "event_key": event_key,
                "frame_idx": result.get("frame_idx", len(self.event_items[event_key])),
                "image": result.get("view5_original"),
                "is_new": True,
                "video_path": None,
                "folder_path": self._event_folder_for(event_key)
            }

            self.after(5000, self._finalize_event_item, event_key, item, result, now)

    def show_event_covers(self, event_key):
        self.selected_event = event_key
        self.event_button_alerts[event_key].configure(text="")

        for child in self.event_cover_frame.winfo_children():
            child.destroy()

        self.event_cover_refs.clear()

        items = list(self.event_items.get(event_key, []))
        keyword = self.event_search.get().strip().lower()

        if self.show_new_only.get():
            items = [item for item in items if item.get("is_new")]

        if keyword:
            event_name = EVENT_LABELS[event_key].lower()
            items = [
                item for item in items
                if (
                    keyword in event_name
                    or keyword in str(item.get("frame_idx", "")).lower()
                    or keyword in os.path.basename(item.get("video_path") or "").lower()
                )
            ]

        per_page = 4
        total_pages = max(1, (len(items) + per_page - 1) // per_page)
        page = min(self.event_pages.get(event_key, 0), total_pages - 1)
        self.event_pages[event_key] = page

        header = ttk.Frame(self.event_cover_frame)
        header.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))

        ttk.Label(
            header,
            text=f"{EVENT_LABELS[event_key]}事件封面"
        ).pack(side="left")

        ttk.Button(
            header,
            text="上一頁",
            command=lambda key=event_key: self.change_event_page(key, -1)
        ).pack(side="left", padx=(16, 4))

        ttk.Label(
            header,
            text=f"第 {page + 1}/{total_pages} 頁"
        ).pack(side="left", padx=4)

        ttk.Button(
            header,
            text="下一頁",
            command=lambda key=event_key: self.change_event_page(key, 1)
        ).pack(side="left", padx=4)

        if not items:
            ttk.Label(self.event_cover_frame, text="目前沒有事件封面。").grid(
                row=1,
                column=0,
                sticky="w"
            )
            return

        page_items = items[page * per_page:(page + 1) * per_page]

        for idx, item in enumerate(page_items):
            row = 1
            col = idx

            card = ttk.Frame(self.event_cover_frame, padding=6)
            card.grid(row=row, column=col, padx=6, pady=6, sticky="n")

            red_dot = "● " if item["is_new"] else ""
            ttk.Label(
                card,
                text=f"{red_dot}{EVENT_LABELS[event_key]} #{item['frame_idx']}",
                foreground="red" if item["is_new"] else "black"
            ).pack(anchor="w")

            image = item["image"]

            if image is not None:
                if len(image.shape) == 2:
                    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                pil_image = Image.fromarray(rgb)
                pil_image.thumbnail((180, 120), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                self.event_cover_refs.append(photo)

                button = ttk.Button(
                    card,
                    image=photo,
                    command=lambda data=item: self.open_event_item(data)
                )
                button.pack()

            ttk.Button(
                card,
                text="開啟影片",
                command=lambda data=item: self.open_event_item(data)
            ).pack(fill="x", pady=(4, 0))

    def change_event_page(self, event_key, delta):
        items = self.event_items.get(event_key, [])
        per_page = 4
        total_pages = max(1, (len(items) + per_page - 1) // per_page)
        next_page = self.event_pages.get(event_key, 0) + delta
        self.event_pages[event_key] = max(0, min(total_pages - 1, next_page))
        self.show_event_covers(event_key)

    def open_event_item(self, item):
        item["is_new"] = False

        video_path = item.get("video_path")
        folder_path = item.get("folder_path")

        if video_path is None or not os.path.exists(video_path):
            video_path = self._latest_event_video(item["event_key"])
            item["video_path"] = video_path

        if video_path is not None and os.path.exists(video_path):
            os.startfile(video_path)
        else:
            self.open_folder(folder_path)

        if self.selected_event == item["event_key"]:
            self.show_event_covers(item["event_key"])

    def _choose_folder(self, var):
        path = filedialog.askdirectory(initialdir=var.get() or APP_DIR)
        if path:
            var.set(path)

    def _choose_file(self, var):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(var.get()) or APP_DIR,
            filetypes=(("影像檔", "*.jpg *.jpeg *.png"), ("所有檔案", "*.*"))
        )
        if path:
            var.set(path)

    def _refresh_cameras(self):
        try:
            cameras = realtime_run.list_available_cameras()
        except Exception as exc:
            self._log(f"攝影機掃描失敗: {exc}")
            cameras = []

        values = [str(cam_id) for cam_id in cameras]
        self.camera_combo.configure(values=values)

        if values and self.camera_id.get() not in values:
            self.camera_id.set(values[0])

        self._log(f"可用攝影機: {values}")

    def _hyrgb_params(self):
        return {
            "varThreshold": self.varThreshold.get(),
            "active_ratio_threshold": self.active_ratio_threshold.get(),
            "active_indices_threshold": self.active_indices_threshold.get(),
            "active_indices_limit": self.active_indices_limit.get(),
            "learningRate": self.learningRate.get(),
            "update_interval": self.update_interval.get(),
            "batch_size": self.batch_size.get(),
            "auto_varThreshold": self.auto_varThreshold.get(),
            "varThreshold_min": self.varThreshold_min.get(),
            "varThreshold_max": self.varThreshold_max.get(),
            "varThreshold_step": self.varThreshold_step.get(),
            "active_ratio_spike": self.active_ratio_spike.get(),
            "active_ratio_drop": self.active_ratio_drop.get(),
            "active_ratio_smooth": self.active_ratio_smooth.get(),
            "foreground_threshold": self.foreground_threshold.get(),
        }

    def _start_monitor(self, name, base_dir):
        self._stop_monitor()
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"{name}_performance_log.txt")
        self.monitor = PerformanceMonitor(save_path=save_path, interval=1.0)
        self.monitor.start()
        self._log(f"效能監控已開始: {save_path}")

    def _stop_monitor(self):
        if self.monitor is not None:
            self.monitor.stop()
            self._log(f"效能監控已儲存: {self.monitor.save_path}")
            self.monitor = None

    def _set_running_state(self, running):
        state = "disabled" if running else "normal"

        if hasattr(self, "start_realtime_button"):
            self.start_realtime_button.configure(state=state)

        if hasattr(self, "start_offline_button"):
            self.start_offline_button.configure(state=state)

    def _active_result(self):
        if self.tabs.select() == str(self.offline_tab):
            return self.last_offline_result

        return self.last_realtime_result

    def _show_result(self, result, panels):
        if result is None:
            return

        for key, panel in panels.items():
            panel.set_image(result.get(key))

        if (
            self.large_panel is not None
            and self.large_window is not None
            and self.large_window.winfo_exists()
        ):
            self.large_panel.set_image(
                self._active_result().get(self.large_view_key)
                if self._active_result() is not None
                else None
            )

    def toggle_playback(self):
        self.playback_paused.set(not self.playback_paused.get())

        if not self.playback_paused.get() and self.realtime_seek_mode:
            self.jump_realtime_latest()

        status = "暫停顯示" if self.playback_paused.get() else "繼續播放"
        self._log(status)

    def restart_playback(self):
        if not self.playback_results:
            self._log("目前沒有可從頭播放的資料夾結果。")
            return

        self.playback_index = 0
        self.timeline_updating = True
        self.offline_timeline.set(0)
        self.timeline_updating = False
        self._update_offline_timeline_label()
        self.playback_paused.set(False)
        self.last_result = None
        self.last_offline_result = None
        self._log("已從頭播放資料夾結果。")

    def _start_offline_playback(self, results):
        self.playback_results = results
        self.playback_index = 0
        self.playback_paused.set(False)
        self.last_result = None
        self.last_offline_result = None
        max_index = max(0, len(results) - 1)
        self.offline_timeline_scale.configure(to=max_index)
        self.timeline_updating = True
        self.offline_timeline.set(0)
        self.timeline_updating = False
        self._update_offline_timeline_label()

    def _update_offline_timeline_label(self):
        total = len(self.playback_results)
        current = min(self.playback_index + 1, total) if total else 0
        self.offline_timeline_text.set(f"圖集回顧: {current}/{total}")

    def seek_offline_timeline(self, value):
        if self.timeline_updating:
            return

        if not self.playback_results:
            return

        index = int(round(float(value)))
        index = max(0, min(len(self.playback_results) - 1, index))
        self.playback_index = index
        self.last_result = self.playback_results[index]
        self.last_offline_result = self.last_result
        self.playback_paused.set(True)
        self._update_offline_timeline_label()

    def seek_realtime_timeline(self, value):
        if self.timeline_updating:
            return

        history = [
            (ts, result)
            for ts, result in self.realtime_history
            if time.time() - ts <= 180
        ]

        if not history:
            return

        position = max(0.0, min(180.0, float(value)))
        seconds_from_latest = 180 - position
        target_time = history[-1][0] - seconds_from_latest
        index = min(
            range(len(history)),
            key=lambda idx: abs(history[idx][0] - target_time)
        )
        self.realtime_seek_mode = position < 179.5
        self.playback_paused.set(True)
        self.last_result = history[index][1]
        self.last_realtime_result = self.last_result
        self.realtime_timeline_text.set(f"回看: -{int(round(seconds_from_latest))} 秒")

    def jump_realtime_latest(self):
        self.realtime_seek_mode = False
        self.timeline_updating = True
        self.realtime_timeline.set(180.0)
        self.timeline_updating = False
        self.realtime_timeline_text.set("即時回顧: 最新")

        if self.realtime_history:
            self.last_result = self.realtime_history[-1][1]
            self.last_realtime_result = self.last_result

        self.playback_paused.set(False)

    def open_large_view(self):
        self.large_view_key = self.large_view.get()

        if self.large_window is not None and self.large_window.winfo_exists():
            self.update_large_view_fill()
            self.large_window.lift()
            return

        self.large_window = tk.Toplevel(self)
        self.large_window.title(f"大視窗 - {self.large_view_key}")
        self.large_window.geometry("980x760")
        self.large_panel = ImagePanel(
            self.large_window,
            self.large_view_key,
            width=940,
            height=700
        )
        self.large_panel.set_fill_window(self.large_view_fill.get())
        self.large_panel.pack(fill="both", expand=True, padx=10, pady=10)

        active_result = self._active_result()

        if active_result is not None:
            self.large_panel.set_image(
                active_result.get(self.large_view_key)
            )

    def update_large_view_fill(self):
        if self.large_panel is not None:
            self.large_panel.set_fill_window(self.large_view_fill.get())

    def start_realtime(self):
        if self.realtime_thread and self.realtime_thread.is_alive():
            messagebox.showinfo("即時模式", "即時模式已經在執行中。")
            return

        if not self.camera_id.get():
            messagebox.showwarning("攝影機", "請先選擇攝影機。")
            return

        self.stop_event.clear()
        self.stopping = False
        self._set_running_state(True)
        self._cleanup_old_storage_files()
        self.output_buffer.clear()
        self.realtime_history = deque(maxlen=10000)
        self.realtime_seek_mode = False
        self.timeline_updating = True
        self.realtime_timeline.set(180.0)
        self.timeline_updating = False
        self.realtime_timeline_text.set("即時回顧: 最新")
        self.playback_results = []
        self.playback_index = 0
        self.event_frame_buffer.clear()
        self.offline_processing = False
        self.offline_progress.set(0)
        self.offline_progress_text.set("0%")
        self.last_result = None
        self.last_realtime_result = None
        self.playback_paused.set(False)
        self._start_monitor("run_realtime", self.base_dir.get())

        self.realtime_thread = threading.Thread(
            target=self._realtime_worker,
            daemon=True
        )
        self.realtime_thread.start()
        self.tabs.select(self.realtime_tab)
        self._log("即時模式已開始。")

    def _realtime_worker(self):
        cap = None

        try:
            width = 320
            height = 240
            crop_margin = self.crop_margin.get()
            display_fps = max(1.0, self.display_fps.get())
            display_delay = max(0.0, self.display_delay_seconds.get())
            max_buffer_frames = max(2, int(display_fps * display_delay) + 2)
            input_queue = queue.Queue(maxsize=max_buffer_frames)

            camera_id = int(self.camera_id.get())
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

            if not cap.isOpened():
                raise RuntimeError(f"無法開啟攝影機 {camera_id}")

            background_path = self.background_path.get()
            realtime_run.ensure_background_image(
                cap,
                background_path,
                width=width,
                height=height,
                crop_margin=crop_margin
            )

            calibration_frames = realtime_run.collect_calibration_frames(
                cap,
                count=30,
                width=width,
                height=height,
                crop_margin=crop_margin
            )

            hyrgb_system, bg_img = test_new_HyRGB.init_hyrgb_system(
                base_dir=self.base_dir.get(),
                roi_model_dir=self.roi_model_dir.get(),
                background_path=background_path,
                calibration_frames=calibration_frames,
                scene_id="camera_realtime",
                **self._hyrgb_params()
            )

            behavior_pack = finalsecond.init_behavior_system()
            result_queue = self.result_queue
            event_dir = os.path.join(self.base_dir.get(), "realtime_event_output")
            os.makedirs(event_dir, exist_ok=True)

            worker = threading.Thread(
                target=realtime_run.inference_worker,
                args=(
                    input_queue,
                    result_queue,
                    self.stop_event,
                    hyrgb_system,
                    bg_img,
                    behavior_pack,
                    self.save_mode.get(),
                    event_dir
                ),
                daemon=True
            )
            worker.start()

            frame_idx = 0
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._log("無法讀取攝影機畫面。")
                    break

                frame = realtime_run.prepare_model_frame(
                    frame,
                    width=width,
                    height=height,
                    crop_margin=crop_margin
                )

                realtime_run.put_latest(input_queue, (frame_idx, frame))
                frame_idx += 1

            self.stop_event.set()
            realtime_run.put_latest(input_queue, None)
            worker.join(timeout=2.0)

        except Exception as exc:
            self._log(f"即時模式錯誤: {exc}")

        finally:
            if cap is not None:
                cap.release()
            self._stop_monitor()
            self.stopping = False
            self.after(0, self._set_running_state, False)
            self._log("即時模式已停止。")

    def _poll_results(self):
        if not self.playback_results:
            try:
                while True:
                    result = self.result_queue.get_nowait()
                    result["source_mode"] = "realtime"
                    self.output_buffer.append(result)
                    self.realtime_history.append((time.time(), result))

                    while (
                        self.realtime_history
                        and time.time() - self.realtime_history[0][0] > 180
                    ):
                        self.realtime_history.popleft()
                    self._remember_event_frame(result)

                    if any(result.get("frame_triggers", {}).values()):
                        self._handle_result_events(result)

            except queue.Empty:
                pass

        speed = max(0.1, self.playback_speed.get())
        display_fps = max(1.0, self.display_fps.get() * speed)
        display_delay = max(0.0, self.display_delay_seconds.get())
        delay_frames = max(1, int(display_fps * display_delay))
        display_interval = 1.0 / display_fps
        now = time.perf_counter()

        if now - self.last_display_time >= display_interval:
            if not self.playback_paused.get():
                if self.playback_results:
                    if self.playback_index < len(self.playback_results):
                        self.last_result = self.playback_results[self.playback_index]
                        self.last_offline_result = self.last_result
                        self.playback_index += 1
                        self.timeline_updating = True
                        self.offline_timeline.set(max(0, self.playback_index - 1))
                        self.timeline_updating = False
                        self._update_offline_timeline_label()
                    elif self.playback_results:
                        self.last_result = self.playback_results[-1]
                        self.last_offline_result = self.last_result

                else:
                    if len(self.output_buffer) > delay_frames:
                        self.last_result = self.output_buffer.popleft()
                        self.last_realtime_result = self.last_result
                        if not self.realtime_seek_mode:
                            self.timeline_updating = True
                            self.realtime_timeline.set(180.0)
                            self.timeline_updating = False
                            self.realtime_timeline_text.set("即時回顧: 最新")
                    elif self.last_result is None and self.output_buffer:
                        self.last_result = self.output_buffer.popleft()
                        self.last_realtime_result = self.last_result
                        if not self.realtime_seek_mode:
                            self.timeline_updating = True
                            self.realtime_timeline.set(180.0)
                            self.timeline_updating = False
                            self.realtime_timeline_text.set("即時回顧: 最新")

            if self.last_result is not None:
                if self.playback_results:
                    self._show_result(self.last_offline_result, self.offline_panels)
                else:
                    self._show_result(self.last_realtime_result, self.realtime_panels)

                self._update_display_fps(now)

            self.last_display_time = now

        self.after(30, self._poll_results)

    def _update_display_fps(self, now):
        self.display_frame_count += 1
        elapsed = now - self.fps_last_time

        if elapsed >= 1.0:
            fps = self.display_frame_count / elapsed
            self.current_display_fps.set(f"顯示 FPS: {fps:.1f}")
            self.display_frame_count = 0
            self.fps_last_time = now

    def capture_background(self):
        if not self.camera_id.get():
            messagebox.showwarning("攝影機", "請先選擇攝影機。")
            return

        cap = cv2.VideoCapture(int(self.camera_id.get()), cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError("無法開啟攝影機。")

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("無法讀取攝影機畫面。")

            frame = realtime_run.prepare_model_frame(
                frame,
                width=320,
                height=240,
                crop_margin=self.crop_margin.get()
            )
            os.makedirs(os.path.dirname(self.background_path.get()), exist_ok=True)
            cv2.imwrite(self.background_path.get(), frame)
            self._log(f"背景圖已儲存: {self.background_path.get()}")

        except Exception as exc:
            self._log(f"拍攝背景失敗: {exc}")
            messagebox.showerror("背景圖", str(exc))

        finally:
            cap.release()

    def start_offline(self):
        if self.offline_thread and self.offline_thread.is_alive():
            messagebox.showinfo("資料夾模式", "資料夾分析已經在執行中。")
            return

        dataset_folder = self.dataset_dir.get()
        if not dataset_folder:
            messagebox.showwarning("資料夾模式", "請先選擇影像資料夾。")
            return

        self.stop_event.clear()
        self.stopping = False
        self._set_running_state(True)
        self._cleanup_old_storage_files()
        self.offline_progress_counts = (0, 0)
        self.output_buffer.clear()
        self.realtime_history.clear()
        self.realtime_seek_mode = False
        self.playback_results = []
        self.playback_index = 0
        self.timeline_updating = True
        self.offline_timeline.set(0)
        self.timeline_updating = False
        self.offline_timeline_scale.configure(to=0)
        self.offline_timeline_text.set("圖集回顧: 0/0")
        self.event_frame_buffer.clear()
        self.last_result = None
        self.last_offline_result = None
        self.offline_processing = True
        self.playback_paused.set(False)
        self.offline_thread = threading.Thread(
            target=self._offline_worker,
            daemon=True
        )
        self.offline_thread.start()
        self.tabs.select(self.offline_tab)
        self._log("資料夾分析已開始。")

    def _offline_worker(self):
        try:
            dataset_folder = os.path.abspath(self.dataset_dir.get())
            base_dir = os.path.dirname(dataset_folder)
            dataset_name = os.path.basename(dataset_folder)
            output_dir = self.output_dir.get()

            if not os.path.isabs(output_dir):
                output_dir = os.path.join(base_dir, output_dir)

            self.base_dir.set(base_dir)
            self._start_monitor("run_offline", base_dir)

            self.output_buffer.clear()
            self.last_result = None
            self.last_offline_result = None
            collected_results = []

            def collect_result(result):
                result["source_mode"] = "offline"
                collected_results.append(result)
                self.after(0, self._remember_event_frame, result)

                if any(result.get("frame_triggers", {}).values()):
                    self.after(0, self._handle_result_events, result)

            def update_progress(done, total):
                percent = 0 if total == 0 else (done / total) * 100
                self.offline_progress_counts = (done, total)
                self.after(
                    0,
                    lambda: (
                        self.offline_progress.set(percent),
                        self.offline_progress_text.set(f"{done}/{total}")
                    )
                )

            main_run.run_streaming_dataset(
                base_dir=base_dir,
                data_dir=dataset_name,
                roi_model_dir=self.roi_model_dir.get(),
                output_dir=output_dir,
                background_path=self.background_path.get(),
                save_mode=self.offline_save_mode.get(),
                hyrgb_params=self._hyrgb_params(),
                result_callback=collect_result,
                progress_callback=update_progress,
                stop_event=self.stop_event
            )

            if self.stop_event.is_set():
                done, total = self.offline_progress_counts
                self.after(0, self.offline_progress_text.set, f"已停止 {done}/{total}")
                self._log("資料夾分析已停止。")

            else:
                self.after(0, self.offline_progress.set, 100)
                self.after(0, self.offline_progress_text.set, "完成")
                self.after(0, self._start_offline_playback, collected_results)
                self._log("資料夾分析完成，開始播放結果。")

        except Exception as exc:
            self._log(f"資料夾模式錯誤: {exc}")

        finally:
            self.offline_processing = False
            self._stop_monitor()
            self.stopping = False
            self.after(0, self._set_running_state, False)

    def stop_current(self):
        if self.stopping:
            return

        self.stopping = True
        self.stop_event.set()
        done, total = self.offline_progress_counts

        if total:
            self.offline_progress_text.set(f"正在停止 {done}/{total}")

        self._log("正在停止，等待工作結束。")
        messagebox.showinfo("停止", "正在停止，完成目前工作後會結束。")

    def _log(self, message):
        if hasattr(self, "logger"):
            self.logger.write(message)


if __name__ == "__main__":
    app = RunApp()
    app.mainloop()
