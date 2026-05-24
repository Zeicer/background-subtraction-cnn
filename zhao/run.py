import os
import queue
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import finalsecond
from function_monitor import PerformanceMonitor
import main_run
import realtime_run
import test_new_HyRGB


APP_DIR = os.path.dirname(os.path.abspath(__file__))


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

    def set_image(self, frame):
        if frame is None:
            return

        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        image.thumbnail((self.width, self.height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self.photo)


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
        self.playback_results = []
        self.playback_index = 0
        self.offline_processing = False
        self.last_result = None
        self.last_display_time = 0.0
        self.large_window = None
        self.large_panel = None
        self.large_view_key = None
        self.event_alert = False

        self.image_refs = {}

        self._build_vars()
        self._build_ui()
        self._refresh_cameras()
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
        self.playback_paused = tk.BooleanVar(value=False)
        self.playback_speed = tk.DoubleVar(value=1.0)
        self.large_view = tk.StringVar(value="view2_skeleton")
        self.offline_progress = tk.DoubleVar(value=0.0)
        self.offline_progress_text = tk.StringVar(value="尚未開始")

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

    def _build_realtime_tab(self):
        top = ttk.LabelFrame(self.realtime_tab, text="即時模式設定", padding=8)
        top.pack(fill="x")

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
        controls.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Button(controls, text="開始即時", command=self.start_realtime).pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(controls, text="停止", command=self.stop_current).pack(
            side="left",
            padx=6
        )
        ttk.Button(
            controls,
            text="拍攝背景",
            command=self.capture_background
        ).pack(side="left", padx=6)

        ttk.Button(
            controls,
            text="播放/暫停",
            command=self.toggle_playback
        ).pack(side="left", padx=6)

        ttk.Label(controls, text="倍速").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            controls,
            textvariable=self.playback_speed,
            state="readonly",
            values=(0.25, 0.5, 1.0, 1.5, 2.0, 4.0),
            width=6
        ).pack(side="left", padx=4)

        ttk.Label(controls, text="大視窗").pack(side="left", padx=(12, 4))
        ttk.Combobox(
            controls,
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
            controls,
            text="開啟大視窗",
            command=self.open_large_view
        ).pack(side="left", padx=6)

        ttk.Button(
            controls,
            text="從頭播放",
            command=self.restart_playback
        ).pack(side="left", padx=6)

        images = ttk.Frame(self.realtime_tab)
        images.pack(fill="both", expand=True, pady=(8, 0))

        self.panels = {
            "view1_box_only": ImagePanel(images, "1. 只顯示框線"),
            "view2_skeleton": ImagePanel(images, "2. 框線 + 骨架"),
            "view3_mask_overlay": ImagePanel(images, "3. 遮罩疊圖"),
            "view4_mask": ImagePanel(images, "4. 純遮罩"),
            "view5_original": ImagePanel(images, "5. 原始畫面"),
        }

        for idx, panel in enumerate(self.panels.values()):
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

    def _build_offline_tab(self):
        top = ttk.LabelFrame(self.offline_tab, text="資料夾模式設定", padding=8)
        top.pack(fill="x")

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
        controls.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Button(controls, text="開始資料夾分析", command=self.start_offline).pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(controls, text="停止", command=self.stop_current).pack(
            side="left",
            padx=6
        )
        ttk.Button(controls, text="從頭播放", command=self.restart_playback).pack(
            side="left",
            padx=6
        )

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

    def _build_params_tab(self):
        frame = ttk.LabelFrame(self.params_tab, text="HyRGB 參數", padding=8)
        frame.pack(fill="x")

        params = [
            ("varThreshold", self.varThreshold),
            ("active_ratio_threshold", self.active_ratio_threshold),
            ("active_indices_threshold", self.active_indices_threshold),
            ("active_indices_limit", self.active_indices_limit),
            ("learningRate", self.learningRate),
            ("update_interval", self.update_interval),
            ("batch_size", self.batch_size),
            ("foreground_threshold", self.foreground_threshold),
            ("crop_margin", self.crop_margin),
            ("display_fps", self.display_fps),
            ("display_delay_seconds", self.display_delay_seconds),
            ("varThreshold_min", self.varThreshold_min),
            ("varThreshold_max", self.varThreshold_max),
            ("varThreshold_step", self.varThreshold_step),
            ("active_ratio_spike", self.active_ratio_spike),
            ("active_ratio_drop", self.active_ratio_drop),
            ("active_ratio_smooth", self.active_ratio_smooth),
        ]

        for idx, (label, var) in enumerate(params):
            ttk.Label(frame, text=label).grid(
                row=idx,
                column=0,
                sticky="w",
                pady=3
            )
            ttk.Entry(frame, textvariable=var, width=18).grid(
                row=idx,
                column=1,
                sticky="w",
                padx=8,
                pady=3
            )

        ttk.Checkbutton(
            frame,
            text="auto_varThreshold",
            variable=self.auto_varThreshold
        ).grid(row=len(params), column=0, columnspan=2, sticky="w", pady=6)

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

    def toggle_playback(self):
        self.playback_paused.set(not self.playback_paused.get())
        status = "暫停顯示" if self.playback_paused.get() else "繼續播放"
        self._log(status)

    def restart_playback(self):
        if not self.playback_results:
            self._log("目前沒有可從頭播放的資料夾結果。")
            return

        self.playback_index = 0
        self.playback_paused.set(False)
        self.last_result = None
        self._log("已從頭播放資料夾結果。")

    def _start_offline_playback(self, results):
        self.playback_results = results
        self.playback_index = 0
        self.playback_paused.set(False)
        self.last_result = None

    def open_large_view(self):
        self.large_view_key = self.large_view.get()

        if self.large_window is not None and self.large_window.winfo_exists():
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
        self.large_panel.pack(fill="both", expand=True, padx=10, pady=10)

        if self.last_result is not None:
            self.large_panel.set_image(
                self.last_result.get(self.large_view_key)
            )

    def start_realtime(self):
        if self.realtime_thread and self.realtime_thread.is_alive():
            messagebox.showinfo("即時模式", "即時模式已經在執行中。")
            return

        if not self.camera_id.get():
            messagebox.showwarning("攝影機", "請先選擇攝影機。")
            return

        self.stop_event.clear()
        self.output_buffer.clear()
        self.playback_results = []
        self.playback_index = 0
        self.offline_processing = True
        self.offline_progress.set(0)
        self.offline_progress_text.set("0%")
        self.last_result = None
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
            self._log("即時模式已停止。")

    def _poll_results(self):
        if not self.playback_results:
            try:
                while True:
                    result = self.result_queue.get_nowait()
                    self.output_buffer.append(result)

                    if any(result.get("frame_triggers", {}).values()):
                        self._set_event_alert()

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
                        self.playback_index += 1
                    elif self.playback_results:
                        self.last_result = self.playback_results[-1]

                else:
                    if len(self.output_buffer) > delay_frames:
                        self.last_result = self.output_buffer.popleft()
                    elif self.last_result is None and self.output_buffer:
                        self.last_result = self.output_buffer.popleft()

            if self.last_result is not None:
                if any(self.last_result.get("frame_triggers", {}).values()):
                    self._set_event_alert()

                for key, panel in self.panels.items():
                    panel.set_image(self.last_result.get(key))

                if (
                    self.large_panel is not None
                    and self.large_window is not None
                    and self.large_window.winfo_exists()
                ):
                    self.large_panel.set_image(
                        self.last_result.get(self.large_view_key)
                    )

            self.last_display_time = now

        self.after(30, self._poll_results)

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
        self.output_buffer.clear()
        self.playback_results = []
        self.playback_index = 0
        self.last_result = None
        self.playback_paused.set(False)
        self.offline_thread = threading.Thread(
            target=self._offline_worker,
            daemon=True
        )
        self.offline_thread.start()
        self.tabs.select(self.realtime_tab)
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
            collected_results = []

            def collect_result(result):
                collected_results.append(result)

                if any(result.get("frame_triggers", {}).values()):
                    self.after(0, self._set_event_alert)

            def update_progress(done, total):
                percent = 0 if total == 0 else (done / total) * 100
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

    def stop_current(self):
        self.stop_event.set()
        self._log("已要求停止。")
        messagebox.showinfo("停止", "已停止。")

    def _log(self, message):
        if hasattr(self, "logger"):
            self.logger.write(message)


if __name__ == "__main__":
    app = RunApp()
    app.mainloop()
