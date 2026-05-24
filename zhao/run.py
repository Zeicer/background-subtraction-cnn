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

        self.title("HyRGB Behavior Monitor")
        self.geometry("1320x860")
        self.minsize(1100, 720)

        self.stop_event = threading.Event()
        self.realtime_thread = None
        self.offline_thread = None
        self.result_queue = queue.Queue(maxsize=8)
        self.output_buffer = deque(maxlen=8)
        self.last_result = None
        self.last_display_time = 0.0

        self.image_refs = {}

        self._build_vars()
        self._build_ui()
        self._refresh_cameras()
        self.after(30, self._poll_results)

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

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)

        self.realtime_tab = ttk.Frame(self.tabs, padding=8)
        self.offline_tab = ttk.Frame(self.tabs, padding=8)
        self.params_tab = ttk.Frame(self.tabs, padding=8)
        self.log_tab = ttk.Frame(self.tabs, padding=8)

        self.tabs.add(self.realtime_tab, text="Realtime")
        self.tabs.add(self.offline_tab, text="Offline Folder")
        self.tabs.add(self.params_tab, text="Params")
        self.tabs.add(self.log_tab, text="Log")

        self._build_realtime_tab()
        self._build_offline_tab()
        self._build_params_tab()
        self._build_log_tab()

    def _build_path_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(parent, text="Browse", command=command).grid(
            row=row,
            column=2,
            sticky="ew",
            pady=3
        )
        parent.columnconfigure(1, weight=1)

    def _build_realtime_tab(self):
        top = ttk.LabelFrame(self.realtime_tab, text="Realtime Settings", padding=8)
        top.pack(fill="x")

        self._build_path_row(
            top,
            0,
            "Base folder",
            self.base_dir,
            lambda: self._choose_folder(self.base_dir)
        )
        self._build_path_row(
            top,
            1,
            "ROI model folder",
            self.roi_model_dir,
            lambda: self._choose_folder(self.roi_model_dir)
        )
        self._build_path_row(
            top,
            2,
            "Background image",
            self.background_path,
            lambda: self._choose_file(self.background_path)
        )

        ttk.Label(top, text="Camera").grid(row=3, column=0, sticky="w", pady=3)
        self.camera_combo = ttk.Combobox(
            top,
            textvariable=self.camera_id,
            state="readonly",
            width=20
        )
        self.camera_combo.grid(row=3, column=1, sticky="w", padx=6, pady=3)
        ttk.Button(top, text="Refresh", command=self._refresh_cameras).grid(
            row=3,
            column=2,
            sticky="ew",
            pady=3
        )

        ttk.Label(top, text="Save mode").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(
            top,
            textvariable=self.save_mode,
            state="readonly",
            values=("none", "event", "all"),
            width=12
        ).grid(row=4, column=1, sticky="w", padx=6, pady=3)

        controls = ttk.Frame(top)
        controls.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Button(controls, text="Start Realtime", command=self.start_realtime).pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(controls, text="Stop", command=self.stop_current).pack(
            side="left",
            padx=6
        )
        ttk.Button(
            controls,
            text="Capture Background",
            command=self.capture_background
        ).pack(side="left", padx=6)

        images = ttk.Frame(self.realtime_tab)
        images.pack(fill="both", expand=True, pady=(8, 0))

        self.panels = {
            "view1_box_only": ImagePanel(images, "1. Boxes Only"),
            "view2_skeleton": ImagePanel(images, "2. Boxes + Skeleton"),
            "view3_mask_overlay": ImagePanel(images, "3. Mask Overlay"),
            "view4_mask": ImagePanel(images, "4. Pure Mask"),
            "view5_original": ImagePanel(images, "5. Original"),
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
        top = ttk.LabelFrame(self.offline_tab, text="Offline Folder Settings", padding=8)
        top.pack(fill="x")

        self._build_path_row(
            top,
            0,
            "Base folder",
            self.base_dir,
            lambda: self._choose_folder(self.base_dir)
        )
        self._build_path_row(
            top,
            1,
            "Dataset folder",
            self.dataset_dir,
            lambda: self._choose_folder(self.dataset_dir)
        )
        self._build_path_row(
            top,
            2,
            "ROI model folder",
            self.roi_model_dir,
            lambda: self._choose_folder(self.roi_model_dir)
        )
        self._build_path_row(
            top,
            3,
            "Output folder",
            self.output_dir,
            lambda: self._choose_folder(self.output_dir)
        )

        ttk.Label(top, text="HyRGB save mode").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Combobox(
            top,
            textvariable=self.offline_save_mode,
            state="readonly",
            values=("none", "event", "all"),
            width=12
        ).grid(row=4, column=1, sticky="w", padx=6, pady=3)

        controls = ttk.Frame(top)
        controls.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        ttk.Button(controls, text="Run Offline Folder", command=self.start_offline).pack(
            side="left",
            padx=(0, 6)
        )
        ttk.Button(controls, text="Stop", command=self.stop_current).pack(
            side="left",
            padx=6
        )

        info = ttk.Label(
            self.offline_tab,
            text=(
                "Choose a dataset image folder. The app uses its parent as base_dir "
                "and the folder name as dataset name when possible."
            ),
            wraplength=900
        )
        info.pack(anchor="w", pady=10)

    def _build_params_tab(self):
        frame = ttk.LabelFrame(self.params_tab, text="HyRGB Parameters", padding=8)
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

    def _choose_folder(self, var):
        path = filedialog.askdirectory(initialdir=var.get() or APP_DIR)
        if path:
            var.set(path)

    def _choose_file(self, var):
        path = filedialog.askopenfilename(
            initialdir=os.path.dirname(var.get()) or APP_DIR,
            filetypes=(("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if path:
            var.set(path)

    def _refresh_cameras(self):
        try:
            cameras = realtime_run.list_available_cameras()
        except Exception as exc:
            self._log(f"Camera scan failed: {exc}")
            cameras = []

        values = [str(cam_id) for cam_id in cameras]
        self.camera_combo.configure(values=values)

        if values and self.camera_id.get() not in values:
            self.camera_id.set(values[0])

        self._log(f"Available cameras: {values}")

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

    def start_realtime(self):
        if self.realtime_thread and self.realtime_thread.is_alive():
            messagebox.showinfo("Realtime", "Realtime is already running.")
            return

        if not self.camera_id.get():
            messagebox.showwarning("Camera", "Please select a camera first.")
            return

        self.stop_event.clear()
        self.output_buffer.clear()
        self.last_result = None

        self.realtime_thread = threading.Thread(
            target=self._realtime_worker,
            daemon=True
        )
        self.realtime_thread.start()
        self.tabs.select(self.realtime_tab)
        self._log("Realtime started.")

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
                raise RuntimeError(f"Cannot open camera {camera_id}")

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
                    self._log("Cannot read camera frame.")
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
            self._log(f"Realtime error: {exc}")

        finally:
            if cap is not None:
                cap.release()
            self._log("Realtime stopped.")

    def _poll_results(self):
        try:
            while True:
                self.output_buffer.append(self.result_queue.get_nowait())
        except queue.Empty:
            pass

        display_fps = max(1.0, self.display_fps.get())
        display_delay = max(0.0, self.display_delay_seconds.get())
        delay_frames = max(1, int(display_fps * display_delay))
        display_interval = 1.0 / display_fps
        now = time.perf_counter()

        if now - self.last_display_time >= display_interval:
            if len(self.output_buffer) > delay_frames:
                self.last_result = self.output_buffer.popleft()
            elif self.last_result is None and self.output_buffer:
                self.last_result = self.output_buffer.popleft()

            if self.last_result is not None:
                for key, panel in self.panels.items():
                    panel.set_image(self.last_result.get(key))

            self.last_display_time = now

        self.after(30, self._poll_results)

    def capture_background(self):
        if not self.camera_id.get():
            messagebox.showwarning("Camera", "Please select a camera first.")
            return

        cap = cv2.VideoCapture(int(self.camera_id.get()), cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera.")

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cannot read camera frame.")

            frame = realtime_run.prepare_model_frame(
                frame,
                width=320,
                height=240,
                crop_margin=self.crop_margin.get()
            )
            os.makedirs(os.path.dirname(self.background_path.get()), exist_ok=True)
            cv2.imwrite(self.background_path.get(), frame)
            self._log(f"Background saved: {self.background_path.get()}")

        except Exception as exc:
            self._log(f"Capture background failed: {exc}")
            messagebox.showerror("Background", str(exc))

        finally:
            cap.release()

    def start_offline(self):
        if self.offline_thread and self.offline_thread.is_alive():
            messagebox.showinfo("Offline", "Offline job is already running.")
            return

        dataset_folder = self.dataset_dir.get()
        if not dataset_folder:
            messagebox.showwarning("Offline", "Please choose a dataset folder.")
            return

        self.stop_event.clear()
        self.offline_thread = threading.Thread(
            target=self._offline_worker,
            daemon=True
        )
        self.offline_thread.start()
        self.tabs.select(self.realtime_tab)
        self._log("Offline job started.")

    def _offline_worker(self):
        try:
            dataset_folder = os.path.abspath(self.dataset_dir.get())
            base_dir = os.path.dirname(dataset_folder)
            dataset_name = os.path.basename(dataset_folder)
            output_dir = self.output_dir.get()

            if not os.path.isabs(output_dir):
                output_dir = os.path.join(base_dir, output_dir)

            self.base_dir.set(base_dir)

            self.output_buffer.clear()
            self.last_result = None

            main_run.run_streaming_dataset(
                base_dir=base_dir,
                data_dir=dataset_name,
                roi_model_dir=self.roi_model_dir.get(),
                output_dir=output_dir,
                save_mode=self.offline_save_mode.get(),
                hyrgb_params=self._hyrgb_params(),
                result_callback=lambda result: realtime_run.put_latest(
                    self.result_queue,
                    result
                ),
                stop_event=self.stop_event
            )
            self._log("Offline job finished.")

        except Exception as exc:
            self._log(f"Offline error: {exc}")

    def stop_current(self):
        self.stop_event.set()
        self._log("Stop requested.")

    def _log(self, message):
        if hasattr(self, "logger"):
            self.logger.write(message)


if __name__ == "__main__":
    app = RunApp()
    app.mainloop()
