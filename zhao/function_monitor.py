import os
import time
import threading
import psutil
from datetime import datetime


class PerformanceMonitor:
    def __init__(self, save_path="performance_log.txt", interval=1.0):
        self.save_path = save_path
        self.interval = interval
        self.running = False
        self.thread = None
        self.records = []

        self.process = psutil.Process(os.getpid())

        self.start_time = None
        self.end_time = None

        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()

    def _get_gpu_info(self):
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()

            if len(gpus) == 0:
                return "GPU: 無偵測到 GPU"

            info = []
            for gpu in gpus:
                info.append(
                    f"GPU {gpu.id}: "
                    f"{gpu.name}, "
                    f"Load={gpu.load * 100:.1f}%, "
                    f"Mem={gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f}MB"
                )

            return " | ".join(info)

        except Exception:
            return "GPU: 無法讀取，請確認是否安裝 GPUtil"

    def _collect(self):
        while self.running:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cpu_total = psutil.cpu_percent(interval=None)
            cpu_process = self.process.cpu_percent(interval=None)

            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info().rss / 1024 / 1024

            disk = psutil.disk_io_counters()
            net = psutil.net_io_counters()

            disk_read_mb = disk.read_bytes / 1024 / 1024
            disk_write_mb = disk.write_bytes / 1024 / 1024

            net_sent_mb = net.bytes_sent / 1024 / 1024
            net_recv_mb = net.bytes_recv / 1024 / 1024

            gpu_info = self._get_gpu_info()

            line = (
                f"[{now}] "
                f"CPU Total={cpu_total:.1f}% | "
                f"CPU Process={cpu_process:.1f}% | "
                f"RAM Total={memory.percent:.1f}% | "
                f"RAM Process={process_memory:.1f}MB | "
                f"Disk Read={disk_read_mb:.1f}MB | "
                f"Disk Write={disk_write_mb:.1f}MB | "
                f"Net Sent={net_sent_mb:.1f}MB | "
                f"Net Recv={net_recv_mb:.1f}MB | "
                f"{gpu_info}"
            )

            print(line)
            self.records.append(line)

            time.sleep(self.interval)

    def start(self):
        self.start_time = time.time()
        self.running = True

        self.thread = threading.Thread(
            target=self._collect,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        self.end_time = time.time()
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=2)

        self.save()

    def save(self):
        total_time = 0

        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time

        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write("效能監控紀錄\n")
            f.write("=" * 60 + "\n")
            f.write(f"開始時間: {datetime.fromtimestamp(self.start_time)}\n")
            f.write(f"結束時間: {datetime.fromtimestamp(self.end_time)}\n")
            f.write(f"總執行時間: {total_time:.2f} 秒\n")
            f.write("=" * 60 + "\n\n")

            for line in self.records:
                f.write(line + "\n")

        print(f"\n效能紀錄已儲存到: {self.save_path}")