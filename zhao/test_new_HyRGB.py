import os
import time
import glob
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from PIL import Image
from collections import OrderedDict

import torchvision.transforms.functional as TF
from torchvision.models import vgg16, VGG16_Weights


FOREGROUND_THRESHOLD = 0.9


# =========================================================
# CNN / LeNet
# =========================================================

class BackgroundSubtractorCNN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.Conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 27, 27)
            x = self.Conv(dummy)
            flat = x.view(1, -1).size(1)

        self.Classes = nn.Sequential(
            nn.Linear(flat, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, num_classes)
        )

    def forward(self, input):
        x = self.Conv(input)
        x = x.reshape(x.size(0), -1)
        x = self.Classes(x)
        return x


# =========================================================
# 全圖切 patch 用
# =========================================================

def create_constant_image(size=(320, 240), value=128):
    w, h = size
    img_array = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(img_array)


def make_rgb_patches_from_rgb(
    bg_img,
    in_img,
    patch_size=27,
    padding=13
):
    size = (320, 240)

    bg_img = bg_img.resize(size, Image.LANCZOS).convert("L")
    in_img = in_img.resize(size, Image.LANCZOS).convert("L")
    const_img = create_constant_image(size, 128).convert("L")

    rgb = Image.merge("RGB", (bg_img, in_img, const_img))

    img_t = TF.to_tensor(rgb)

    img_t = F.pad(
        img_t,
        (padding, padding, padding, padding)
    )

    patches = img_t.unfold(1, patch_size, 1).unfold(2, patch_size, 1)

    patches = patches.permute(
        1, 2, 0, 3, 4
    ).reshape(
        -1,
        3,
        patch_size,
        patch_size
    )

    return patches.contiguous()


# =========================================================
# Scene Cache
# =========================================================

class SceneCache:

    def __init__(self, max_size=10, cache_path=None):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.cache_path = None

        if cache_path is not None:
            self.set_cache_path(cache_path)

    def set_cache_path(self, cache_path):
        if self.cache_path == cache_path:
            return

        self.cache_path = cache_path
        self.cache.clear()
        self.load_from_txt()

    def load_from_txt(self):
        if (
            self.cache_path is None
            or not os.path.exists(self.cache_path)
        ):
            return

        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or "\t" not in line:
                    continue

                scene_id, channels_text = line.split("\t", 1)
                channels = []

                for value in channels_text.split(","):
                    value = value.strip()

                    if value:
                        channels.append(int(value))

                if len(channels) < 2:
                    continue

                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)

                self.cache[scene_id] = channels[:2]

        print(f"Loaded scene cache: {self.cache_path}")

    def save_to_txt(self):
        if self.cache_path is None:
            return

        os.makedirs(
            os.path.dirname(self.cache_path),
            exist_ok=True
        )

        with open(self.cache_path, "w", encoding="utf-8") as f:
            for scene_id, channels in self.cache.items():
                channels_text = ",".join(
                    str(ch) for ch in channels
                )
                f.write(f"{scene_id}\t{channels_text}\n")

    def get_channels(self, scene_id):
        if scene_id in self.cache:
            self.cache.move_to_end(scene_id)
            return self.cache[scene_id]
        return None

    def set_channels(self, scene_id, channels):
        if len(self.cache) >= self.max_size:
            oldest = next(iter(self.cache))
            print(f"快取已滿，移除舊場景資料: {oldest}")
            self.cache.popitem(last=False)

        self.cache[scene_id] = channels
        self.save_to_txt()
        print(f"已存儲場景 {scene_id} 的最佳通道: {channels}")


# =========================================================
# Main System
# =========================================================

class HybridBGSSystem:

    def __init__(
        self,
        cnn_model,
        full_lenet_model,
        vgg_extractor,
        device,
        update_interval,
        varThreshold,
        scene_cache,
        active_ratio_threshold=0.35,
        active_indices_threshold=200,
        active_indices_limit=8000,
        learningRate=0.001,
        batch_size=2048,
        auto_varThreshold=False,
        varThreshold_min=8,
        varThreshold_max=64,
        varThreshold_step=2,
        active_ratio_spike=0.20,
        active_ratio_drop=0.08,
        active_ratio_smooth=0.4,
        foreground_threshold=FOREGROUND_THRESHOLD
    ):

        self.cnn_model = cnn_model.to(device).eval()
        self.full_lenet_model = full_lenet_model.to(device).eval()

        self.vgg_extractor = vgg_extractor.to(device).eval()
        self.device = device

        self.active_ratio_threshold = active_ratio_threshold
        self.active_indices_threshold = active_indices_threshold
        self.active_indices_limit = active_indices_limit
        self.learningRate = learningRate
        self.batch_size = batch_size
        self.varThreshold = varThreshold
        self.auto_varThreshold = auto_varThreshold
        self.varThreshold_min = varThreshold_min
        self.varThreshold_max = varThreshold_max
        self.varThreshold_step = varThreshold_step
        self.active_ratio_spike = active_ratio_spike
        self.active_ratio_drop = active_ratio_drop
        self.active_ratio_smooth = active_ratio_smooth
        self.prev_active_ratio = None
        self.foreground_threshold = foreground_threshold

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=self.varThreshold,
            detectShadows=False
        )

        self.best_channels = None
        self.is_calibrated = False

        self.patch_size = 27
        self.padding = 13
        self.size = (320, 240)

        self.last_active_indices = None
        self.update_interval = update_interval
        self.frame_counter = 0

        self.scene_cache = scene_cache

        self.profile = {
            "roi_time": 0.0,
            "patch_time": 0.0,
            "cnn_time": 0.0,
            "post_time": 0.0,
            "total_time": 0.0,
            "frames": 0
        }

    # =====================================================
    # Profiling
    # =====================================================

    def print_profile(self):

        frames = self.profile["frames"]

        if frames == 0:
            return

        print("\n" + "=" * 45)
        print("📊 Infer() 平均瓶頸分析")
        print(f"統計幀數: {frames}")

        roi_avg = self.profile["roi_time"] / frames
        patch_avg = self.profile["patch_time"] / frames
        cnn_avg = self.profile["cnn_time"] / frames
        post_avg = self.profile["post_time"] / frames
        total_avg = self.profile["total_time"] / frames

        print(f"ROI / VGG + MOG2 : {roi_avg * 1000:.3f} ms")
        print(f"Patch extraction : {patch_avg * 1000:.3f} ms")
        print(f"CNN inference    : {cnn_avg * 1000:.3f} ms")
        print(f"Post process     : {post_avg * 1000:.3f} ms")
        print(f"Total infer avg  : {total_avg * 1000:.3f} ms")

        if total_avg > 0:
            print(f"Estimated FPS    : {1.0 / total_avg:.2f}")

        print("=" * 45 + "\n")

    # =====================================================
    # 校準 VGG channels
    # =====================================================

    def calibrate_vgg_channels(
        self,
        scene_id,
        calibration_frames
    ):

        cached = self.scene_cache.get_channels(scene_id)

        if cached is not None:
            self.best_channels = cached
            self.is_calibrated = True
            print(f"從快取讀取場景 [{scene_id}] 的通道: {self.best_channels}")
            return

        print(f"場景 [{scene_id}] 為新資料，開始校準...")

        channel_counts = {}

        for frame in calibration_frames:

            img_t = TF.to_tensor(
                frame.resize(self.size)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.vgg_extractor(img_t)[0]

            gray_np = np.array(
                frame.resize(self.size).convert("L")
            )

            psnr_list = []

            for ch in range(features.shape[0]):

                fmap = features[ch].cpu().numpy()

                fmap = cv2.normalize(
                    fmap,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                ).astype(np.uint8)

                score = cv2.PSNR(gray_np, fmap)
                psnr_list.append((ch, score))

            psnr_list.sort(
                key=lambda x: x[1],
                reverse=True
            )

            top_k = min(2, len(psnr_list))

            for i in range(top_k):
                idx = psnr_list[i][0]
                channel_counts[idx] = channel_counts.get(idx, 0) + 1

        if len(channel_counts) < 2:
            print("⚠️ 校準失敗或通道不足，使用預設通道 [0, 1]")
            self.best_channels = [0, 1]

        else:
            sorted_channels = sorted(
                channel_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            self.best_channels = [
                sorted_channels[0][0],
                sorted_channels[1][0]
            ]

        self.scene_cache.set_channels(
            scene_id,
            self.best_channels
        )

        self.is_calibrated = True

        print(f"✅ 校準完成: {self.best_channels}")

    # =====================================================
    # VGG + MOG2
    # =====================================================

    def adjust_varThreshold(self, active_ratio):
        if not self.auto_varThreshold:
            return

        if self.prev_active_ratio is None:
            self.prev_active_ratio = active_ratio
            return

        delta = active_ratio - self.prev_active_ratio
        new_varThreshold = self.varThreshold

        if delta > self.active_ratio_spike:
            new_varThreshold += self.varThreshold_step

        elif delta < -self.active_ratio_drop:
            new_varThreshold -= self.varThreshold_step

        new_varThreshold = max(
            self.varThreshold_min,
            min(self.varThreshold_max, new_varThreshold)
        )

        if new_varThreshold != self.varThreshold:
            self.varThreshold = new_varThreshold
            self.fgbg.setVarThreshold(self.varThreshold)
            print(f"Auto varThreshold: {self.varThreshold}")

        smooth = self.active_ratio_smooth
        self.prev_active_ratio = (
            smooth * active_ratio
            + (1.0 - smooth) * self.prev_active_ratio
        )

    def get_vgg_enhanced_frame(self, frame):

        if (
            not self.is_calibrated
            or self.best_channels is None
        ):
            return np.array(
                frame.resize(self.size).convert("L")
            )

        img_t = TF.to_tensor(
            frame.resize(self.size)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():

            features = self.vgg_extractor(img_t)[0]

            f1 = features[self.best_channels[0]].cpu().numpy()
            f2 = features[self.best_channels[1]].cpu().numpy()

        f1 = cv2.normalize(
            f1,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        f2 = cv2.normalize(
            f2,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        enhanced_frame = cv2.addWeighted(
            f1,
            0.5,
            f2,
            0.5,
            0
        )

        raw_mask = self.fgbg.apply(
            enhanced_frame,
            learningRate=self.learningRate
        )

        return raw_mask

    # =====================================================
    # ROI active patches
    # =====================================================

    def make_active_patches_vectorized(
        self,
        bg_img,
        in_img,
        active_indices,
        Hp,
        Wp
    ):

        bg_np = np.array(
            bg_img.resize(self.size).convert("L")
        )

        in_np = np.array(
            in_img.resize(self.size).convert("L")
        )

        const_np = np.full_like(bg_np, 128)

        combined = np.stack(
            [bg_np, in_np, const_np],
            axis=-1
        )

        padded = cv2.copyMakeBorder(
            combined,
            13,
            13,
            13,
            13,
            cv2.BORDER_CONSTANT,
            value=0
        )

        img_t = torch.from_numpy(
            padded
        ).permute(2, 0, 1).float() / 255.0

        rows = torch.from_numpy(active_indices // Wp)
        cols = torch.from_numpy(active_indices % Wp)

        ps = self.patch_size

        grid_y, grid_x = torch.meshgrid(
            torch.arange(ps),
            torch.arange(ps),
            indexing='ij'
        )

        idx_y = rows.view(-1, 1, 1) + grid_y.view(1, ps, ps)
        idx_x = cols.view(-1, 1, 1) + grid_x.view(1, ps, ps)

        patches = img_t[:, idx_y, idx_x].permute(1, 0, 2, 3)

        return patches.contiguous()

    # =====================================================
    # 純 LeNet 全圖推論
    # =====================================================

    def infer_full_lenet(
        self,
        bg_img,
        curr_img,
        Hp=240,
        Wp=320
    ):

        print("🚀 使用純 LeNet 全圖 patch-based 推論")

        t_start = time.perf_counter()

        patches = make_rgb_patches_from_rgb(
            bg_img,
            curr_img,
            patch_size=self.patch_size,
            padding=self.padding
        )

        preds = []
        batch_size = self.batch_size

        with torch.inference_mode():

            for i in range(0, len(patches), batch_size):

                batch = patches[i:i + batch_size].to(
                    self.device,
                    non_blocking=True
                )

                logits = self.full_lenet_model(batch)

                prob = torch.softmax(
                    logits,
                    dim=1
                )[:, 1]

                pred = (
                    prob > self.foreground_threshold
                ).long()

                preds.append(pred.cpu().numpy())

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        preds = np.concatenate(preds)

        if len(preds) != Hp * Wp:
            print("⚠️ 全圖 LeNet patch 數量錯誤")
            print("目前數量:", len(preds))
            print("應該數量:", Hp * Wp)

        final_mask = preds.reshape(Hp, Wp).astype(np.uint8) * 255

        mask_img = Image.fromarray(final_mask).resize(
            self.size,
            Image.NEAREST
        )

        t_end = time.perf_counter()

        print(
            f"✅ 全圖 LeNet 完成 | "
            f"foreground: {np.sum(preds == 1)} | "
            f"time: {t_end - t_start:.3f}s"
        )

        return mask_img, int(np.sum(preds == 1))

    # =====================================================
    # Infer
    # =====================================================

    def infer(self, bg_img, curr_img):

        infer_start = time.perf_counter()

        self.frame_counter += 1

        Hp, Wp = 240, 320
        total_pixels = Hp * Wp

        # =================================================
        # 1. ROI / VGG + MOG2
        # =================================================

        t0 = time.perf_counter()

        if (
            self.last_active_indices is None
            or self.frame_counter % self.update_interval == 0
        ):

            roi_mask = self.get_vgg_enhanced_frame(curr_img)

            small_mask = cv2.resize(
                roi_mask,
                (Wp, Hp),
                interpolation=cv2.INTER_NEAREST
            )

            # 建議用 > 200，避開 MOG2 shadow=127
            temp_active_indices = np.where(
                small_mask.flatten() > self.active_indices_threshold
            )[0]

            active_ratio = len(temp_active_indices) / total_pixels
            self.adjust_varThreshold(active_ratio)

            print(
                f"ROI active: {len(temp_active_indices)} | "
                f"ratio: {active_ratio:.4f}"
            )

            # cv2.imwrite("debug_roi_mask.png", small_mask)

            # =================================================
            # 核心：
            # active_ratio > 0.35 時，改用純 LeNet 全圖推論
            # =================================================

            if active_ratio > self.active_ratio_threshold:

                print(
                    f"⚠️ active_ratio > {self.active_ratio_threshold}, "
                    "改用純 LeNet 全圖推論"
                )

                return self.infer_full_lenet(
                    bg_img,
                    curr_img,
                    Hp,
                    Wp
                )

            else:
                self.last_active_indices = temp_active_indices

        active_indices = self.last_active_indices

        t1 = time.perf_counter()
        roi_time = t1 - t0

        if len(active_indices) == 0:

            infer_end = time.perf_counter()

            self.profile["roi_time"] += roi_time
            self.profile["total_time"] += infer_end - infer_start
            self.profile["frames"] += 1

            return Image.fromarray(
                np.zeros((Hp, Wp), dtype=np.uint8)
            ), 0

        # ROI 模式才限制 8000
        if len(active_indices) > self.active_indices_limit:
            active_indices = np.random.choice(
                active_indices,
                self.active_indices_limit,
                replace=False
            )

        # =================================================
        # 2. Patch extraction
        # =================================================

        t2 = time.perf_counter()

        selected_patches = self.make_active_patches_vectorized(
            bg_img,
            curr_img,
            active_indices,
            Hp,
            Wp
        )

        t3 = time.perf_counter()
        patch_time = t3 - t2

        # =================================================
        # 3. ROI CNN inference
        # =================================================

        t4 = time.perf_counter()

        preds = []
        batch_size = self.batch_size

        with torch.inference_mode():

            for i in range(
                0,
                len(selected_patches),
                batch_size
            ):

                batch = selected_patches[
                    i:i + batch_size
                ].to(
                    self.device,
                    non_blocking=True
                )

                logits = self.cnn_model(batch)

                prob = torch.softmax(
                    logits,
                    dim=1
                )[:, 1]

                pred = (
                    prob > self.foreground_threshold
                ).long()

                preds.append(pred.cpu().numpy())

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t5 = time.perf_counter()
        cnn_time = t5 - t4

        # =================================================
        # 4. Mask 回填
        # =================================================

        t6 = time.perf_counter()

        final_preds = np.concatenate(preds)

        print("ROI CNN foreground:", np.sum(final_preds == 1))

        full_mask_flat = np.zeros(
            total_pixels,
            dtype=np.uint8
        )

        full_mask_flat[active_indices] = final_preds * 255

        final_mask = full_mask_flat.reshape(Hp, Wp)

        mask_img = Image.fromarray(final_mask).resize(
            self.size,
            Image.NEAREST
        )

        t7 = time.perf_counter()
        post_time = t7 - t6

        infer_end = time.perf_counter()

        # =================================================
        # 5. Profiling
        # =================================================

        self.profile["roi_time"] += roi_time
        self.profile["patch_time"] += patch_time
        self.profile["cnn_time"] += cnn_time
        self.profile["post_time"] += post_time
        self.profile["total_time"] += infer_end - infer_start
        self.profile["frames"] += 1

        return mask_img, len(active_indices)


# =========================================================
# Global Cache
# =========================================================

global_scene_cache = SceneCache(max_size=10)


# =========================================================
# Main
# =========================================================

def run_hyrgb(base_dir,data_dir,roi_model_dir,input_picture=None,ghz_mask_dir = "ghz_mask1/mask",
            save_mode = "all",
            varThreshold = 24,
            active_ratio_threshold = 0.35,
            active_indices_threshold = 200,
            active_indices_limit = 8000,
            learningRate = 0.001,
            update_interval = 1,
            batch_size = 2048,
            auto_varThreshold = False,
            varThreshold_min = 8,
            varThreshold_max = 64,
            varThreshold_step = 2,
            active_ratio_spike = 0.20,
            active_ratio_drop = 0.08,
            active_ratio_smooth = 0.4,
            foreground_threshold = FOREGROUND_THRESHOLD):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("使用裝置:", device)

    num_classes = 2

    # base_dir = "./dataset/room"
    # data_dir = "pic_obalanuwalk"

    # roi_model_dir = r"C:\zeicer\room\batch_pth"

    # 這裡改成你的純 LeNet 權重
    # full_lenet_model_path = "./batch_pth/best_model_step130000_lagi_lagi_patch.pth"
    full_lenet_model_dir = roi_model_dir
    scene_cache_path = os.path.join(
        base_dir,
        "scene_channel_cache.txt"
    )
    global_scene_cache.set_cache_path(scene_cache_path)

    # ghz_mask_dir = "ghz_mask1/mask"
    # input_picture = range(0, 915)
    if input_picture is None:
        raise ValueError("input_picture 不能是 None，請從 run.py 傳入圖片路徑清單")

    mask_save_dir = os.path.join(
        base_dir,
        "maskpicture",
        ghz_mask_dir
    )

    os.makedirs(mask_save_dir, exist_ok=True)

    bg_img = Image.open(
        os.path.join(base_dir, "background.jpg")
    )

    # =====================================================
    # ROI CNN model
    # =====================================================

    cnn_model = BackgroundSubtractorCNN(num_classes)

    ckpt_list = glob.glob(
        os.path.join(roi_model_dir, "best_model_step*.pth")
    )

    if ckpt_list:
        roi_model_path = sorted(ckpt_list)[-1]

        print("載入 ROI CNN 權重:", roi_model_path)

        cnn_model.load_state_dict(
            torch.load(
                roi_model_path,
                map_location=device
            )
        )
    else:
        print("⚠️ 找不到 ROI CNN 權重，會使用未訓練模型")

    # =====================================================
    # Full LeNet model
    # =====================================================

    full_lenet_model = BackgroundSubtractorCNN(num_classes)

    full_ckpt_list = glob.glob(
        os.path.join(full_lenet_model_dir, "best_model_step*.pth")
    )
    if full_ckpt_list:
        full_lenet_model_path = sorted(full_ckpt_list)[-1]

        print("載入 LeNet 全圖權重:", full_lenet_model_path)

        full_lenet_model.load_state_dict(
            torch.load(
                full_lenet_model_path,
                map_location=device
            )
        )
    else:
        print("⚠️ 找不到 LeNet 權重檔，會使用未訓練模型")
    # if os.path.exists(full_lenet_model_path):

    #     print("載入純 LeNet 全圖權重:", full_lenet_model_path)

    #     full_lenet_model.load_state_dict(
    #         torch.load(
    #             full_lenet_model_path,
    #             map_location=device
    #         )
    #     )

    # else:
    #     raise FileNotFoundError(
    #         f"找不到純 LeNet 權重檔: {full_lenet_model_path}"
    #     )

    # =====================================================
    # VGG
    # =====================================================

    vgg16_full = vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1
    ).features.to(device)

    vgg_extractor = nn.Sequential(
        vgg16_full[0]
    ).to(device).eval()

    # =====================================================
    # System
    # =====================================================

    system = HybridBGSSystem(
        cnn_model=cnn_model,
        full_lenet_model=full_lenet_model,
        vgg_extractor=vgg_extractor,
        device=device,
        update_interval=update_interval,
        varThreshold=varThreshold,
        scene_cache=global_scene_cache,
        active_ratio_threshold=active_ratio_threshold,
        active_indices_threshold=active_indices_threshold,
        active_indices_limit=active_indices_limit,
        learningRate=learningRate,
        batch_size=batch_size,
        auto_varThreshold=auto_varThreshold,
        varThreshold_min=varThreshold_min,
        varThreshold_max=varThreshold_max,
        varThreshold_step=varThreshold_step,
        active_ratio_spike=active_ratio_spike,
        active_ratio_drop=active_ratio_drop,
        active_ratio_smooth=active_ratio_smooth,
        foreground_threshold=foreground_threshold
    )

    # =====================================================
    # Calibration
    # =====================================================

    calibration_frames = []

    cal_indices = list(input_picture)[:100]

    for img_path in cal_indices:

        path = img_path

        if os.path.exists(path):
            calibration_frames.append(
                Image.open(path)
            )

    system.calibrate_vgg_channels(
        data_dir,
        calibration_frames
    )

    calibration_frames.clear()
    del calibration_frames
    gc.collect()

    # =====================================================
    # Infer
    # =====================================================

    print("🚀 開始正式推論")

    start_time = time.perf_counter()

    model_total_time = 0.0
    processed_frames = 0

    for frame_idx,img_path in enumerate(input_picture):
        if not os.path.exists(img_path):
            continue
        curr_img = Image.open(img_path)

        frame_start = time.perf_counter()

        mask_img, active_count = system.infer(
            bg_img,
            curr_img
        )

        model_total_time += (
            time.perf_counter() - frame_start
        )

        processed_frames += 1

        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0]

        save_path = os.path.join(
            mask_save_dir,
            f"{name}_mask.png"
        )
        if save_mode == "all":
            mask_img.save(save_path)

        percentage = (
            active_count / (320 * 240)
        ) * 100

        print(
            f"Frame {frame_idx} | "
            f"Active: {active_count} "
            f"({percentage:.2f}%) | "
            f"Saved: {save_path}"
        )

        if frame_idx % 50 == 0:
            system.print_profile()

    # =====================================================
    # Stats
    # =====================================================

    end_time = time.perf_counter()

    total_sec = end_time - start_time

    print("\n" + "=" * 40)
    print(f"Processed Frames: {processed_frames}")

    if total_sec > 0:
        print(
            f"End-to-End FPS: "
            f"{processed_frames / total_sec:.2f}"
        )

    if model_total_time > 0:
        print(
            f"Model-only FPS: "
            f"{processed_frames / model_total_time:.2f}"
        )

    print(f"總耗時: {total_sec:.2f} 秒")
    print("=" * 40)

    # =====================================================
    # 將 mask 合成影片
    # =====================================================

    mask_folder = mask_save_dir

    video_path = os.path.join(
        base_dir,
        "maskpicture",
        "mask_video",
        "mask_video.mp4"
    )

    os.makedirs(
        os.path.dirname(video_path),
        exist_ok=True
    )

    mask_files = sorted(
        glob.glob(os.path.join(mask_folder, "*.png"))
    )

    if len(mask_files) > 0:

        first_frame = cv2.imread(mask_files[0])

        h, w, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            video_path,
            fourcc,
            10,
            (w, h)
        )

        for file in mask_files:

            frame = cv2.imread(file)

            if frame is None:
                continue

            out.write(frame)

        out.release()

        print(f"✅ mask 影片已輸出: {video_path}")

    else:
        print("⚠️ 沒有找到 mask 圖片")

# =========================================================
# Realtime API
# =========================================================

def init_hyrgb_system(
        base_dir,
        roi_model_dir,
        background_path=None,
        calibration_frames = None,
        scene_id = "realtime",
        update_interval = 1,
        varThreshold = 16,
        active_ratio_threshold=0.35,
        active_indices_threshold=200,
        active_indices_limit=8000,
        learningRate=0.001,
        batch_size=2048,
        auto_varThreshold=False,
        varThreshold_min=8,
        varThreshold_max=64,
        varThreshold_step=2,
        active_ratio_spike=0.20,
        active_ratio_drop=0.08,
        active_ratio_smooth=0.4,
        foreground_threshold=FOREGROUND_THRESHOLD
):
    device = torch.device("cuda"if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)
    num_classes = 2
    full_lenet_model_dir = roi_model_dir
    scene_cache_path = os.path.join(
        base_dir,
        "scene_channel_cache.txt"
    )
    global_scene_cache.set_cache_path(scene_cache_path)

    if background_path is None:
        background_path = os.path.join(
            base_dir,
            "background.jpg"
        )

    bg_img = Image.open(background_path).convert("RGB")

    cnn_model = BackgroundSubtractorCNN(num_classes)

    ckpt_list = glob.glob(
        os.path.join(
            roi_model_dir,
            "best_model_step*.pth"
        )
    )

    if ckpt_list:
        roi_model_path = sorted(ckpt_list)[-1]
        print("載入 ROI CNN 權重:", roi_model_path)

        cnn_model.load_state_dict(
            torch.load(
                roi_model_path,
                map_location=device
            )
        )
    else:
        print("⚠️ 找不到 ROI CNN 權重，會使用未訓練模型")

    full_lenet_model = BackgroundSubtractorCNN(num_classes)

    full_ckpt_list = glob.glob(
        os.path.join(
            full_lenet_model_dir,
            "best_model_step*.pth"
        )
    )

    if full_ckpt_list:
        full_lenet_model_path = sorted(full_ckpt_list)[-1]
        print("載入 LeNet 全圖權重:", full_lenet_model_path)

        full_lenet_model.load_state_dict(
            torch.load(
                full_lenet_model_path,
                map_location=device
            )
        )
    else:
        print("⚠️ 找不到 LeNet 權重檔，會使用未訓練模型")

    vgg16_full = vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1
    ).features.to(device)

    vgg_extractor = nn.Sequential(
        vgg16_full[0]
    ).to(device).eval()

    system = HybridBGSSystem(
        cnn_model=cnn_model,
        full_lenet_model=full_lenet_model,
        vgg_extractor=vgg_extractor,
        device=device,
        update_interval=update_interval,
        varThreshold=varThreshold,
        scene_cache=global_scene_cache,
        active_ratio_threshold=active_ratio_threshold,
        active_indices_threshold=active_indices_threshold,
        active_indices_limit=active_indices_limit,
        learningRate=learningRate,
        batch_size=batch_size,
        auto_varThreshold=auto_varThreshold,
        varThreshold_min=varThreshold_min,
        varThreshold_max=varThreshold_max,
        varThreshold_step=varThreshold_step,
        active_ratio_spike=active_ratio_spike,
        active_ratio_drop=active_ratio_drop,
        active_ratio_smooth=active_ratio_smooth,
        foreground_threshold=foreground_threshold
    )

    if calibration_frames is not None and len(calibration_frames) > 0:
        system.calibrate_vgg_channels(
            scene_id,
            calibration_frames
        )
    else:
        print("⚠️ 即時模式沒有校準影格，使用預設通道 [0, 1]")
        system.best_channels = [0, 1]
        system.is_calibrated = True

    return system, bg_img


def infer_one_frame(
    system,
    bg_img,
    frame
):
    if isinstance(frame, np.ndarray):
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        curr_img = Image.fromarray(
            frame_rgb
        ).convert("RGB")

    elif isinstance(frame, Image.Image):
        curr_img = frame.convert("RGB")

    else:
        raise TypeError(
            "frame 必須是 OpenCV numpy.ndarray 或 PIL.Image"
        )

    mask_img, active_count = system.infer(
        bg_img,
        curr_img
    )

    return mask_img, active_count


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    print("請使用 main_run.py 或 realtime_run.py 呼叫此模組")
