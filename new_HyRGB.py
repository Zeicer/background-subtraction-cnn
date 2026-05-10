import os
import time
import glob
import gc
import torch
import torch.nn as nn
import cv2
import numpy as np

from PIL import Image
from collections import OrderedDict

import torchvision.transforms.functional as TF
from torchvision.models import vgg16, VGG16_Weights


# =========================================================
# CNN
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
# Scene Cache
# =========================================================

class SceneCache:

    def __init__(self, max_size=10):
        self.cache = OrderedDict()
        self.max_size = max_size

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
        print(f"已存儲場景 {scene_id} 的最佳通道: {channels}")


# =========================================================
# Main System
# =========================================================

class HybridBGSSystem:

    def __init__(
        self,
        cnn_model,
        vgg_extractor,
        device,
        update_interval,
        varThreshold,
        scene_cache
    ):

        self.cnn_model = cnn_model.to(device).eval()
        self.vgg_extractor = vgg_extractor.to(device).eval()
        self.device = device

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=varThreshold,
            detectShadows=True
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
        # =================================================
        # Profiling 統計
        # =================================================
        self.profile = {
            "roi_time": 0.0,
            "patch_time": 0.0,
            "cnn_time": 0.0,
            "post_time": 0.0,
            "total_time": 0.0,
            "frames": 0}

    # =====================================================
    # 印出平均耗時
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

        raw_mask = self.fgbg.apply(enhanced_frame)

        return raw_mask


    # =====================================================
    # Patch
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
    # Infer
    # =====================================================

    def infer(self, bg_img, curr_img):
        infer_start = time.perf_counter()
        self.frame_counter += 1
        Hp, Wp = 240, 320
        # =================================================
        # 1. ROI / VGG + MOG2 計時
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

            self.last_active_indices = np.where(
                small_mask.flatten() >0
            )[0]
            print("ROI active:", len(self.last_active_indices))
            cv2.imwrite("debug_roi_mask.png", small_mask)
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

        # if len(active_indices) > 15000:
        #     active_indices = active_indices[:15000]
        if len(active_indices) > 8000:
            active_indices = np.random.choice(
                active_indices,
                8000,
                replace=False
            )
        # =================================================
        # 2. Patch extraction 計時
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
        patch_time = t3-t2
        # =================================================
        # 3. CNN inference 計時
        # =================================================
        t4 = time.perf_counter()
        preds = []
        batch_size = 2048

        with torch.no_grad():

            for i in range(
                0,
                len(selected_patches),
                batch_size
            ):

                batch = selected_patches[
                    i:i+batch_size
                ].to(
                    self.device,
                    non_blocking=True
                )

                logits = self.cnn_model(batch)

                pred = torch.argmax(
                    logits,
                    dim=1
                ).cpu().numpy()

                preds.append(pred)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t5 = time.perf_counter()
        cnn_time = t5 - t4
        # =================================================
        # 4. Post process / mask 回填計時
        # =================================================
        t6 = time.perf_counter()
        final_preds = np.concatenate(preds)
        print("CNN foreground:", np.sum(final_preds == 1))
        full_mask_flat = np.zeros(
            Hp * Wp,
            dtype=np.uint8
        )

        full_mask_flat[active_indices] = final_preds * 255

        final_mask = full_mask_flat.reshape(Hp, Wp)
        mask_img = Image.fromarray(final_mask).resize(
            self.size,
            Image.NEAREST
        )
        t7 = time.perf_counter()
        post_time = t7-t6
        infer_end = time.perf_counter()
        # =================================================
        # 5. 累積 profiling 統計
        # =================================================
        self.profile["roi_time"] += roi_time
        self.profile["patch_time"] += patch_time
        self.profile["cnn_time"] += cnn_time
        self.profile["post_time"] += post_time
        self.profile["total_time"] += infer_end - infer_start
        self.profile["frames"] += 1
        return mask_img,len(active_indices)



# =========================================================
# Global Cache
# =========================================================

global_scene_cache = SceneCache(max_size=10)


# =========================================================
# Main
# =========================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    update_interval = 2
    varThreshold = 30
    num_classes = 2

    base_dir = "./dataset"
    data_dir = "input"
    batch_pth = "batch_path"
    input_picture = range(0, 255)

    os.makedirs(f"{base_dir}/maskpicture", exist_ok=True)

    bg_img = Image.open(
        os.path.join(base_dir, "background.jpg")
    )

    # =====================================================
    # CNN
    # =====================================================

    cnn_model = BackgroundSubtractorCNN(num_classes)

    ckpt_list = glob.glob(
        os.path.join(batch_pth, "best_model_step*.pth")
    )

    if ckpt_list:
        cnn_model.load_state_dict(
            torch.load(
                sorted(ckpt_list)[-1],
                map_location=device
            )
        )

    # =====================================================
    # VGG
    # =====================================================

    vgg16_full = vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1
    ).features.to(device)

    # vgg_extractor = nn.Sequential(
    #     *list(vgg16_full[:4])
    # ).to(device).eval()
    vgg_extractor = nn.Sequential(vgg16_full[0]).to(device).eval()
    # =====================================================
    # System
    # =====================================================

    system = HybridBGSSystem(
        cnn_model,
        vgg_extractor,
        device,
        update_interval,
        varThreshold,
        global_scene_cache
    )

    # =====================================================
    # 校準
    # =====================================================

    calibration_frames = []

    cal_indices = list(input_picture)[:100]

    for i in cal_indices:

        path = os.path.join(
            base_dir,
            data_dir,
            f"frame_{i:06d}_aligned.jpg"
        )

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

    model_total_time = 0
    processed_frames = 0

    for i in input_picture:

        img_path = os.path.join(
            base_dir,
            data_dir,
            f"frame_{i:06d}_aligned.jpg"
        )

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

        save_path = os.path.join(
            base_dir,
            "maskpicture",
            f"pred_mask_{i:06d}.png"
        )

        mask_img.save(save_path)

        if i % 50 == 0:

            percentage = (
                active_count / (320 * 240)
            ) * 100

            print(
                f"Frame {i} | "
                f"Active: {active_count} "
                f"({percentage:.2f}%)"
            )
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


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()