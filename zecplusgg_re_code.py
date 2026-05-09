import cv2
import numpy as np
import os
import glob
import time


# =========================================================
# 使用者設定區
# =========================================================

# 你的裁切後影格資料夾
input_dir = r"C:\video2pic\color2black\input\pic_wolk2"

# 輸出資料夾
output_root = r"C:\video2pic\color2black\pseudo_gt_output"

aligned_dir = os.path.join(output_root, "aligned_frames")
diff_dir = os.path.join(output_root, "diff_images")
mask_dir = os.path.join(output_root, "pseudo_masks")
background_path = os.path.join(output_root, "background_median.jpg")

# 支援的圖片格式
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

# 對齊設定
motion_mode = cv2.MOTION_AFFINE
ecc_max_iter = 50
ecc_eps = 1e-5

# 建立背景時，每隔幾張取一張
background_sample_step = 5

# 背景相減 threshold
diff_threshold = 30

# 形態學 kernel 大小
kernel_size = 3

# 移除太小的前景區塊
min_area = 50

# 邊界忽略範圍，避免對齊後黑邊被判斷成前景
border_ignore = 10

# 是否額外輸出 diff image
save_diff = True


# =========================================================
# 工具函式
# =========================================================

def make_dirs():
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)


def get_image_paths(input_dir):
    paths = []
    for ext in image_exts:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    paths = sorted(paths)

    if len(paths) == 0:
        raise FileNotFoundError(f"找不到影格圖片：{input_dir}")

    return paths


def read_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"圖片讀取失敗：{path}")

    return img


def remove_small_components(mask, min_area=50):
    """
    移除太小的白色連通區。
    mask: 0/255 二值圖
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask


def ignore_border(mask, border=10):
    """
    將邊界區域設成黑色，避免對齊產生的黑邊被當成前景。
    """
    if border <= 0:
        return mask

    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0

    return mask


# =========================================================
# Step 1：影像對齊 / 穩定化
# =========================================================

def align_frames(paths):
    print("\n[Step 1] 開始影像對齊 / 穩定化")

    # 使用第一張影格作為基準
    ref = read_image(paths[0])
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    height, width = ref_gray.shape

    # 先儲存第一張基準圖
    first_name = os.path.splitext(os.path.basename(paths[0]))[0]
    cv2.imwrite(os.path.join(aligned_dir, f"{first_name}_aligned.jpg"), ref)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ecc_max_iter,
        ecc_eps
    )

    aligned_paths = []

    first_aligned_path = os.path.join(aligned_dir, f"{first_name}_aligned.jpg")
    aligned_paths.append(first_aligned_path)

    fail_count = 0

    for i, path in enumerate(paths[1:], start=1):
        img = read_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ECC 需要目前影像尺寸和基準影像一致
        if gray.shape != ref_gray.shape:
            img = cv2.resize(img, (width, height))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 仿射矩陣：可處理平移、旋轉、縮放、些微變形
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_gray,
                gray,
                warp_matrix,
                motion_mode,
                criteria
            )

            aligned = cv2.warpAffine(
                img,
                warp_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        except cv2.error:
            # 對齊失敗就保留原圖
            fail_count += 1
            aligned = img

        name = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(aligned_dir, f"{name}_aligned.jpg")
        cv2.imwrite(save_path, aligned)
        aligned_paths.append(save_path)

        if (i + 1) % 50 == 0:
            print(f"   對齊進度：{i + 1}/{len(paths)}")

    print(f"影像對齊完成，共 {len(paths)} 張")
    print(f"對齊失敗張數：{fail_count}")

    return aligned_paths


# =========================================================
# Step 2：建立 Median Background
# =========================================================

def build_median_background(aligned_paths):
    print("\n[Step 2] 開始建立 median background")

    sample_paths = aligned_paths[::background_sample_step]

    if len(sample_paths) == 0:
        raise RuntimeError("沒有可用影格建立背景")

    frames = []

    for i, path in enumerate(sample_paths):
        img = read_image(path)
        frames.append(img)

        if (i + 1) % 50 == 0:
            print(f"   背景取樣進度：{i + 1}/{len(sample_paths)}")

    frames_np = np.stack(frames, axis=0)
    median_bg = np.median(frames_np, axis=0).astype(np.uint8)

    cv2.imwrite(background_path, median_bg)

    print(f"median background 建立完成：{background_path}")

    return median_bg


# =========================================================
# Step 3：背景相減產生 pseudo mask
# =========================================================

def generate_pseudo_masks(aligned_paths, median_bg):
    print("\n[Step 3] 開始產生 pseudo masks")

    bg_gray = cv2.cvtColor(median_bg, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i, path in enumerate(aligned_paths):
        img = read_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 如果尺寸不一致，調整成背景圖尺寸
        if gray.shape != bg_gray.shape:
            gray = cv2.resize(gray, (bg_gray.shape[1], bg_gray.shape[0]))

        # 背景相減
        diff = cv2.absdiff(gray, bg_gray)

        # 二值化
        _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 去雜訊
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 補洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 移除小區塊
        mask = remove_small_components(mask, min_area=min_area)

        # 忽略邊界
        mask = ignore_border(mask, border=border_ignore)

        name = os.path.splitext(os.path.basename(path))[0]

        if save_diff:
            diff_save_path = os.path.join(diff_dir, f"{name}_diff.jpg")
            cv2.imwrite(diff_save_path, diff)

        mask_save_path = os.path.join(mask_dir, f"{name}_mask.png")
        cv2.imwrite(mask_save_path, mask)

        if (i + 1) % 50 == 0:
            print(f"   mask 產生進度：{i + 1}/{len(aligned_paths)}")

    print(f"pseudo masks 產生完成：{mask_dir}")


# =========================================================
# 主程式
# =========================================================

def main():
    start_time = time.perf_counter()

    make_dirs()

    print("======================================")
    print("Pseudo Ground Truth 產生流程")
    print("======================================")
    print(f"輸入影格資料夾：{input_dir}")
    print(f"輸出資料夾：{output_root}")

    paths = get_image_paths(input_dir)
    print(f"讀取到影格數量：{len(paths)}")

    aligned_paths = align_frames(paths)

    median_bg = build_median_background(aligned_paths)

    generate_pseudo_masks(aligned_paths, median_bg)

    end_time = time.perf_counter()

    print("\n======================================")
    print("全部處理完成")
    print("======================================")
    print(f"總耗時：{end_time - start_time:.2f} 秒")
    print(f"對齊後影格：{aligned_dir}")
    print(f"背景圖：{background_path}")
    print(f"差異圖：{diff_dir}")
    print(f"pseudo mask：{mask_dir}")


if __name__ == "__main__":
    main()