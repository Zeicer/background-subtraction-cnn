import os
import glob
import re
import cv2
import numpy as np

input_folder = r"C:\zeicer\dataset\room\sorted_input"
gt_folder = r"C:\zeicer\dataset\room\sorted_groundtruth"

os.makedirs(gt_folder, exist_ok=True)


def natural_key(path):
    name = os.path.basename(path)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


# 讀取 sorted_input 裡的所有圖片
image_paths = []
image_paths += glob.glob(os.path.join(input_folder, "*.jpg"))
image_paths += glob.glob(os.path.join(input_folder, "*.png"))
image_paths += glob.glob(os.path.join(input_folder, "*.jpeg"))

image_paths = sorted(image_paths, key=natural_key)

print(f"sorted_input 圖片數量：{len(image_paths)}")

for idx, img_path in enumerate(image_paths, start=1):
    img = cv2.imread(img_path)

    if img is None:
        print("讀取失敗，跳過：", img_path)
        continue

    h, w = img.shape[:2]

    # 建立同尺寸全黑圖片
    black_gt = np.zeros((h, w), dtype=np.uint8)

    gt_name = f"frame_{idx:06d}_aligned_mask.png"
    gt_path = os.path.join(gt_folder, gt_name)

    cv2.imwrite(gt_path, black_gt)

print("全黑 groundtruth 建立完成：", gt_folder)