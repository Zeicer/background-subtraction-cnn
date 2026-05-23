import os
import glob
import re
import cv2
import numpy as np

targetdir = r"C:\zeicer\dataset\room"

input_folder = os.path.join(targetdir, "input")
gt_folder = os.path.join(targetdir, "groundtruth")

side_by_side_video_path = os.path.join(targetdir, "merged_side_by_side_preview.mp4")
overlay_video_path = os.path.join(targetdir, "merged_overlay_preview.mp4")

fps = 5
overlay_alpha = 0.4


def natural_key(path):
    name = os.path.basename(path)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", name)
    ]


def extract_number(path):
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def get_images(folder):
    paths = []
    paths += glob.glob(os.path.join(folder, "*.jpg"))
    paths += glob.glob(os.path.join(folder, "*.png"))
    paths += glob.glob(os.path.join(folder, "*.jpeg"))
    return sorted(paths, key=natural_key)


input_paths = get_images(input_folder)
gt_paths = get_images(gt_folder)

if len(input_paths) == 0:
    raise RuntimeError(f"找不到 input 圖片：{input_folder}")

if len(gt_paths) == 0:
    raise RuntimeError(f"找不到 groundtruth 圖片：{gt_folder}")

if len(input_paths) != len(gt_paths):
    raise RuntimeError("input 與 groundtruth 數量不同，無法輸出影片")

print(f"input 數量: {len(input_paths)}")
print(f"groundtruth 數量: {len(gt_paths)}")

pairs = []
for in_path, gt_path in zip(input_paths, gt_paths):
    in_num = extract_number(in_path)
    gt_num = extract_number(gt_path)

    if in_num is None:
        raise RuntimeError(f"input 檔名抓不到編號：{in_path}")
    if gt_num is None:
        raise RuntimeError(f"groundtruth 檔名抓不到編號：{gt_path}")

    if in_num != gt_num:
        raise RuntimeError(
            f"編號不一致：\n"
            f"input = {os.path.basename(in_path)}\n"
            f"gt    = {os.path.basename(gt_path)}"
        )

    pairs.append((in_path, gt_path, in_num))

first_input = cv2.imread(pairs[0][0])
if first_input is None:
    raise RuntimeError("第一張 input 圖片讀取失敗")

h, w = first_input.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# 左右並排影片
side_writer = cv2.VideoWriter(
    side_by_side_video_path,
    fourcc,
    fps,
    (w * 2, h)
)

# overlay 影片
overlay_writer = cv2.VideoWriter(
    overlay_video_path,
    fourcc,
    fps,
    (w, h)
)

for idx, (in_path, gt_path, num) in enumerate(pairs, start=1):
    input_img = cv2.imread(in_path)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if input_img is None:
        print("input 讀取失敗，跳過：", in_path)
        continue

    if gt_img is None:
        print("groundtruth 讀取失敗，跳過：", gt_path)
        continue

    input_img = cv2.resize(input_img, (w, h))
    gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # =========================
    # 1. 左右並排版
    # =========================
    input_show = input_img.copy()
    gt_bgr = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        input_show,
        f"INPUT {num:06d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        gt_bgr,
        f"GT {num:06d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    side_by_side = np.hstack((input_show, gt_bgr))
    cv2.line(side_by_side, (w, 0), (w, h), (0, 255, 255), 2)

    side_writer.write(side_by_side)

    # =========================
    # 2. Overlay 版
    # =========================
    overlay = input_img.copy()

    # 紅色圖層
    red_layer = input_img.copy()
    red_layer[:, :, 0] = 0
    red_layer[:, :, 1] = 0
    red_layer[:, :, 2] = 255

    # 整張先混合
    blended = cv2.addWeighted(
        input_img,
        1 - overlay_alpha,
        red_layer,
        overlay_alpha,
        0
    )

    # 只把 GT 白色區域套上去
    mask_binary = gt_img > 0
    overlay[mask_binary] = blended[mask_binary]

    cv2.putText(
        overlay,
        f"OVERLAY {num:06d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    overlay_writer.write(overlay)

side_writer.release()
overlay_writer.release()

print("輸出完成：")
print("左右並排影片：", side_by_side_video_path)
print("Overlay 影片：", overlay_video_path)