import os
import glob
import re
import shutil

targetdir = r"C:\zeicer\dataset\room"

sorted_input_dir = os.path.join(targetdir, "sorted_input")
sorted_gt_dir = os.path.join(targetdir, "sorted_groundtruth")

old_input_dir = os.path.join(targetdir, "input")
old_gt_dir = os.path.join(targetdir, "groundtruth")

merged_input_dir = os.path.join(targetdir, "merged_input")
merged_gt_dir = os.path.join(targetdir, "merged_groundtruth")

os.makedirs(merged_input_dir, exist_ok=True)
os.makedirs(merged_gt_dir, exist_ok=True)


def get_images(folder):
    paths = []
    paths += glob.glob(os.path.join(folder, "*.jpg"))
    paths += glob.glob(os.path.join(folder, "*.png"))
    paths += glob.glob(os.path.join(folder, "*.jpeg"))
    return paths


def extract_number(path):
    """
    從檔名抓數字：
    frame_000001_aligned.jpg -> 1
    frame_000001_aligned_mask.png -> 1
    000001_pred_mask.png -> 1
    """
    name = os.path.basename(path)
    match = re.search(r"(\d+)", name)

    if match is None:
        return None

    return int(match.group(1))


def build_number_dict(paths, group_name):
    """
    建立 編號 -> 檔案路徑
    如果有檔名抓不到數字，或重複編號，就停止
    """
    result = {}

    for path in paths:
        num = extract_number(path)

        if num is None:
            raise RuntimeError(
                f"{group_name} 有檔案抓不到編號：\n{path}"
            )

        if num in result:
            raise RuntimeError(
                f"{group_name} 有重複編號 {num:06d}：\n"
                f"第一個：{result[num]}\n"
                f"第二個：{path}"
            )

        result[num] = path

    return result


def validate_by_number(input_dir, gt_dir, group_name):
    """
    完全依編號檢查 input 和 groundtruth 是否一一對應
    """
    input_paths = get_images(input_dir)
    gt_paths = get_images(gt_dir)

    input_dict = build_number_dict(input_paths, f"{group_name} input")
    gt_dict = build_number_dict(gt_paths, f"{group_name} groundtruth")

    input_nums = set(input_dict.keys())
    gt_nums = set(gt_dict.keys())

    missing_gt = sorted(input_nums - gt_nums)
    missing_input = sorted(gt_nums - input_nums)

    if missing_gt:
        raise RuntimeError(
            f"{group_name} 有 input 找不到對應 groundtruth：\n"
            f"{missing_gt[:50]}"
        )

    if missing_input:
        raise RuntimeError(
            f"{group_name} 有 groundtruth 找不到對應 input：\n"
            f"{missing_input[:50]}"
        )

    common_nums = sorted(input_nums)

    pairs = []

    for num in common_nums:
        pairs.append((input_dict[num], gt_dict[num]))

    print(f"{group_name} 檢查通過，共 {len(pairs)} 組配對")

    return pairs


# 先檢查 sorted 資料
sorted_pairs = validate_by_number(
    sorted_input_dir,
    sorted_gt_dir,
    "sorted dataset"
)

# 再檢查原本資料
old_pairs = validate_by_number(
    old_input_dir,
    old_gt_dir,
    "old dataset"
)

print("全部編號檢查通過，開始合併...")


new_idx = 1

# 先放 sorted_input / sorted_groundtruth
for in_path, gt_path in sorted_pairs:
    new_input_name = f"frame_{new_idx:06d}_aligned.jpg"
    new_gt_name = f"frame_{new_idx:06d}_aligned_mask.png"

    new_input_path = os.path.join(merged_input_dir, new_input_name)
    new_gt_path = os.path.join(merged_gt_dir, new_gt_name)

    shutil.copy2(in_path, new_input_path)
    shutil.copy2(gt_path, new_gt_path)

    new_idx += 1


# 再放原本 input / groundtruth
for in_path, gt_path in old_pairs:
    new_input_name = f"frame_{new_idx:06d}_aligned.jpg"
    new_gt_name = f"frame_{new_idx:06d}_aligned_mask.png"

    new_input_path = os.path.join(merged_input_dir, new_input_name)
    new_gt_path = os.path.join(merged_gt_dir, new_gt_name)

    shutil.copy2(in_path, new_input_path)
    shutil.copy2(gt_path, new_gt_path)

    new_idx += 1


print("合併完成")
print(f"總共輸出：{new_idx - 1} 組資料")
print("merged_input：", merged_input_dir)
print("merged_groundtruth：", merged_gt_dir)