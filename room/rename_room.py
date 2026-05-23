import os
import glob
import re

targetdir = r"C:\zeicer\dataset\room"

input_dir = os.path.join(targetdir, "new_input")
gt_dir = os.path.join(targetdir, "new_groundtruth")

START = 97

def extract_number(path):
    """
    從檔名中抓出第一段數字
    例如：
    frame_000177_aligned.jpg -> 177
    000001_pred_mask.png -> 1
    frame_000177_aligned_mask.png -> 177
    """
    basename = os.path.basename(path)
    match = re.search(r"(\d+)", basename)

    if match is None:
        return None

    return int(match.group(1))


# 讀取 input 圖片
input_paths = []
input_paths += glob.glob(os.path.join(input_dir, "*.jpg"))
input_paths += glob.glob(os.path.join(input_dir, "*.png"))
input_paths += glob.glob(os.path.join(input_dir, "*.jpeg"))

# 讀取 groundtruth 圖片
gt_paths = []
gt_paths += glob.glob(os.path.join(gt_dir, "*.jpg"))
gt_paths += glob.glob(os.path.join(gt_dir, "*.png"))
gt_paths += glob.glob(os.path.join(gt_dir, "*.jpeg"))

input_dict = {}
gt_dict = {}

# 建立 input 編號對應
for path in input_paths:
    num = extract_number(path)
    if num is not None:
        input_dict[num] = path

# 建立 groundtruth 編號對應
for path in gt_paths:
    num = extract_number(path)
    if num is not None:
        gt_dict[num] = path

# 找出兩邊都有的編號
common_nums = sorted(set(input_dict.keys()) & set(gt_dict.keys()))

print(f"input 數量: {len(input_dict)}")
print(f"groundtruth 數量: {len(gt_dict)}")
print(f"成功配對數量: {len(common_nums)}")

# 檢查沒有配對到的資料
missing_gt = sorted(set(input_dict.keys()) - set(gt_dict.keys()))
missing_input = sorted(set(gt_dict.keys()) - set(input_dict.keys()))

if missing_gt:
    print("以下 input 找不到對應 groundtruth:")
    print(missing_gt[:30])

if missing_input:
    print("以下 groundtruth 找不到對應 input:")
    print(missing_input[:30])

# 安全檢查
if len(common_nums) == 0:
    raise RuntimeError("沒有找到任何可以配對的 input / groundtruth，請檢查檔名。")


# 第一步：先改成暫存檔名，避免撞名
temp_pairs = []

for new_idx, old_num in enumerate(common_nums, start=1):
    old_input_path = input_dict[old_num]
    old_gt_path = gt_dict[old_num]

    temp_input_path = os.path.join(input_dir, f"__temp_input_{new_idx:06d}.jpg")
    temp_gt_path = os.path.join(gt_dir, f"__temp_gt_{new_idx:06d}.png")

    os.rename(old_input_path, temp_input_path)
    os.rename(old_gt_path, temp_gt_path)

    temp_pairs.append((temp_input_path, temp_gt_path))


# 第二步：改成正式檔名
for new_idx, (temp_input_path, temp_gt_path) in enumerate(temp_pairs, start=START):
    new_input_path = os.path.join(
        input_dir,
        f"frame_{new_idx:06d}_aligned.jpg"
    )

    new_gt_path = os.path.join(
        gt_dir,
        f"frame_{new_idx:06d}_aligned_mask.png"
    )

    os.rename(temp_input_path, new_input_path)
    os.rename(temp_gt_path, new_gt_path)

print("重新命名完成")