import os
import glob
import re
import shutil

folder = r"C:\zeicer\dataset\room\background_input"
output_folder = r"C:\zeicer\dataset\room\sorted_input"

os.makedirs(output_folder, exist_ok=True)

def natural_key(path):
    """
    自然排序：
    frame_1.jpg, frame_2.jpg, frame_10.jpg
    不會變成 frame_1, frame_10, frame_2
    """
    name = os.path.basename(path)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", name)]

# 讀取所有圖片
image_paths = []
image_paths += glob.glob(os.path.join(folder, "*.jpg"))
image_paths += glob.glob(os.path.join(folder, "*.png"))
image_paths += glob.glob(os.path.join(folder, "*.jpeg"))

folder2 = r"C:\zeicer\dataset\room\background_input\sec"

image_paths += glob.glob(os.path.join(folder2, "*.jpg"))
image_paths += glob.glob(os.path.join(folder2, "*.png"))
image_paths += glob.glob(os.path.join(folder2, "*.jpeg"))

# 排序
image_paths = sorted(image_paths, key=natural_key)

print(f"共找到 {len(image_paths)} 張圖片")

# 複製並重新命名到新資料夾
for idx, old_path in enumerate(image_paths, start=1):
    ext = os.path.splitext(old_path)[1].lower()

    new_name = f"frame_{idx:06d}_aligned{ext}"
    new_path = os.path.join(output_folder, new_name)

    shutil.copy2(old_path, new_path)

print("完成，已存到：", output_folder)