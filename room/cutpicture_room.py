import os
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
import numpy as np
import glob
from tqdm import tqdm
from multiprocessing import Pool
# from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
import joblib


# ========== 參數 ==========
patch_size = 27
padding = 13
test_min = 3
environment1 = ''
environment2 = ''
input_dir = ["input","ghz1","ghz2","room"]
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
targetdir = f"./dataset/{input_dir[3]}"
filedir = os.path.join(targetdir, "room_patch")
bgrange = pd.read_csv(f'{targetdir}/temporalROI.txt', sep=' ', header=None)
total_start_frame =  bgrange.iloc[0, 0] # 更4
total_end_frame =  bgrange.iloc[0, 1] + 1

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def process_frame_wrapper(args):
    frame_num, split_name = args
    try:
        return process_frame(frame_num, split_name)
    except AssertionError as e:
        print(f"跳過幀 {frame_num}, AssertionError: {e}")
        return None
    except Exception as e:
        print(f"跳過幀 {frame_num}, 其他錯誤: {e}")
        return None

def create_constant_image(size=(320, 240), value=128):
    # size 是 (width, height)，要轉成 (height, width) 給 numpy
    w, h = size
    img_array = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(img_array)

def make_rgb_patches_from_rgb(bg_img, in_img, patch_size=27, padding=13):
    """先疊成 RGB，再轉灰階+padding+切 patch，回傳 [N,3,patch,patch]"""
    size = (320, 240)

    # print("before resize:", bg_img.size, in_img.size)  # Debug 用
    # 統一 resize
    bg_img = bg_img.resize(size, Image.LANCZOS)
    in_img = in_img.resize(size, Image.LANCZOS)
    # bg_img = create_constant_image(size, 0) 
    # in_img = create_constant_image(size, 0)
    const_img = create_constant_image(size, 128) 

        # 統一轉成灰階 L，避免 mode mismatch
    bg_img = bg_img.convert("L")
    in_img = in_img.convert("L")
    const_img = const_img.convert("L")

    # 加一個 assert 確保真的 100% 一樣
    # assert bg_img.size == in_img.size == const_img.size, f"resize 後 still mismatch: {bg_img.size}, {in_img.size}, {const_img.size}"

    # 疊成 RGB：R=bg, G=input, B=128
    rgb = Image.merge("RGB", (bg_img, in_img, const_img))

    # 疊成 RGB → padding → 切 patch → [N,3,patch,patch]
    # gray = rgb.convert("L")                              # 單通道
    img_t = TF.to_tensor(rgb)                          # [3,H,W]
    img_t = F.pad(img_t, (padding, padding, padding, padding))  # [3,H',W']

    # 切成 patch
    patches = img_t.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    # [1,Hp,Wp,patch,patch] → [N,3,patch,patch]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size)
    return patches

def count_patches(base_dir):
    return len(glob.glob(os.path.join(base_dir, "**/*.npy"), recursive=True))

def process_frame(patchnum, split_name):
    """處理單幀，直接存到 train/valid，每幀一個 npy 包 patches 與 labels"""
    try:
        # 複製背景圖
        bg_img = bg_img_global.copy()

        # 讀取 GT 與輸入影像，用 with 自動關閉
        gt_path = os.path.join(targetdir, "groundtruth", f"frame_{patchnum:06d}_aligned_mask.png")
        in_path = os.path.join(targetdir, "input", f"frame_{patchnum:06d}_aligned.jpg")

        with Image.open(gt_path) as gt_img_file, Image.open(in_path) as in_img_file:
            gt_img = gt_img_file.copy()    # 用 copy 拿到記憶體影像，關閉原檔
            in_img = in_img_file.copy()    # 用 copy 拿到記憶體影像，關閉原檔
    except Exception as e:
        print(f"跳過幀 {patchnum}, error={e}")
        return 0, 0
    
    # === 1. 先疊再切：產生輸入 patches ===
    input_patches = make_rgb_patches_from_rgb(bg_img, in_img,
                                                patch_size=patch_size,
                                                padding=padding)   # [N,3,patch,patch]

    # === 2. GT patches，取中心像素作 label ===
    gt_img = gt_img.resize((320, 240), Image.LANCZOS)
    gt_tensor = TF.to_tensor(gt_img) # [1,H,W]

    gt_img = F.pad(gt_tensor, pad=(padding, padding, padding, padding), mode='constant', value=0)
    
    # 將中心像素作為 label
    center = patch_size // 2
    H, W = gt_img.shape[1], gt_img.shape[2]

    # 切成 patch 的起始位置
    rows = H - patch_size + 1
    cols = W - patch_size + 1

    # 建立一個陣列判斷前景/背景
    labels = []
    for i in range(rows):
        for j in range(cols):
            label = gt_img[0, i + center, j + center] > 0.3  # True = fg
            labels.append(label)
    labels = np.array(labels, dtype=np.uint8)  # shape [num_patches]

    # === 3. 儲存每幀 npy（patch + label） ===
    save_dir = os.path.join(filedir, split_name)
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{patchnum:06d}.npy")
    CHUNK_SIZE = 1000

    input_np = input_patches.numpy().astype(np.float16)

    save_dir = os.path.join(filedir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    num_patches = input_np.shape[0]

    for i in range(0, num_patches, CHUNK_SIZE):
        patches_chunk = input_np[i:i+CHUNK_SIZE]
        labels_chunk = labels[i:i+CHUNK_SIZE]

        save_file = os.path.join(save_dir, f"{patchnum:06d}_{i//CHUNK_SIZE:03d}.npy")

        np.save(save_file, {
            "patches": patches_chunk,
            "labels": labels_chunk
        })

    num_chunks = (num_patches + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"frame {patchnum} 切成 {num_chunks} 個 chunk")

    # # 目標資料夾
    # bg_dir = os.path.join(filedir, split_name, "bg")
    # fg_dir = os.path.join(filedir, split_name, "fg")

    # # 檢查這一幀是否已經處理過（只要其中一個 patch 檔存在就當作處理過）
    # sample_bg = os.path.join(bg_dir, f"bg.{patchnum}.0.jpg")
    # sample_fg = os.path.join(fg_dir, f"fg.{patchnum}.0.jpg")
    # # 決定是否覆蓋圖片
    # if os.path.exists(sample_bg) or os.path.exists(sample_fg):
    #     print(f"第 {patchnum} 幀已存在，跳過")
    #     return
    
    # os.makedirs(bg_dir, exist_ok=True)
    # os.makedirs(fg_dir, exist_ok=True)

    # # === 3. 根據 GT 是否有前景決定存到 fg/bg ===
    # input_np = input_patches.numpy().astype(np.float16)

    # bg_patches = input_np[~labels]
    # fg_patches = input_np[labels]

    # # 存 bg
    # if len(bg_patches) > 0:
    #     np.save(os.path.join(bg_dir, f"{patchnum}.npy"), bg_patches)

    # # 存 fg
    # if len(fg_patches) > 0:
    #     np.save(os.path.join(fg_dir, f"{patchnum}.npy"), fg_patches)

def build_median_background(targetdir, start_frame, end_frame, num_frames=96, size=(320, 240)):
    """
    從 temporalROI 範圍內，往前 num_frames 張 input 取中位數生成背景圖 background.jpg
    存在 targetdir/background.jpg
    """
    # 讀取 temporalROI（你前面已經有 bgrange，可改成傳進來）
    roi_start = total_start_frame   # 或 bgrange.iloc[0, 0]
    roi_end   = total_end_frame     # 或 bgrange.iloc[0, 1]

    # 從 ROI 前面往回取 num_frames 張，如果不夠就從資料一開始取
    # 這裡假設 frame index 從 1 開始，可以依你實際情況調
    first_frame = roi_start
    last_frame  = min(roi_start + num_frames, roi_end)

    frames = []
    used_ids = []

    for fid in range(first_frame, last_frame):
        # img_path = os.path.join(targetdir, input_dir[3], f"in{fid:06d}.jpg")
        img_path = os.path.join(targetdir, "input", f"frame_{fid:06d}_aligned.jpg")
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.LANCZOS)
        frames.append(np.array(img))   # [H, W, 3]
        used_ids.append(fid)

    if len(frames) == 0:
        raise RuntimeError("建立背景失敗：找不到可用的影像來做中位數")

    # 堆疊成 (N, H, W)，對時間軸 (axis=0) 做中位數
    stack = np.stack(frames, axis=0)   # [N, H, W, 3]
    median_img = np.median(stack, axis=0).astype(np.uint8)  # [H, W, 3]

    bg = Image.fromarray(median_img)
    bg.save(os.path.join(targetdir, "background.jpg"))
    print(f"背景圖已用 {len(used_ids)} 張影像的中位數生成，存到 {os.path.join(targetdir, 'background.jpg')}")

def preprocess():
        # 先用 ROI 前的 150 張建立背景圖
    build_median_background(targetdir, start_frame=bgrange.iloc[0, 0],
                            end_frame=bgrange.iloc[0, 1], num_frames=150)
    
    global bg_img_global
    bg_img_global = Image.open(os.path.join(targetdir, "background.jpg"))

    print(f"處理範圍: {total_start_frame} ~ {total_end_frame - 1}") #修改範圍 更1


    os.makedirs(os.path.join(filedir, "train"), exist_ok=True)
    os.makedirs(os.path.join(filedir, "valid"), exist_ok=True)
    print("所有 split 根目錄已建立")

    
    # 2. 收集有效幀
    true_frames = []
    for i in range(total_start_frame, total_end_frame): #修改範圍 更3
        # img_path = os.path.join(targetdir, input_dir[3], f"in{i:06d}.jpg")
        img_path = os.path.join(targetdir, "input", f"frame_{i:06d}_aligned.jpg")
        if os.path.exists(img_path) :
            true_frames.append(i)
    
    print(f"找到 {len(true_frames)} 個有效幀") 
    
    # 3. 以幀為單位隨機分割 60/20/20（修正版）
    random.shuffle(true_frames)
    n = len(true_frames)

    train_size = int(0.8 * n)
    train_frames = true_frames[:train_size]
    valid_frames = true_frames[train_size:]

    print(f"訓練幀: {len(train_frames)}, 驗證幀: {len(valid_frames)}")
    
    count = 0
    
    # 4.建立資料夾並處理
    tasks = []
    split_to_frames = {
        "train": train_frames,
        "valid": valid_frames,
    }

    for split_name, frames in split_to_frames.items():
        for frame_num in frames:
            tasks.append((frame_num, split_name))
    print(f"\n總共要處理 {len(tasks)} 個 frame 任務")

    # ✅ joblib 加速版（8 workers）
    print("使用 joblib 加速處理...")
    Parallel(n_jobs=8, backend='loky')(
        delayed(process_frame_wrapper)(task) for task in tqdm(tasks)
    )
            
    # 🔥 新增：儲存 test 幀數到 txt
    train_frames_txt = os.path.join(targetdir, "train_frames.txt")
    valid_frames_txt = os.path.join(targetdir, "valid_frames.txt")
    with open(train_frames_txt, 'w') as f:
        f.write(f"Train frames ({len(train_frames)} frames):\n")
        for frame in sorted(train_frames):  # 排序好讀
            f.write(f"{frame:06d}\n")
    with open(valid_frames_txt, 'w') as f:
        f.write(f"Valid frames ({len(valid_frames)} frames):\n")
        for frame in sorted(valid_frames):  # 排序好讀
            f.write(f"{frame:06d}\n")
    

    print("\n✅ 完成！")
    print(f"train: {count_patches(os.path.join(filedir, 'train'))}")
    print(f"valid: {count_patches(os.path.join(filedir, 'valid'))}")


if __name__ == "__main__":
    print(f"{input_dir}")
    os.makedirs(filedir, exist_ok=True)
    preprocess()
