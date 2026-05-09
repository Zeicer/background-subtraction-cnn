import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import glob
import re
# from backgroundsub_2 import Models  # 從上面那個檔案匯入模型

patch_size = 27
padding = 13

# 定義完整的 background subtraction 模型架構
class BackgroundSubtractorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.Conv = nn.Sequential(
            # 改成3通道輸入
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),  # 27→27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                 # 27→9

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),  # 9→9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)                  # 9→3
        )
        
        # 動態算 flatten size，避免手算錯
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 27, 27)
            x = self.Conv(dummy)           # [1,16,3,3]
            flat = x.view(1, -1).size(1)   # 16*3*3=144

        self.Classes = nn.Sequential(
            nn.Linear(flat, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, num_classes)    # 用 CrossEntropy → num_classes=2
        )
        
    def forward(self, input):
        x = self.Conv(input)
        x = x.view(x.size(0), -1)          # [N, flat]
        x = self.Classes(x)                # [N,2]
        return x
    
class PatchDataset(Dataset):
    def __init__(self, data_dir, split_name, transform=None):
        self.transform = transform
        self.data_dir = os.path.join(data_dir, split_name)
        
        # 遞迴搜尋所有 bg/fg patches
        self.bg_files = sorted(glob.glob(os.path.join(self.data_dir, "bg", "**/*.jpg"), recursive=True))
        self.fg_files = sorted(glob.glob(os.path.join(self.data_dir, "fg", "**/*.jpg"), recursive=True))
        
        # 合併並標記類別 (0=bg, 1=fg)
        self.samples = [(path, 0) for path in self.bg_files] + [(path, 1) for path in self.fg_files]
        random.shuffle(self.samples)  # 隨機打亂
        
        print(f"{split_name} 總筆數: {len(self.samples)} (bg: {len(self.bg_files)}, fg: {len(self.fg_files)})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def postprocess_mask(mask, kernel_size=5, min_area=30):
    """
    對模型輸出的二值 mask 做後處理。
    mask: 0/255 的二值圖
    """

    # 確保是 uint8
    mask = mask.astype(np.uint8)

    # 建立形態學 kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 先做 close：補人物身上的小洞，讓主體更完整
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 再做 open：去除小雜訊
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 移除太小的連通區
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask


def draw_detection_boxes(image, mask, person_area_threshold=800, object_min_area=30):
    """
    根據前景 mask 畫出偵測框。
    面積大的區域視為人物主體，小區域視為可能的移動物品。
    image: 原始彩色圖 BGR
    mask: 0/255 二值圖
    """

    result = image.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= person_area_threshold:
            label_text = "person"
            color = (0, 255, 0)
        elif area >= object_min_area:
            label_text = "object"
            color = (0, 255, 255)
        else:
            continue

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result,
            f"{label_text}:{area}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return result
    
def get_data_loader(data_dir, split_name, batch_size=32):
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    dataset = PatchDataset(data_dir, split_name, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_name == "train"),
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    return loader, dataset

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

    # 轉灰階 + ToTensor + padding
    # gray = rgb.convert("L")                              # 單通道
    img_t = TF.to_tensor(rgb)                          # [3,H,W]
    img_t = F.pad(img_t, (padding, padding, padding, padding))  # [3,H',W']

    # 切成 patch
    patches = img_t.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    Hp, Wp = patches.shape[1], patches.shape[2]
    # [1,Hp,Wp,patch,patch] → [N,3,patch,patch]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, patch_size, patch_size)
    return patches, Hp, Wp

def infer_frame(model, bg_path, input_path, device):
    size = (320, 240)

    try:
        bg_img = Image.open(bg_path)
        in_img = Image.open(input_path)
    except FileNotFoundError as e:
        print("讀取圖片失敗：", e)
        return None

    patches, Hp, Wp = make_rgb_patches_from_rgb(
        bg_img,
        in_img,
        patch_size=patch_size,
        padding=padding
    )

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, patches.size(0), 512):
            batch = patches[i:i+512].to(device)
            logits = model(batch)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(Hp, Wp)

    mask = (preds * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize(size, Image.NEAREST)

    return mask_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_environment1 = input("要讀取的環境1(大環境): ")
    # input_environment2 = input("要讀取的環境2(場景): ")
    input_picture = int(input("要讀取的圖片編號: "))
    start_time = time.perf_counter()
    input_path = fr"C:\zeicer\dataset\lagi\input\frame_{input_picture:06d}_aligned.jpg"
    bg_path = r"C:\zeicer\dataset\lagi\background.jpg"
    ckpt_dir = r"C:\zeicer\batch_pth"
    output_dir = r"C:\zeicer\dataset\lagi\maskpicture"
    output_path = os.path.join(output_dir, f"pred_mask_lagi_{input_picture}.png")

    os.makedirs(output_dir, exist_ok=True)

    # 建一個同樣的模型並載入權重
    model = BackgroundSubtractorCNN(num_classes=2).to(device)
    ckpt_list = glob.glob(os.path.join(ckpt_dir, f'best_model_step*_lagi_lagi_patch.pth'))
    assert len(ckpt_list) > 0, "找不到對應的權重檔"

    def get_step(path):
        filename = os.path.basename(path)
        match = re.search(r"best_model_step(\d+)_lagi_lagi_patch\.pth", filename)
        return int(match.group(1)) if match else -1

    ckpt_path = max(ckpt_list, key=get_step)

    print("使用權重檔：", ckpt_path)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 存成圖片看結果
    mask_img = infer_frame(model, bg_path, input_path, device)

    if mask_img is None:
        print("推論失敗，沒有產生結果圖")
        return
    
    mask_img.save(output_path)

    print(f"已輸出：{output_path}")
    end_time = time.perf_counter()
    during_time = end_time - start_time
    print(f"執行時長為 : {during_time:.6f} 秒")

    



if __name__ == "__main__":
    main()
