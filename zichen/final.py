import os
import cv2
import torch
import numpy as np
import glob
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# --- 設定參數 ---
PATCH_SIZE = 27
PADDING = 13
IMAGE_SIZE = (320, 240)
INFER_BATCH_SIZE = 4096

# --- 模型載入 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 載入 YOLO11-pose
pose_model = YOLO("yolo11n-pose.pt")

# 2. 載入你的 BackgroundSubtractorCNN (這裡假設類別已定義)
class BackgroundSubtractorCNN(torch.nn.Module):
    # ... (此處省略你提供的模型架構代碼) ...
    def __init__(self, num_classes=2):
        super().__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=3),
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(144, 120),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(120, num_classes)
        )
    def forward(self, x):
        x = self.Conv(x)
        x = x.view(x.size(0), -1)
        return self.Classes(x)

# 實例化並載入權重
bg_model = BackgroundSubtractorCNN().to(device)
ckpt_path = r"D:\subgarbage\patchtest" # 請指向你最新的權重
bg_model.load_state_dict(torch.load(ckpt_path, map_location=device))
bg_model.eval()

# --- 輔助函數 ---
def get_person_mask(bg_img, in_img):
    """ 利用 CNN 取得人體遮罩 """
    # 預處理：疊合背景與現有幀
    bg_l = bg_img.convert("L").resize(IMAGE_SIZE)
    in_l = in_img.convert("L").resize(IMAGE_SIZE)
    const_l = Image.new("L", IMAGE_SIZE, 128)
    
    rgb = Image.merge("RGB", (bg_l, in_l, const_l))
    img_t = TF.to_tensor(rgb)
    img_t = F.pad(img_t, (PADDING, PADDING, PADDING, PADDING))
    
    # 切 Patch 並推論
    patches = img_t.unfold(1, PATCH_SIZE, 1).unfold(2, PATCH_SIZE, 1)
    Hp, Wp = patches.shape[1], patches.shape[2]
    patches = patches.permute(1, 2, 0, 3, 4).reshape(-1, 3, PATCH_SIZE, PATCH_SIZE)
    
    preds = []
    with torch.inference_mode():
        for i in range(0, patches.size(0), INFER_BATCH_SIZE):
            batch = patches[i:i+INFER_BATCH_SIZE].to(device)
            logits = bg_model(batch)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    
    mask_np = np.concatenate(preds).reshape(Hp, Wp).astype(np.uint8) * 255
    return cv2.resize(mask_np, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

# --- 主程式迴圈 ---
def main():
    input_dir = r"D:\subgarbage\input"
    bg_path = r"D:\subgarbage\background.jpg"
    bg_img = Image.open(bg_path)
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    for path in image_paths:
        # 1. 讀取與推論 Mask
        in_img_pil = Image.open(path)
        mask = get_person_mask(bg_img, in_img_pil)
        
        # 2. 後處理 Mask (使用你寫的 postprocess_mask_for_person)
        # 這裡假設該函數已存在於你的 script 中
        clean_mask = postprocess_mask_for_person(mask, kernel_size=7) 
        
        # 3. 影像裁切：只保留人體區域
        raw_cv_img = cv2.cvtColor(np.array(in_img_pil), cv2.COLOR_RGB2BGR)
        raw_cv_img = cv2.resize(raw_cv_img, IMAGE_SIZE)
        
        person_only = cv2.bitwise_and(raw_cv_img, raw_cv_img, mask=clean_mask)
        
        # 4. YOLO Pose 偵測
        results = pose_model(person_only, verbose=False)
        
        # 5. 繪製結果 (把 YOLO 畫好的關鍵點貼回原圖，或是直接顯示裁切後的圖)
        pose_frame = results[0].plot()
        
        cv2.imshow("Optimized Pose Detection", pose_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()