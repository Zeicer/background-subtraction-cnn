import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import re
# from backgroundsub_2 import Models  # 從上面那個檔案匯入模型

PATCH_SIZE = 27
PADDING = 13
IMAGE_SIZE = (320, 240)

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

def postprocess_mask(mask, kernel_size, min_area, border_ignore):
    """
    保守後處理版本：
    目標是去除小雜訊與邊界誤判，不要過度膨脹，避免 person 和 object 黏在一起。
    """

    mask = mask.astype(np.uint8)

    if mask.max() == 1:
        mask = mask * 255

    # 清除邊界，避免對齊偏移造成誤判
    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    # 用小 kernel，不要用太大的 7、9、10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 輕微 close，補很小的洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 輕微 open，去除小雜訊
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 再清一次邊界
    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    # 移除太小連通區
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask

def postprocess_mask_for_person(mask, kernel_size=7, min_area=100, border_ignore=20):
    """
    人物專用後處理：
    目標是讓人物碎片更容易連成完整區域。
    這張 mask 只拿來抓 person，不拿來抓 object。
    """

    mask = mask.astype(np.uint8)

    if mask.max() == 1:
        mask = mask * 255

    # 清除邊界
    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 人物可以稍微膨脹，讓頭、身體、手腳比較容易接起來
    mask = cv2.dilate(mask, kernel, iterations=1)

    # close 補洞與連接斷裂區
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 不要用太強的 open，避免人物又被切碎
    small_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    # 再清一次邊界
    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    # 移除太小區塊
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask

def box_inside_or_overlap(box_a, box_b, overlap_threshold=0.3):
    """
    判斷 box_a 是否和 box_b 重疊太多。
    box 格式: (x, y, w, h)
    """

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    area_a = aw * ah

    if area_a == 0:
        return False

    overlap_ratio = inter_area / area_a

    return overlap_ratio >= overlap_threshold

def draw_detection_boxes_dual_mask(
    image,
    person_mask,
    object_mask,
    person_area_threshold=700,
    object_min_area=15,
    object_max_area=450,
    border_margin=15,
    person_min_height=80,
    person_min_aspect_ratio=1.4
):
    """
    使用兩張 mask 畫框：
    person_mask 負責人物
    object_mask 負責物品
    """

    result = image.copy()
    img_h, img_w = person_mask.shape[:2]

    person_boxes = []
    object_boxes = []

    # ========== 1. 從 person_mask 找人物 ==========
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(person_mask, connectivity=8)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        touches_border = (
            x <= border_margin or
            y <= border_margin or
            x + w >= img_w - border_margin or
            y + h >= img_h - border_margin
        )

        if touches_border:
            continue

        aspect_ratio = h / max(w, 1)

        is_person_shape = (
            area >= person_area_threshold and
            h >= person_min_height and
            aspect_ratio >= person_min_aspect_ratio
        )

        if not is_person_shape:
            continue

        person_boxes.append((x, y, w, h, area))

    # ========== 2. 從 object_mask 找物品 ==========
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(object_mask, connectivity=8)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < object_min_area or area > object_max_area:
            continue

        touches_border = (
            x <= border_margin or
            y <= border_margin or
            x + w >= img_w - border_margin or
            y + h >= img_h - border_margin
        )

        if touches_border:
            continue

        object_box = (x, y, w, h)

        # 如果 object 和 person 重疊太多，代表它可能只是人物碎片，不畫成 object
        overlap_with_person = False

        for px, py, pw, ph, _ in person_boxes:
            person_box = (px, py, pw, ph)

            if box_inside_or_overlap(object_box, person_box, overlap_threshold=0.8):
                overlap_with_person = True
                break

        if not overlap_with_person:
            object_boxes.append((x, y, w, h, area))

    # ========== 3. 畫人物框 ==========
    for x, y, w, h, area in person_boxes:
        color = (0, 255, 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result,
            f"person:{area}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    # ========== 4. 畫物品框 ==========
    for x, y, w, h, area in object_boxes:
        color = (0, 255, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            result,
            f"object:{area}",
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )

    return result

def merge_person_boxes(boxes, merge_distance=35):
    """
    只合併人物候選框。
    boxes 格式: [(x, y, w, h, area), ...]
    """

    merged = []

    for box in boxes:
        x, y, w, h, area = box
        x1, y1, x2, y2 = x, y, x + w, y + h

        has_merged = False

        for idx, old_box in enumerate(merged):
            ox, oy, ow, oh, oarea = old_box
            ox1, oy1, ox2, oy2 = ox, oy, ox + ow, oy + oh

            # 只允許人物碎片彼此靠近時合併
            expanded_ox1 = ox1 - merge_distance
            expanded_oy1 = oy1 - merge_distance
            expanded_ox2 = ox2 + merge_distance
            expanded_oy2 = oy2 + merge_distance

            overlap_or_close = not (
                x2 < expanded_ox1 or
                x1 > expanded_ox2 or
                y2 < expanded_oy1 or
                y1 > expanded_oy2
            )

            if overlap_or_close:
                new_x1 = min(x1, ox1)
                new_y1 = min(y1, oy1)
                new_x2 = max(x2, ox2)
                new_y2 = max(y2, oy2)
                new_area = area + oarea

                merged[idx] = (
                    new_x1,
                    new_y1,
                    new_x2 - new_x1,
                    new_y2 - new_y1,
                    new_area
                )

                has_merged = True
                break

        if not has_merged:
            merged.append(box)

    return merged


def draw_detection_boxes(
    image,
    mask,
    person_area_threshold,
    object_min_area,
    object_max_area,
    border_margin
):
    """
    不合併框版本：
    目標是避免 object 被合併進 person。
    面積大的連通區視為 person。
    面積較小的連通區視為 object。
    """

    result = image.copy()
    img_h, img_w = mask.shape[:2]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < object_min_area:
            continue

        touches_border = (
            x <= border_margin or
            y <= border_margin or
            x + w >= img_w - border_margin or
            y + h >= img_h - border_margin
        )

        if touches_border:
            continue

        # 大區域：人物
        if area >= person_area_threshold:
            label_text = "person"
            color = (0, 255, 0)

        # 小區域：物品
        elif object_min_area <= area <= object_max_area:
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
    const_img = create_constant_image(size, 128) 

        # 統一轉成灰階 L，避免 mode mismatch
    bg_img = bg_img.convert("L")
    in_img = in_img.convert("L")
    const_img = const_img.convert("L")

    # 加一個 assert 確保真的 100% 一樣
    # assert bg_img.size == in_img.size == const_img.size, f"resize 後 still mismatch: {bg_img.size}, {in_img.size}, {const_img.size}"

    # 疊成 RGB：R=bg, G=input, B=128
    rgb = Image.merge("RGB", (bg_img, in_img, const_img))

    # ToTensor + padding
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

    input_picture = int(input("要讀取的圖片編號: "))
    start_time = time.perf_counter()
    input_path = fr"C:\zeicer\dataset\lagi\input\frame_{input_picture:06d}_aligned.jpg"
    bg_path = r"C:\zeicer\dataset\lagi\background.jpg"
    ckpt_dir = r"C:\zeicer\batch_pth"

    output_root = r"C:\zeicer\dataset\lagi\maskpicture"

    raw_dir = os.path.join(output_root, "raw_masks")
    post_dir = os.path.join(output_root, "post_masks")
    box_dir = os.path.join(output_root, "detection_boxes")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    raw_mask_path = os.path.join(raw_dir, f"raw_mask_lagi_{input_picture}.png")
    post_mask_path = os.path.join(post_dir, f"post_mask_lagi_{input_picture}.png")
    box_path = os.path.join(box_dir, f"detection_box_lagi_{input_picture}.png")

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
    
    # 1. raw mask
    mask_img.save(raw_mask_path)

    raw_mask = np.array(mask_img).astype(np.uint8)

    if raw_mask.max() == 1:
        raw_mask = raw_mask * 255

    # 2. post mask：保守處理，不要把人和物品黏起來
    object_mask = postprocess_mask(
        raw_mask,
        kernel_size=3,
        min_area=20,
        border_ignore=20
    )

    person_mask = postprocess_mask_for_person(
        raw_mask,
        kernel_size=7,
        min_area=100,
        border_ignore=20
    )

    cv2.imwrite(post_mask_path, object_mask)

    # 3. detection box
    original_image = cv2.imread(input_path)

    if original_image is None:
        print("原圖讀取失敗，無法輸出 detection box：", input_path)
    else:
        original_image = cv2.resize(original_image, (320, 240))

        box_image = draw_detection_boxes_dual_mask(
            original_image,
            person_mask,
            object_mask,
            person_area_threshold=700,
            object_min_area=15,
            object_max_area=450,
            border_margin=15
        )

        cv2.imwrite(box_path, box_image)

    print("已輸出 raw mask：", raw_mask_path)
    print("已輸出 post mask：", post_mask_path)
    print("已輸出 detection box：", box_path)
    
    end_time = time.perf_counter()
    during_time = end_time - start_time
    print(f"執行時長為 : {during_time:.6f} 秒")

    



if __name__ == "__main__":
    main()
