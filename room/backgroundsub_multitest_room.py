import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import time
import glob
import re
# from backgroundsub_2 import Models  # 從上面那個檔案匯入模型

PATCH_SIZE = 27
PADDING = 13
IMAGE_SIZE = (320, 240)
INFER_BATCH_SIZE = 4096
SAVE_DEBUG_MASKS = True

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
        with torch.inference_mode():
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

        # 人物允許從邊界進入，只排除很薄的邊界雜訊
        is_thin_border_noise = (
            touches_border and
            (w <= 8 or h <= 8)
        )

        if is_thin_border_noise:
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

        object_aspect_ratio = h / max(w, 1)

        # 只在邊界附近時，才排除像人物碎片的物件
        # 避免剛進畫面的人被誤判成 object
        if touches_border and (h >= 50 or object_aspect_ratio >= 1.4):
            continue

        # object 不要完全禁止靠邊，否則靠近邊緣的遺留物會被排除
        # 只排除很薄、很像邊界雜訊的區塊
        is_thin_border_noise = (
            touches_border and
            (w <= 8 or h <= 8)
        )

        if is_thin_border_noise:
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

    return result, person_boxes, object_boxes

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

def create_constant_image(size=IMAGE_SIZE, value=128):
    # size 是 (width, height)，要轉成 (height, width) 給 numpy
    w, h = size
    img_array = np.full((h, w), value, dtype=np.uint8)
    return Image.fromarray(img_array)

def make_rgb_patches_from_rgb(bg_img, in_img, patch_size=PATCH_SIZE, padding=PADDING, size=IMAGE_SIZE):
    """先疊成 RGB，再轉灰階+padding+切 patch，回傳 [N,3,patch,patch]"""

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

def infer_frame(model, bg_img, input_path, device, patch_size=PATCH_SIZE, padding=PADDING, size=IMAGE_SIZE):

    try:
        in_img = Image.open(input_path)
    except FileNotFoundError as e:
        print("讀取圖片失敗：", e)
        return None

    patches, Hp, Wp = make_rgb_patches_from_rgb(
        bg_img,
        in_img,
        patch_size=patch_size,
        padding=padding,
        size=size
    )

    model.eval()
    preds = []

    with torch.inference_mode():
        for i in range(0, patches.size(0), INFER_BATCH_SIZE):
            batch = patches[i:i+INFER_BATCH_SIZE].to(device)
            logits = model(batch)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(Hp, Wp)

    mask = (preds * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).resize(size, Image.NEAREST)

    return mask_img


def main():
    environment = "room"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.perf_counter()

    # input_dir = rf"C:\zeicer\dataset\{environment}\input"
    # input_dir = rf"C:\Users\lab1\Desktop\video\pic_lagi2"
    input_dir = rf"C:\Users\lab1\Desktop\video\pic_obalanuwalk"
    bg_path = rf"C:\zeicer\dataset\{environment}\background.jpg"
    bg_img = Image.open(bg_path)
    ckpt_dir = r"C:\zeicer\batch_pth"

    # output_root = rf"C:\zeicer\dataset\{environment}\maskpicture"
    output_root = rf"C:\zeicer\dataset\{environment}\maskpicture\init2"

    raw_dir = os.path.join(output_root, "raw_masks")
    object_dir = os.path.join(output_root, "object_masks")
    person_dir = os.path.join(output_root, "person_masks")
    box_dir = os.path.join(output_root, "detection_boxes")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(object_dir, exist_ok=True)
    os.makedirs(person_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    # image_paths = image_paths[100:]

    if len(image_paths) == 0:
        print("找不到任何輸入圖片：", input_dir)
        return

    # 測試用：先只跑前 10 張，確認沒問題後可以註解掉
    # image_paths = image_paths[:10]

    print(f"共找到 {len(image_paths)} 張圖片")

    # 建立模型並載入權重，只做一次
    model = BackgroundSubtractorCNN(num_classes=2).to(device)

    ckpt_list = glob.glob(
        os.path.join(ckpt_dir, f"best_model_step*_{environment}_{environment}_patch.pth")
    )
    assert len(ckpt_list) > 0, "找不到對應的權重檔"

    def get_step(path):
        filename = os.path.basename(path)
        match = re.search(rf"best_model_step(\d+)_{environment}_{environment}_patch\.pth", filename)
        return int(match.group(1)) if match else -1

    ckpt_path = max(ckpt_list, key=get_step)

    print("使用權重檔：", ckpt_path)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    success_count = 0
    model_total_time = 0.0

    last_object_boxes = []
    object_missing_count = 0
    max_object_missing_frames = 10   # 物品消失後最多保留幾幀
    max_object_alive_frames = 50     # 物品最多顯示幾幀

    video_path = os.path.join(output_root, "detection_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        video_path,
        fourcc,
        30,          # FPS，可以改成 15、20、30
        IMAGE_SIZE   # 你的 IMAGE_SIZE 是 (320, 240)
    )

    for idx, input_path in enumerate(image_paths):
        filename = os.path.splitext(os.path.basename(input_path))[0]

        print(f"[{idx + 1}/{len(image_paths)}] 處理：{filename}")

        # ========== 1. 模型推論 raw mask ==========
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_start = time.perf_counter()

        mask_img = infer_frame(model, bg_img, input_path, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        model_end = time.perf_counter()
        model_total_time += model_end - model_start

        if mask_img is None:
            print("推論失敗，跳過：", input_path)
            continue

        raw_mask_path = os.path.join(raw_dir, f"{filename}_raw_mask.png")
        object_mask_path = os.path.join(object_dir, f"{filename}_object_mask.png")
        person_mask_path = os.path.join(person_dir, f"{filename}_person_mask.png")
        box_path = os.path.join(box_dir, f"{filename}_detection_box.png")

        raw_mask = np.array(mask_img).astype(np.uint8)

        if raw_mask.max() == 1:
            raw_mask = raw_mask * 255

        # ========== 2. object mask：保守處理，保留小物品 ==========
        object_mask = postprocess_mask(
            raw_mask,
            kernel_size=3,
            min_area=20,
            border_ignore=10
        )

        # ========== 3. person mask：加強處理，讓人物比較完整 ==========
        person_mask = postprocess_mask_for_person(
            raw_mask,
            kernel_size=7,
            min_area=100,
            border_ignore=5
        )

        # ========== 4. detection box ==========
        original_image = cv2.imread(input_path)

        if original_image is None:
            print("原圖讀取失敗，無法輸出 detection box：", input_path)
            continue

        original_image = cv2.resize(original_image, IMAGE_SIZE)

        box_image, person_boxes, object_boxes  = draw_detection_boxes_dual_mask(
            original_image,
            person_mask,
            object_mask,
            person_area_threshold=700,
            object_min_area=50,
            object_max_area=800,
            border_margin=15,
            person_min_height=80,
            person_min_aspect_ratio=1.4
        )

        if SAVE_DEBUG_MASKS:
            mask_img.save(raw_mask_path)
            cv2.imwrite(object_mask_path, object_mask)
            cv2.imwrite(person_mask_path, person_mask)

        # # ========== 5. object 記憶機制 ==========
        # if len(object_boxes) > 0:
        #     # 這一幀有偵測到 object，更新記憶
        #     last_object_boxes = object_boxes
        #     object_missing_count = 0
        # else:
        #     # 這一幀沒偵測到 object
        #     object_missing_count += 1

        #     # 如果還在允許範圍內，就沿用上一幀 object
        #     if object_missing_count <= max_object_missing_frames:
        #         for x, y, w, h, area in last_object_boxes:
        #             color = (0, 165, 255)  # 橘色，代表 tracked object
        #             cv2.rectangle(box_image, (x, y), (x + w, y + h), color, 2)
        #             cv2.putText(
        #                 box_image,
        #                 f"tracked_object:{area}",
        #                 (x, max(y - 5, 15)),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 color,
        #                 1
        #             )
        #     else:
        #         # 消失太久，清除記憶
        #         last_object_boxes = []

        cv2.imwrite(box_path, box_image)
        video_writer.write(box_image)

        cv2.imshow("Detection Result", box_image)

        key = cv2.waitKey(1)  # 1ms，越大播放越慢
        if key == 27:  # 按 ESC 離開
            break

        success_count += 1
    

    video_writer.release()
    print("影片輸出完成：", video_path)
    cv2.destroyAllWindows()

    end_time = time.perf_counter()
    during_time = end_time - start_time

    print("全部處理完成")
    print(f"成功輸出：{success_count}/{len(image_paths)} 張")
    print(f"總耗時：{during_time:.6f} 秒")

    if success_count > 0:
        avg_time = during_time / success_count
        fps = success_count / during_time

        print(f"平均每張耗時：{avg_time:.6f} 秒")
        print(f"End-to-End FPS：{fps:.2f}")

    if model_total_time > 0 and success_count > 0:
        model_fps = success_count / model_total_time

        print(f"模型推論總耗時：{model_total_time:.6f} 秒")
        print(f"Model-only FPS：{model_fps:.2f}")




if __name__ == "__main__":
    main()
