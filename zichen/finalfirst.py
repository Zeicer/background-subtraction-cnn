import os
import cv2
import torch
import numpy as np
import time
import glob
import re
import subprocess
from ultralytics import YOLO

# 官方命名：PerfectCombinedSystem_YoloPerson_GeometryObject

# =========================================================================
# 幾何邏輯：專門用來過濾與計算「物品（Object）」的遮罩與邊框
# =========================================================================
def postprocess_mask_for_object(mask, kernel_size=3, min_area=20, border_ignore=10):
    mask = mask.astype(np.uint8)
    if mask.max() == 1: mask = mask * 255

    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if border_ignore > 0:
        mask[:border_ignore, :] = 0
        mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0
        mask[:, -border_ignore:] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255
    return clean_mask

def box_inside_or_overlap(box_a, box_b, overlap_threshold=0.3):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = aw * ah
    return (inter_area / area_a) >= overlap_threshold if area_a > 0 else False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 設備啟動: {device}")

    base_dir = "D:/subgarbage"
    input_dir = os.path.join(base_dir, "input")
    mask_dir = os.path.join(base_dir, "maskpicture")
    
    output_dir = os.path.join(base_dir, "11111/combined_results")
    mask_output_dir = os.path.join(base_dir, "11111/mask_results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # =========================================================================
    # 🌟 背景自動執行前置任務 (new_HyRGB.py 生成 raw_mask)
    # =========================================================================
    print("⏳ [Hybrid System] 正在背景自動執行 new_HyRGB.py 生成基礎遮罩...")
    import sys
    python_exe = sys.executable
    hy_rgb_script = os.path.join(base_dir, "new_HyRGB.py")
    try:
        subprocess.run([python_exe, hy_rgb_script], check=True)
        print("✅ [Hybrid System] 前置基礎遮罩生成完畢！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 前置腳本執行失敗，嘗試直接讀取現有遮罩... 錯誤: {e}")

    # =========================================================================
    # 載入人體專用的 YOLO11-Pose
    # =========================================================================
    print("🚀 載入 YOLO11-Pose 模型...")
    pose_model = YOLO("yolo11n-pose.pt")

    image_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*_aligned.jpg")))[100:]
    print(f"📊 開始【YOLO 人體骨架 + 幾何特定物品】聯合辨識，共 {len(image_paths)} 幀")

    # 設定標準雙視窗並排
    cv2.namedWindow("YOLO Pose & Geometry Object Result", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Geometry Object Mask View", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("YOLO Pose & Geometry Object Result", 100, 200)
    cv2.moveWindow("Geometry Object Mask View", 450, 200)

    # 物品記憶體與失蹤補償變數
    last_object_boxes, object_missing_count = [], 0

    for path in image_paths:
        filename = os.path.basename(path)
        frame_num = re.search(r"frame_(\d+)", filename).group(1)
        
        # 1. 讀取原始彩色原圖
        ori_img = cv2.imread(path)
        ori_img = cv2.resize(ori_img, (320, 240))
        
        # 2. 讀取 CNN 基礎黑白遮罩
        mask_path = os.path.join(mask_dir, f"pred_mask_{frame_num}.png")
        if not os.path.exists(mask_path):
            continue
        raw_mask = cv2.imread(mask_path, 0)
        raw_mask = cv2.resize(raw_mask, (320, 240), interpolation=cv2.INTER_NEAREST)

        # 3. 🧠 核心分流：
        # 物品部分：走你的幾何後處理函式 (Morphology + 連通域)
        obj_mask = postprocess_mask_for_object(raw_mask, kernel_size=3, min_area=20, border_ignore=10)
        
        # 人體部分：拿原始遮罩做開閉運算優化，用來刷黑背景餵給 YOLO
        kernel = np.ones((3,3), np.uint8)
        person_bgs_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        person_bgs_mask = cv2.morphologyEx(person_bgs_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_only = cv2.bitwise_and(ori_img, ori_img, mask=person_bgs_mask)

        # 4. 🔥 執行 YOLO11-Pose 骨架人體偵測
        pose_results = pose_model(foreground_only, verbose=False, conf=0.3)
        
        # 建立最終畫布 (複製一份原始彩色原圖)
        annotated_frame = ori_img.copy()
        person_boxes_for_overlap_check = []

        # 5. 🎨 繪製 YOLO 人體框與精準骨架線
        if len(pose_results[0].boxes) > 0:
            pose_results[0].orig_img = annotated_frame
            annotated_frame = pose_results[0].plot(boxes=False) # 先畫出綠色骨架線

            # 手動疊加白色人體框並記錄座標，用來做後續的物品重疊過濾
            for box in pose_results[0].boxes:
                if box.cls[0] == 0: # 類別 0 是人
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    pw, ph = px2 - px1, py2 - py1
                    person_boxes_for_overlap_check.append((px1, py1, pw, ph))
                    
                    # 畫上白色人體框與標籤
                    cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Person {box.conf[0]:.2f}", (px1, max(py1 - 8, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 6. 🔍 執行你的幾何物品邊框計算 (連通域面積過濾)
        num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
        current_frame_object_boxes = []
        
        object_min_area, object_max_area, border_margin = 15, 450, 15
        
        for i in range(1, num_o):
            x, y, w, h, area = stats_o[i, 0], stats_o[i, 1], stats_o[i, 2], stats_o[i, 3], stats_o[i, 4]
            if area < object_min_area or area > object_max_area: continue
            
            touches_border = (x <= border_margin or y <= border_margin or 
                              x + w >= 320 - border_margin or y + h >= 240 - border_margin)
            if touches_border and (h >= 50 or (h / max(w, 1)) >= 1.4): continue
            if touches_border and (w <= 8 or h <= 8): continue
            
            # 💡 核心過濾：利用 YOLO 算出來的精準人體框，排除黏在人身上的衣服或肢體殘影
            is_overlap = any(box_inside_or_overlap((x, y, w, h), p_box, 0.8) for p_box in person_boxes_for_overlap_check)
            if not is_overlap:
                current_frame_object_boxes.append((x, y, w, h, area))

        # 7. 🔄 你的物品失蹤補償與歷史追蹤邏輯
# =========================================================================
        # 7. 🔄 自適應信任度追蹤與動態補償機制 (時間越長，補償幀數越多)
        # =========================================================================
        # 初始配置參數（可根據實際場景微調）
        BASE_COMPENSATION = 10     # 剛出現時的基本補償幀數 (防微小閃爍)
        MAX_COMPENSATION = 150     # 補償幀數的上限值 (大約 5 秒)
        TRUST_ACCUMULATION_RATE = 0.5 # 物品每存在 1 幀，補償寬限期就增加 0.5 幀

        if len(current_frame_object_boxes) > 0:
            last_object_boxes = current_frame_object_boxes
            object_missing_count = 0
            
            # 💡 核心創新：物品持續存在，累加「存活幀數時間」，進而動態拉高補償幀數上限
            if 'object_alive_frames' not in locals() and 'object_alive_frames' not in globals():
                object_alive_frames = 0
            object_alive_frames += 1
            
            # 計算目前這幀該給的動態補償額度 = 基本額度 + (存活時間 * 累積速率)
            current_allowed_compensation = int(BASE_COMPENSATION + (object_alive_frames * TRUST_ACCUMULATION_RATE))
            current_allowed_compensation = min(current_allowed_compensation, MAX_COMPENSATION)

            # 畫上即時偵測到的黃色物品框，並把動態寬限期數值秀在畫面上方便除錯
            for x, y, w, h, a in current_frame_object_boxes:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"object:{a} (Grace:{current_allowed_compensation}f)", 
                            (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # 進入失蹤狀態
            object_missing_count += 1
            
            # 如果之前根本沒算過動態額度，給予基本基本補償
            if 'current_allowed_compensation' not in locals():
                current_allowed_compensation = BASE_COMPENSATION
            
            # 判斷是否在該物品「專屬的動態寬限期」之內
            if object_missing_count <= current_allowed_compensation:
                for x, y, w, h, a in last_object_boxes:
                    # 畫上橘色的歷史追蹤框，並顯示正在倒數的幀數
                    countdown = current_allowed_compensation - object_missing_count
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, f"tracked:{a} (Hold:{countdown}f)", 
                                (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            else:
                # 超過它自己賺到的動態寬限期，徹底遺忘，各項指標歸零
                last_object_boxes = []
                object_alive_frames = 0
                current_allowed_compensation = BASE_COMPENSATION

        # 右邊視窗：單純看物品的幾何修補遮罩
        if obj_mask.max() == 1:
            display_mask = obj_mask * 255
        else:
            display_mask = obj_mask

        # 顯示最終成果
        cv2.imshow("YOLO Pose & Geometry Object Result", annotated_frame) # 左：彩色原圖+YOLO人(骨架/框)+幾何物
        cv2.imshow("Geometry Object Mask View", display_mask)             # 右：純物品的黑白遮罩

        # 儲存彩色成果圖
        cv2.imwrite(os.path.join(output_dir, f"hybrid_pose_obj_{frame_num}.jpg"), annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✨ [Hybrid System] 骨架人與幾何物雙軌一條龍任務完美完成！")

if __name__ == "__main__":
    main()