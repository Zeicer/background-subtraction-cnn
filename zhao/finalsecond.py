import os
import cv2
import torch
import numpy as np
import time
import glob
import re
import subprocess
import sys
from ultralytics import YOLO

# 🌟 正確跨檔案引入行為分析器 (從獨立的 behavior_analyzer.py 讀取)
from behavior_analyzer1 import BehaviorAnalyzer

# 官方命名：PerfectCombinedSystem_UltimateBehaviorSystem

def postprocess_mask_for_object(mask, kernel_size=3, min_area=20, border_ignore=10):
    mask = mask.astype(np.uint8)
    if mask.max() == 1: mask = mask * 255
    if border_ignore > 0:
        mask[:border_ignore, :] = 0; mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0; mask[:, -border_ignore:] = 0
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if border_ignore > 0:
        mask[:border_ignore, :] = 0; mask[-border_ignore:, :] = 0
        mask[:, :border_ignore] = 0; mask[:, -border_ignore:] = 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: clean_mask[labels == i] = 255
    return clean_mask

def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a; bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by); ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1); inters = iw * ih
    uni = (aw * ah) + (bw * bh) - inters
    return inters / uni if uni > 0 else 0

def box_inside_or_overlap(box_a, box_b, overlap_threshold=0.3):
    ax, ay, aw, ah = box_a; bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by); ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1); area_a = aw * ah
    return (inter_area / area_a) >= overlap_threshold if area_a > 0 else False

class SimpleGeometryTracker:
    def __init__(self, max_lost=10):
        self.next_id = 1
        self.tracked_objects = {}
        self.max_lost = max_lost  # 💡 關鍵修正：把傳入的參數存進 self 內部

    def update(self, detected_boxes):
        updated_objects = {}; detected_used = [False] * len(detected_boxes)
        for obj_id, (tx, ty, tw, th, lost) in self.tracked_objects.items():
            best_iou, best_idx = 0.3, -1
            for idx, d_box in enumerate(detected_boxes):
                if detected_used[idx]: continue
                iou = calculate_iou((tx, ty, tw, th), d_box)
                if iou > best_iou: best_iou = iou; best_idx = idx
            if best_idx != -1:
                dx, dy, dw, dh = detected_boxes[best_idx]
                updated_objects[obj_id] = (dx, dy, dw, dh, 0); detected_used[best_idx] = True
            else:
                # 💡 關鍵修正：這裡要改成 self.max_lost 程式才認得
                if lost < self.max_lost: 
                    updated_objects[obj_id] = (tx, ty, tw, th, lost + 1)
        for idx, d_box in enumerate(detected_boxes):
            if not detected_used[idx]:
                dx, dy, dw, dh = d_box; updated_objects[self.next_id] = (dx, dy, dw, dh, 0); self.next_id += 1
        self.tracked_objects = updated_objects
        return {k: v[:4] for k, v in self.tracked_objects.items() if v[4] == 0}

def main(base_dir,input_dir,mask_dir,output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 終極行為監控系統啟動。運行設備: {device}")

    # base_dir = "D:/subgarbage"
    input_dir = os.path.join(base_dir, input_dir)
    mask_dir = os.path.join(base_dir, mask_dir)
    output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 🌟 背景執行前置 new_HyRGB.py
    # print("⏳ 正在背景自動執行 new_HyRGB.py...")
    # python_exe = sys.executable
    # hy_rgb_script = os.path.join(base_dir, "new_HyRGB.py")
    # try:
    #     # subprocess.run([python_exe, hy_rgb_script], check=True)
    #     print("✅ 前置遮罩生成完畢！")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ 背景生成遮罩失敗，直接讀取現有遮罩。錯誤: {e}")

    # 模型與分析組件初始化
    pose_model = YOLO("yolo11n-pose.pt")
    person_tracker = SimpleGeometryTracker(max_lost=10)
    object_tracker = SimpleGeometryTracker(max_lost=15)
    analyzer = BehaviorAnalyzer()

    BASE_COMPENSATION, MAX_COMPENSATION, TRUST_ACCUMULATION_RATE = 10, 150, 0.5
    last_object_boxes, object_missing_count = [], 0
    object_alive_frames, current_allowed_compensation = 0, BASE_COMPENSATION

    image_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*_aligned.jpg")))
    
    cv2.namedWindow("Ultimate Behavioral Monitor", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Geometry Object Mask View", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Ultimate Behavioral Monitor", 100, 150)
    cv2.moveWindow("Geometry Object Mask View", 450, 150)

    for path in image_paths:
        filename = os.path.basename(path)
        frame_num = re.search(r"frame_(\d+)", filename).group(1)
        
        ori_img = cv2.resize(cv2.imread(path), (320, 240))
        # mask_path = os.path.join(mask_dir, f"pred_mask_{frame_num}.png")
        name = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_dir, f"{name}_mask.png")
        if not os.path.exists(mask_path): continue
        raw_mask = cv2.resize(cv2.imread(mask_path, 0), (320, 240), interpolation=cv2.INTER_NEAREST)

        # 幾何分流與 YOLO 背景過濾
        obj_mask = postprocess_mask_for_object(raw_mask, kernel_size=3, min_area=20, border_ignore=10)
        kernel = np.ones((3,3), np.uint8)
        person_bgs_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        person_bgs_mask = cv2.morphologyEx(person_bgs_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_only = cv2.bitwise_and(ori_img, ori_img, mask=person_bgs_mask)

        # 推論
        pose_results = pose_model(foreground_only, verbose=False, conf=0.3)
        annotated_frame = ori_img.copy()
        detected_people_boxes = []

        if len(pose_results[0].boxes) > 0:
            pose_results[0].orig_img = annotated_frame
            annotated_frame = pose_results[0].plot(boxes=False) # 繪製綠色骨架

            for box in pose_results[0].boxes:
                if box.cls[0] == 0:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    detected_people_boxes.append((px1, py1, px2-px1, py2-py1))

        # 追蹤同一個人，進行行為分析
        active_people = person_tracker.update(detected_people_boxes)
        for pid, bbox in active_people.items():
            analyzer.update_person(pid, bbox)
            
            # 🚨 四大行為分析
            is_run, speed = analyzer.check_running(pid)
            is_loiter = analyzer.check_loitering(pid)
            is_fall = analyzer.check_fall(pid)

            px, py, pw, ph = bbox
            cv2.rectangle(annotated_frame, (px, py), (px+pw, py+ph), (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"ID:{pid}", (px, max(py-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 在彩色原圖上渲染警報
            if is_fall:
                cv2.putText(annotated_frame, "ALERT: FALLING!", (px, py + ph + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            elif is_run:
                cv2.putText(annotated_frame, f"RUNNING ({int(speed)}px/s)", (px, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            if is_loiter:
                cv2.putText(annotated_frame, "LOITERING", (px + pw - 50, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 碰撞分析
        pids = list(active_people.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if analyzer.check_collision(pids[i], pids[j]):
                    bx1, by1, bw1, bh1 = active_people[pids[i]]
                    bx2, by2, bw2, bh2 = active_people[pids[j]]
                    cv2.putText(annotated_frame, "💥 COLLISION!", ((bx1+bx2)//2, (by1+by2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 75, 255), 2)

        # 幾何物品偵測
        num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
        current_frame_object_boxes = []
        for i in range(1, num_o):
            x, y, w, h, area = stats_o[i, 0], stats_o[i, 1], stats_o[i, 2], stats_o[i, 3], stats_o[i, 4]
            if area < 15 or area > 450: continue
            if (x <= 15 or y <= 15 or x + w >= 305 or y + h >= 225) and (h >= 50 or (h / max(w, 1)) >= 1.4 or w <= 8 or h <= 8): continue
            if not any(box_inside_or_overlap((x, y, w, h), b, 0.8) for b in detected_people_boxes):
                current_frame_object_boxes.append((x, y, w, h))

        # 物品自適應時序追蹤與亂丟垃圾警告
        if len(current_frame_object_boxes) > 0:
            last_object_boxes, object_missing_count = current_frame_object_boxes, 0
            object_alive_frames += 1
            current_allowed_compensation = min(int(BASE_COMPENSATION + (object_alive_frames * TRUST_ACCUMULATION_RATE)), MAX_COMPENSATION)
            
            active_objects = object_tracker.update(current_frame_object_boxes)
            for oid, obbox in active_objects.items():
                analyzer.update_object(oid, obbox)
                is_litter = analyzer.check_littering(oid)
                ox, oy, ow, oh = obbox
                cv2.rectangle(annotated_frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 2)
                if is_litter:
                    cv2.putText(annotated_frame, "⚠️ LITTERING DETECTED!", (ox, max(oy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, f"object:{ow*oh} (Grace:{current_allowed_compensation}f)", (ox, max(oy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            object_missing_count += 1
            if object_missing_count <= current_allowed_compensation:
                for x, y, w, h in last_object_boxes:
                    countdown = current_allowed_compensation - object_missing_count
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(annotated_frame, f"tracked (Hold:{countdown}f)", (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            else:
                last_object_boxes = []; object_alive_frames = 0; current_allowed_compensation = BASE_COMPENSATION

        analyzer.clear_dead_tracks(list(active_people.keys()), list(object_tracker.tracked_objects.keys()))

        # 顯示雙視窗
        display_mask = obj_mask if obj_mask.max() == 255 else obj_mask * 255
        cv2.imshow("Ultimate Behavioral Monitor", annotated_frame)
        cv2.imshow("Geometry Object Mask View", display_mask)

        cv2.imwrite(os.path.join(output_dir, f"behavior_{frame_num}.jpg"), annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()
    print("✨ 所有行為整合監控與影像輸出全部完美搞定！")

if __name__ == "__main__":
    main(base_dir="D:/subgarbage",
        input_dir="input",
        mask_dir="maskpicture",
        output_dir="11111/combined_results")
