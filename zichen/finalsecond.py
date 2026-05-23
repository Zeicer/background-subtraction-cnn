import os
import cv2
import torch
import numpy as np
import time
import glob
import re
import subprocess
import sys
from collections import deque
from ultralytics import YOLO

# 🌟 正確跨檔案引入行為分析器
from behavior_analyzer import BehaviorAnalyzer

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
        self.max_lost = max_lost  

    def update(self, detected_boxes):
        updated_objects = {}; detected_used = [False] * len(detected_boxes)
        for obj_id, (tx, ty, tw, th, lost) in self.tracked_objects.items():
            best_iou, best_idx = 0.15, -1  
            for idx, d_box in enumerate(detected_boxes):
                if detected_used[idx]: continue
                iou = calculate_iou((tx, ty, tw, th), d_box)
                if iou > best_iou: best_iou = iou; best_idx = idx
            if best_idx != -1:
                dx, dy, dw, dh = detected_boxes[best_idx]
                updated_objects[obj_id] = (dx, dy, dw, dh, 0); detected_used[best_idx] = True
            else:
                if lost < self.max_lost: 
                    updated_objects[obj_id] = (tx, ty, tw, th, lost + 1)
        for idx, d_box in enumerate(detected_boxes):
            if not detected_used[idx]:
                dx, dy, dw, dh = d_box; updated_objects[self.next_id] = (dx, dy, dw, dh, 0); self.next_id += 1
        self.tracked_objects = updated_objects
        return {k: v[:4] for k, v in self.tracked_objects.items() if v[4] == 0}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 終極行為監控系統啟動。運行設備: {device}")

    base_dir = "D:/subgarbage"
    input_dir = os.path.join(base_dir, "input")
    mask_dir = os.path.join(base_dir, "maskpicture")
    output_dir = os.path.join(base_dir, "11111/combined_results")
    os.makedirs(output_dir, exist_ok=True)

    behavior_main_dir = os.path.join(base_dir, "11111/fivebehavior_videos")
    os.makedirs(behavior_main_dir, exist_ok=True)

    behavior_dirs = {
        "running": os.path.join(behavior_main_dir, "RUNNING"),
        "loiter": os.path.join(behavior_main_dir, "LOITERING"),
        "faint": os.path.join(behavior_main_dir, "FAINT"),
        "collision": os.path.join(behavior_main_dir, "COLLISION"),
        "litter": os.path.join(behavior_main_dir, "LITTERING")
    }
    for b_dir in behavior_dirs.values():
        os.makedirs(b_dir, exist_ok=True)

    print("⏳ 正在背景自動執行 new_HyRGB.py...")
    python_exe = sys.executable
    hy_rgb_script = os.path.join(base_dir, "new_HyRGB.py")
    try:
        subprocess.run([python_exe, hy_rgb_script], check=True)
        print("✅ 前置遮罩生成完畢！")
    except subprocess.CalledProcessError as e:
        print(f"❌ 背景生成遮罩失敗，直接讀取現有遮罩。錯誤: {e}")

    pose_model = YOLO("yolo11n-pose.pt")
    person_tracker = SimpleGeometryTracker(max_lost=25) 
    object_tracker = SimpleGeometryTracker(max_lost=15)
    analyzer = BehaviorAnalyzer()

    BASE_COMPENSATION, MAX_COMPENSATION, TRUST_ACCUMULATION_RATE = 10, 150, 0.5
    last_object_boxes, object_missing_count = [], 0
    object_alive_frames, current_allowed_compensation = 0, BASE_COMPENSATION
    object_alive_frames_dict = {}

    image_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*_aligned.jpg")))
    
    cv2.namedWindow("1. Ultimate Behavioral Monitor (RGB)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("2. Geometry Object Mask View (B&W)", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("3. Foreground Debug View (Mask+RGB)", cv2.WINDOW_AUTOSIZE)

    frame_buffer = deque(maxlen=61)         
    video_tasks = []                        
    behavior_cooldown = {k: -999 for k in behavior_dirs.keys()} 
    global_frame_idx = 0                    

    for path in image_paths:
        filename = os.path.basename(path)
        frame_num = re.search(r"frame_(\d+)", filename).group(1)
        
        ori_img = cv2.resize(cv2.imread(path), (320, 240))
        mask_path = os.path.join(mask_dir, f"pred_mask_{frame_num}.png")
        if not os.path.exists(mask_path): continue
        raw_mask = cv2.resize(cv2.imread(mask_path, 0), (320, 240), interpolation=cv2.INTER_NEAREST)

        # 幾何分流與 YOLO 背景過濾
        obj_mask = postprocess_mask_for_object(raw_mask, kernel_size=3, min_area=20, border_ignore=10)
        kernel = np.ones((3,3), np.uint8)
        person_bgs_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        person_bgs_mask = cv2.morphologyEx(person_bgs_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground_only = cv2.bitwise_and(ori_img, ori_img, mask=person_bgs_mask)

        # 初始化視窗影像
        annotated_frame = ori_img.copy()        # 第一視窗：原彩圖
        bg_remove_render = foreground_only.copy() # 第三視窗：去背原彩圖

        # 推論與骨架繪製
        pose_results = pose_model(ori_img, verbose=False, conf=0.3) 
        if len(pose_results[0].boxes) > 0:
            # 第一視窗：標出骨架
            pose_results[0].orig_img = annotated_frame
            annotated_frame = pose_results[0].plot(boxes=False)
            # 第三視窗：標出骨架
            pose_results[0].orig_img = bg_remove_render
            bg_remove_render = pose_results[0].plot(boxes=False)

        detected_people_boxes = []
        detected_keypoints_list = []

        if len(pose_results[0].boxes) > 0:
            all_boxes = pose_results[0].boxes
            all_kpts_data = pose_results[0].keypoints.xy.cpu().numpy() if pose_results[0].keypoints is not None else None

            for idx, box in enumerate(all_boxes):
                if box.cls[0] == 0: 
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    detected_people_boxes.append((px1, py1, px2-px1, py2-py1))
                    if all_kpts_data is not None and idx < len(all_kpts_data):
                        detected_keypoints_list.append(all_kpts_data[idx])
                    else:
                        detected_keypoints_list.append(None)

        frame_triggers = {k: False for k in behavior_dirs.keys()}

        # 追蹤人員與繪製
        active_people = person_tracker.update(detected_people_boxes)
        for pid, bbox in active_people.items():
            px, py, pw, ph = bbox
            matched_kpts = None
            best_iou = 0.5 
            for idx, d_box in enumerate(detected_people_boxes):
                if calculate_iou(bbox, d_box) > best_iou:
                    matched_kpts = detected_keypoints_list[idx]
                    break
            
            analyzer.update_person(pid, bbox, keypoints=matched_kpts)
            is_run, speed = analyzer.check_running(pid)
            is_loiter = analyzer.check_loitering(pid)
            is_fall = analyzer.check_fall(pid)

            # 🟢 第一視窗與第三視窗：同步繪製白色人物框與 ID
            for view in [annotated_frame, bg_remove_render]:
                cv2.rectangle(view, (px, py), (px+pw, py+ph), (255, 255, 255), 2)
                cv2.putText(view, f"ID:{pid}", (px, max(py-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 判定行為觸發
            if is_fall: frame_triggers["faint"] = True
            elif is_run: frame_triggers["running"] = True
            if is_loiter: frame_triggers["loiter"] = True

            # 警報文字提示（標在畫面上）
            for view in [annotated_frame, bg_remove_render]:
                if is_fall: cv2.putText(view, " ALERT: FAINT!", (px - 10, py + ph + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                elif is_run: cv2.putText(view, f"RUNNING ({int(speed)}%)", (px, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if is_loiter: cv2.putText(view, "LOITERING", (px + pw - 65, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        # 多人碰撞分析
        pids = list(active_people.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if analyzer.check_collision(pids[i], pids[j]):
                    frame_triggers["collision"] = True 
                    bx1, by1, _ , _ = active_people[pids[i]]
                    bx2, by2, _ , _ = active_people[pids[j]]
                    for view in [annotated_frame, bg_remove_render]:
                        cv2.putText(view, " COLLISION!", ((bx1+bx2)//2, (by1+by2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 75, 255), 2)
# 幾何物品偵測
        num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
        current_frame_object_boxes = []
        for i in range(1, num_o):
            x, y, w, h, area = stats_o[i, 0], stats_o[i, 1], stats_o[i, 2], stats_o[i, 3], stats_o[i, 4]
            
            # 1. 基礎面積過濾 (太小或太大的連通集不要)
            if area < 15 or area > 450: continue
            
            # 🛠️ 2. 【新增長寬硬限制】如果白塊的寬度或高度小於 3 像素，直接判定為非物品（雜訊過濾）
            if w < 3 or h < 3: continue
            
            # 3. 邊緣無效拉長或極小碎屑過濾 (貼邊且形狀異常的過濾)
            if (x <= 15 or y <= 15 or x + w >= 305 or y + h >= 225) and (h >= 50 or (h / max(w, 1)) >= 1.4 or w <= 8 or h <= 8): continue
            
            # 4. 排除壓在人體 YOLO 框內部的物件（避免把人的腳或手當作垃圾）
            if not any(box_inside_or_overlap((x, y, w, h), b, 0.8) for b in detected_people_boxes):
                current_frame_object_boxes.append((x, y, w, h))
        # 處理物品追蹤與「補償框/物品框」繪製邏輯
        if len(current_frame_object_boxes) > 0:
            last_object_boxes, object_missing_count = current_frame_object_boxes, 0
            object_alive_frames += 1
            current_allowed_compensation = min(int(BASE_COMPENSATION + (object_alive_frames * TRUST_ACCUMULATION_RATE)), MAX_COMPENSATION)
            
            active_objects = object_tracker.update(current_frame_object_boxes)
            for oid in active_objects.keys():
                object_alive_frames_dict[oid] = object_alive_frames_dict.get(oid, 0) + 1
                
            for oid, obbox in active_objects.items():
                analyzer.update_object(oid, obbox)
                is_litter = analyzer.check_littering(oid)
                ox, oy, ow, oh = obbox
                
                if is_litter:
                    frame_triggers["litter"] = True 
                    assigned_pid = getattr(analyzer, 'object_owner_memory', {}).get(oid)
                    
                    # 🔴 觸發丟垃圾：第一視窗與第三視窗皆同步對「人體」畫粗紅框
                    if assigned_pid is not None and assigned_pid in active_people:
                        px, py, pw, ph = active_people[assigned_pid]
                        for view in [annotated_frame, bg_remove_render]:
                            cv2.rectangle(view, (px, py), (px + pw, py + ph), (0, 0, 255), 3)
                            cv2.putText(view, f"LITTERER CAUGHT! (ID:{assigned_pid})", (px, max(py - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    
                    # 🟡 【第三視窗限定】標出紅色物品框，第一視窗不標
                    cv2.rectangle(bg_remove_render, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                    cv2.putText(bg_remove_render, f"LITTER OBJ:{oid}", (ox, max(oy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                else:
                    # 🟡 【第三視窗限定】未觸發時，標出黃色常態物品框，第一視窗不標
                    cv2.rectangle(bg_remove_render, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 1)
                    cv2.putText(bg_remove_render, f"obj:{oid}", (ox, max(oy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        else:
            object_missing_count += 1
            # 💡 進入補償機制：代表當前影格幾何沒偵測到，但歷史補償額度尚未扣完
            if object_missing_count <= current_allowed_compensation and max(object_alive_frames_dict.values() if object_alive_frames_dict else [0]) >= 3:
                # 🟡 【第三視窗限定】繪製補償框，並使用乾淨、無「???」錯誤的專業倒數文字
                for last_box in last_object_boxes:
                    lox, loy, low, loh = last_box
                    cv2.rectangle(bg_remove_render, (lox, loy), (lox + low, loy + loh), (0, 165, 255), 1, cv2.LINE_AA)
                    
                    # 🛠️ 修正文字：替換掉原本可能帶有未定義變數或錯誤符號的代碼，改用穩定的全局倒數變數
                    cv2.putText(bg_remove_render, f"Tracking Lost ({object_missing_count}/{current_allowed_compensation})", (lox, max(loy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
            else:
                last_object_boxes = []; object_alive_frames = 0; current_allowed_compensation = BASE_COMPENSATION
                object_alive_frames_dict.clear()

        analyzer.clear_dead_tracks(list(active_people.keys()), list(object_tracker.tracked_objects.keys()))

        # 📺 影像送入暫存滾動隊列：儲存僅帶有骨架、人體框的「第一視窗原彩圖」
        frame_buffer.append((global_frame_idx, frame_num, annotated_frame.copy()))

        # 🎬 =========================================================================
        # 🎬 錄影任務派發與去重檢查
        # =========================================================================
        for b_name, triggered in frame_triggers.items():
            if triggered:
                if global_frame_idx - behavior_cooldown[b_name] <= 60:
                    continue  
                
                behavior_cooldown[b_name] = global_frame_idx
                print(f"🎬 [事件觸發] 行為: {b_name.upper()} | 觸發點 Frame: {frame_num} | 啟動去重影片錄製。")
                
                start_idx = max(0, global_frame_idx - 60)
                end_idx = global_frame_idx + 30
                
                video_tasks.append({
                    "behavior": b_name,
                    "trigger_frame": frame_num,
                    "start_global_idx": start_idx,
                    "end_global_idx": end_idx,
                    "saved": False
                })

        for task in video_tasks:
            if not task["saved"] and global_frame_idx >= task["end_global_idx"]:
                extracted_frames = [f for idx, f_num, f in frame_buffer if task["start_global_idx"] <= idx <= task["end_global_idx"]]
                
                if len(extracted_frames) > 0:
                    b_name = task["behavior"]
                    t_frame = task["trigger_frame"]
                    video_filename = os.path.join(behavior_dirs[b_name], f"event_trigger_{t_frame}.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (320, 240))
                    
                    for f in extracted_frames:
                        out_video.write(f)
                    out_video.release()
                    
                    print(f"💾 [影片分類成功] {b_name.upper()} 原彩去重影片已存入: {video_filename}")
                    task["saved"] = True

        video_tasks = [t for t in video_tasks if not t["saved"]]
        global_frame_idx += 1
        # =========================================================================

        # 📺 三視窗同步輸出渲染顯示
        display_mask = obj_mask if obj_mask.max() == 255 else obj_mask * 255
        
        cv2.imshow("1. Ultimate Behavioral Monitor (RGB)", annotated_frame)
        cv2.imshow("2. Geometry Object Mask View (B&W)", display_mask)
        cv2.imshow("3. Foreground Debug View (Mask+RGB)", bg_remove_render) 

        cv2.imwrite(os.path.join(output_dir, f"behavior_{frame_num}.jpg"), annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # 片尾保底處理
    for task in video_tasks:
        if not task["saved"]:
            extracted_frames = [f for idx, f_num, f in frame_buffer if task["start_global_idx"] <= idx]
            if len(extracted_frames) > 0:
                b_name = task["behavior"]
                t_frame = task["trigger_frame"]
                video_filename = os.path.join(behavior_dirs[b_name], f"event_trigger_{t_frame}_end.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (320, 240))
                for f in extracted_frames: out_video.write(f)
                out_video.release()
    
    cv2.destroyAllWindows()
    print("✨ 三視窗自定義視覺化標記重構完畢！")

if __name__ == "__main__":
    main()