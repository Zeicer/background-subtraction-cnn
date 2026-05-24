import os
import cv2
import torch
import numpy as np
import glob
import re
from collections import deque
from ultralytics import YOLO

from behavior_analyzer import BehaviorAnalyzer


def postprocess_mask_for_object(mask, kernel_size=3, min_area=20, border_ignore=10):
    mask = mask.astype(np.uint8)

    if mask.max() == 1:
        mask = mask * 255

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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask


def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inters = iw * ih
    uni = (aw * ah) + (bw * bh) - inters

    return inters / uni if uni > 0 else 0


def box_inside_or_overlap(box_a, box_b, overlap_threshold=0.3):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = aw * ah

    return (inter_area / area_a) >= overlap_threshold if area_a > 0 else False


class SimpleGeometryTracker:
    def __init__(self, max_lost=10):
        self.next_id = 1
        self.tracked_objects = {}
        self.max_lost = max_lost

    def update(self, detected_boxes):
        updated_objects = {}
        detected_used = [False] * len(detected_boxes)

        for obj_id, (tx, ty, tw, th, lost) in self.tracked_objects.items():
            best_iou = 0.15
            best_idx = -1

            for idx, d_box in enumerate(detected_boxes):
                if detected_used[idx]:
                    continue

                iou = calculate_iou(
                    (tx, ty, tw, th),
                    d_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx != -1:
                dx, dy, dw, dh = detected_boxes[best_idx]
                updated_objects[obj_id] = (dx, dy, dw, dh, 0)
                detected_used[best_idx] = True
            else:
                if lost < self.max_lost:
                    updated_objects[obj_id] = (
                        tx,
                        ty,
                        tw,
                        th,
                        lost + 1
                    )

        for idx, d_box in enumerate(detected_boxes):
            if not detected_used[idx]:
                dx, dy, dw, dh = d_box
                updated_objects[self.next_id] = (
                    dx,
                    dy,
                    dw,
                    dh,
                    0
                )
                self.next_id += 1

        self.tracked_objects = updated_objects

        return {
            k: v[:4]
            for k, v in self.tracked_objects.items()
            if v[4] == 0
        }


def main(
    base_dir,
    input_dir,
    mask_dir,
    output_dir,
    save_mode="event"
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"🔹 終極行為監控系統啟動。運行設備: {device}")

    input_dir = os.path.join(base_dir, input_dir)
    mask_dir = os.path.join(base_dir, mask_dir)
    output_dir = os.path.join(base_dir, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    save_1111_dir = os.path.join(base_dir, "1111")
    os.makedirs(save_1111_dir, exist_ok=True)

    behavior_main_dir = os.path.join(
        base_dir,
        "11111",
        "fivebehavior_videos"
    )

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

    pose_model = YOLO("yolo11n-pose.pt")

    person_tracker = SimpleGeometryTracker(max_lost=25)
    object_tracker = SimpleGeometryTracker(max_lost=15)

    analyzer = BehaviorAnalyzer()

    BASE_COMPENSATION = 10
    MAX_COMPENSATION = 150
    TRUST_ACCUMULATION_RATE = 0.5

    last_object_boxes = []
    object_missing_count = 0
    object_alive_frames_dict = {}

    image_paths = []

    extensions = ["*.jpg", "*.png", "*.jpeg"]

    for ext in extensions:
        image_paths.extend(
            glob.glob(
                os.path.join(input_dir, ext)
            )
        )

    image_paths = sorted(image_paths)

    print(f"找到原圖數量: {len(image_paths)} 張")
    print(f"讀取 mask 資料夾: {mask_dir}")
    print(f"行為分析輸出資料夾: {output_dir}")

    cv2.namedWindow("1. Geometry Object Mask View (B&W)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Ultimate Behavioral Monitor (RGB)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3. Foreground Debug View (Mask+RGB)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("4. Clean Behavioral Monitor (RGB)", cv2.WINDOW_NORMAL)

    for win in ["1. Geometry Object Mask View (B&W)", "2. Ultimate Behavioral Monitor (RGB)", 
                "3. Foreground Debug View (Mask+RGB)", "4. Clean Behavioral Monitor (RGB)"]:
        cv2.resizeWindow(win, 480, 360)

    cv2.moveWindow("1. Geometry Object Mask View (B&W)", 50, 50)
    cv2.moveWindow("2. Ultimate Behavioral Monitor (RGB)", 550, 50)
    cv2.moveWindow("3. Foreground Debug View (Mask+RGB)", 50, 460)
    cv2.moveWindow("4. Clean Behavioral Monitor (RGB)", 550, 460)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0
    video_size = (320, 240)

    out_v1 = cv2.VideoWriter(os.path.join(save_1111_dir, "1_geometry_mask.mp4"), fourcc, fps, video_size, isColor=True)
    out_v2 = cv2.VideoWriter(os.path.join(save_1111_dir, "2_ultimate_monitor.mp4"), fourcc, fps, video_size)
    out_v3 = cv2.VideoWriter(os.path.join(save_1111_dir, "3_foreground_debug.mp4"), fourcc, fps, video_size)
    out_v4 = cv2.VideoWriter(os.path.join(save_1111_dir, "4_clean_monitor.mp4"), fourcc, fps, video_size)

    frame_buffer = deque(maxlen=100)
    video_tasks = []
    behavior_cooldown = {
        k: -999
        for k in behavior_dirs.keys()
    }

    global_frame_idx = 0

    for path in image_paths:
        filename = os.path.basename(path)
        frame_match = re.search(r"frame_(\d+)", filename)

        if not frame_match:
            continue

        frame_num = frame_match.group(1)
        ori_img = cv2.imread(path)

        if ori_img is None:
            continue

        ori_img = cv2.resize(ori_img, (320, 240))
        name = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_dir, f"{name}_mask.png")

        if not os.path.exists(mask_path):
            print(f"⚠️ 找不到對應 mask: {mask_path}")
            continue

        raw_mask = cv2.imread(mask_path, 0)
        if raw_mask is None:
            continue

        raw_mask = cv2.resize(
            raw_mask,
            (320, 240),
            interpolation=cv2.INTER_NEAREST
        )

        obj_mask = postprocess_mask_for_object(
            raw_mask,
            kernel_size=3,
            min_area=20,
            border_ignore=10
        )

        kernel = np.ones((3, 3), np.uint8)

        person_bgs_mask = cv2.morphologyEx(
            raw_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        person_bgs_mask = cv2.morphologyEx(
            person_bgs_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        foreground_only = cv2.bitwise_and(
            ori_img,
            ori_img,
            mask=person_bgs_mask
        )

        annotated_frame = ori_img.copy() 
        bg_remove_render = foreground_only.copy()
        clean_frame = ori_img.copy()

        # 🔄【改回最穩定的寫法】將模型輸入改回原始有背景的「ori_img」，確保 AI 特徵不丟失
        pose_results = pose_model(
            ori_img,
            verbose=False,
            conf=0.3
        )

        if len(pose_results[0].boxes) > 0:
            pose_results[0].orig_img = annotated_frame
            annotated_frame = pose_results[0].plot(boxes=False)

            pose_results[0].orig_img = bg_remove_render
            bg_remove_render = pose_results[0].plot(boxes=False)

        detected_people_boxes = []
        detected_keypoints_list = []

        if len(pose_results[0].boxes) > 0:
            all_boxes = pose_results[0].boxes

            if pose_results[0].keypoints is not None:
                all_kpts_data = (
                    pose_results[0]
                    .keypoints
                    .xy
                    .cpu()
                    .numpy()
                )
            else:
                all_kpts_data = None

            for idx, box in enumerate(all_boxes):
                if int(box.cls[0]) == 0:
                    px1, py1, px2, py2 = map(
                        int,
                        box.xyxy[0]
                    )

                    detected_people_boxes.append(
                        (
                            px1,
                            py1,
                            px2 - px1,
                            py2 - py1
                        )
                    )

                    if (
                        all_kpts_data is not None
                        and idx < len(all_kpts_data)
                    ):
                        detected_keypoints_list.append(
                            all_kpts_data[idx]
                        )
                    else:
                        detected_keypoints_list.append(None)

        frame_triggers = {
            k: False
            for k in behavior_dirs.keys()
        }

        active_people = person_tracker.update(
            detected_people_boxes
        )

        for pid, bbox in active_people.items():
            px, py, pw, ph = bbox

            matched_kpts = None
            best_iou = 0.5

            for idx, d_box in enumerate(detected_people_boxes):
                iou = calculate_iou(
                    bbox,
                    d_box
                )

                if iou > best_iou:
                    best_iou = iou
                    matched_kpts = detected_keypoints_list[idx]

            analyzer.update_person(
                pid,
                bbox,
                keypoints=matched_kpts
            )

            is_run, speed = analyzer.check_running(pid)
            is_loiter = analyzer.check_loitering(pid)
            is_fall = analyzer.check_fall(pid)

            for view in [annotated_frame, bg_remove_render, clean_frame]:
                cv2.rectangle(
                    view,
                    (px, py),
                    (px + pw, py + ph),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    view,
                    f"ID:{pid}",
                    (px, max(py - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

            if is_fall:
                frame_triggers["faint"] = True
            elif is_run:
                frame_triggers["running"] = True

            if is_loiter:
                frame_triggers["loiter"] = True

            for view in [annotated_frame, bg_remove_render, clean_frame]:
                if is_fall:
                    cv2.putText(
                        view,
                        " ALERT: FAINT!",
                        (px - 10, py + ph + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2
                    )
                    frame_triggers["faint"] = True
                elif is_run:
                    cv2.putText(
                        view,
                        f"RUNNING ({int(speed)}%)",
                        (px, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1
                    )
                    frame_triggers["running"] = True

                if is_loiter:
                    cv2.putText(
                        view,
                        "LOITERING",
                        (px + pw - 65, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1
                    )
                    frame_triggers["loiter"] = True

        pids = list(active_people.keys())

        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                if analyzer.check_collision(
                    pids[i],
                    pids[j]
                ):
                    frame_triggers["collision"] = True

                    bx1, by1, _, _ = active_people[pids[i]]
                    bx2, by2, _, _ = active_people[pids[j]]

                    for view in [annotated_frame, bg_remove_render, clean_frame]:
                        cv2.putText(
                            view,
                            " COLLISION!",
                            ((bx1 + bx2) // 2, (by1 + by2) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 75, 255),
                            2
                        )

        num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(
            obj_mask,
            connectivity=8
        )

        current_frame_object_boxes = []

        for i in range(1, num_o):
            x = stats_o[i, 0]
            y = stats_o[i, 1]
            w = stats_o[i, 2]
            h = stats_o[i, 3]
            area = stats_o[i, 4]

            if area < 15 or area > 450:
                continue
            if w < 3 or h < 3:
                continue

            if (
                x <= 15
                or y <= 15
                or x + w >= 305
                or y + h >= 225
            ) and (
                h >= 50
                or (h / max(w, 1)) >= 1.4
                or w <= 8
                or h <= 8
            ):
                continue

            if not any(
                box_inside_or_overlap(
                    (x, y, w, h),
                    b,
                    0.8
                )
                for b in detected_people_boxes
            ):
                current_frame_object_boxes.append((x, y, w, h))

        if len(current_frame_object_boxes) > 0:
            last_object_boxes = current_frame_object_boxes
            object_missing_count = 0

            active_objects = object_tracker.update(current_frame_object_boxes)

            for oid in active_objects.keys():
                object_alive_frames_dict[oid] = object_alive_frames_dict.get(oid, 0) + 1

            for oid, obbox in active_objects.items():
                analyzer.update_object(oid, obbox)
                is_litter = analyzer.check_littering(oid)
                ox, oy, ow, oh = obbox

                if is_litter:
                    frame_triggers["litter"] = True
                    assigned_pid = getattr(analyzer, "object_owner_memory", {}).get(oid)

                    if assigned_pid is not None and assigned_pid in active_people:
                        px, py, pw, ph = active_people[assigned_pid]

                        for view in [annotated_frame, bg_remove_render, clean_frame]:
                            cv2.rectangle(view, (px, py), (px + pw, py + ph), (0, 0, 255), 3)
                            cv2.putText(view, f"LITTERER CAUGHT! (ID:{assigned_pid})", 
                                        (px, max(py - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    for view in [annotated_frame, bg_remove_render, clean_frame]:
                        cv2.rectangle(view, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                        cv2.putText(view, f"LITTER OBJ:{oid}", (ox, max(oy - 5, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                else:
                    for view in [annotated_frame, bg_remove_render, clean_frame]:
                        cv2.rectangle(view, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 1)
                        cv2.putText(view, f"obj:{oid}", (ox, max(oy - 5, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        else:
            object_missing_count += 1
            valid_compensation_boxes = []

            for oid, alive_count in list(object_alive_frames_dict.items()):
                if alive_count >= 5:
                    allowed_comp = min(
                        int(BASE_COMPENSATION + (alive_count * TRUST_ACCUMULATION_RATE)),
                        MAX_COMPENSATION
                    )

                    if object_missing_count <= allowed_comp:
                        if oid in object_tracker.tracked_objects:
                            tx, ty, tw, th, _ = object_tracker.tracked_objects[oid]
                            valid_compensation_boxes.append((oid, (tx, ty, tw, th), allowed_comp))
                    else:
                        object_alive_frames_dict.pop(oid, None)
                else:
                    object_alive_frames_dict.pop(oid, None)

            if len(valid_compensation_boxes) > 0:
                for oid, box, allowed_comp in valid_compensation_boxes:
                    lox, loy, low, loh = box
                    for view in [annotated_frame, bg_remove_render, clean_frame]:
                        cv2.rectangle(view, (lox, loy), (lox + low, loy + loh), (0, 165, 255), 1, cv2.LINE_AA)
                        cv2.putText(view, f"Tracking Lost ({object_missing_count}/{allowed_comp})", 
                                    (lox, max(loy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
            else:
                last_object_boxes = []
                object_alive_frames_dict.clear()

        analyzer.clear_dead_tracks(
            list(active_people.keys()),
            list(object_tracker.tracked_objects.keys())
        )

        frame_buffer.append(
            (
                global_frame_idx,
                frame_num,
                annotated_frame.copy()
            )
        )

        for b_name, triggered in frame_triggers.items():
            if triggered:
                if global_frame_idx - behavior_cooldown[b_name] <= 60:
                    continue

                behavior_cooldown[b_name] = global_frame_idx

                print(
                    f"🎬 [事件觸發] 行為: {b_name.upper()} | "
                    f"觸發點 Frame: {frame_num}"
                )

                start_idx = max(0, global_frame_idx - 60)
                end_idx = global_frame_idx + 30

                video_tasks.append(
                    {
                        "behavior": b_name,
                        "trigger_frame": frame_num,
                        "start_global_idx": start_idx,
                        "end_global_idx": end_idx,
                        "saved": False
                    }
                )

        for task in video_tasks:
            if (
                not task["saved"]
                and global_frame_idx >= task["end_global_idx"]
            ):
                extracted_frames = [
                    f
                    for idx, f_num, f in frame_buffer
                    if task["start_global_idx"] <= idx <= task["end_global_idx"]
                ]

                if len(extracted_frames) > 0:
                    b_name = task["behavior"]
                    t_frame = task["trigger_frame"]

                    video_filename = os.path.join(
                        behavior_dirs[b_name],
                        f"event_trigger_{t_frame}.mp4"
                    )

                    out_video = cv2.VideoWriter(
                        video_filename,
                        fourcc,
                        20.0,
                        (320, 240)
                    )

                    for f in extracted_frames:
                        out_video.write(f)

                    out_video.release()

                    print(
                        f"💾 [影片分類成功] {b_name.upper()} "
                        f"影片已存入: {video_filename}"
                    )

                    task["saved"] = True

        video_tasks = [t for t in video_tasks if not t["saved"]]
        global_frame_idx += 1

        display_mask = (
            obj_mask
            if obj_mask.max() == 255
            else obj_mask * 255
        )
        display_mask_3ch = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)

        cv2.imshow("1. Geometry Object Mask View (B&W)", display_mask)
        cv2.imshow("2. Ultimate Behavioral Monitor (RGB)", annotated_frame)
        cv2.imshow("3. Foreground Debug View (Mask+RGB)", bg_remove_render)
        cv2.imshow("4. Clean Behavioral Monitor (RGB)", clean_frame)

        out_v1.write(display_mask_3ch)
        out_v2.write(annotated_frame)
        out_v3.write(bg_remove_render)
        out_v4.write(clean_frame)

        has_event = any(frame_triggers.values())
        if save_mode == "all":
            cv2.imwrite(
                os.path.join(output_dir, f"behavior_{frame_num}.jpg"),
                annotated_frame
            )
        elif save_mode == "event" and has_event:
            cv2.imwrite(
                os.path.join(output_dir, f"event_behavior_{frame_num}.jpg"),
                annotated_frame
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for task in video_tasks:
        if not task["saved"]:
            extracted_frames = [
                f
                for idx, f_num, f in frame_buffer
                if task["start_global_idx"] <= idx
            ]

            if len(extracted_frames) > 0:
                b_name = task["behavior"]
                t_frame = task["trigger_frame"]

                video_filename = os.path.join(
                    behavior_dirs[b_name],
                    f"event_trigger_{t_frame}_end.mp4"
                )

                out_video = cv2.VideoWriter(
                    video_filename,
                    fourcc,
                    20.0,
                    (320, 240)
                )

                for f in extracted_frames:
                    out_video.write(f)

                out_video.release()

    out_v1.release()
    out_v2.release()
    out_v3.release()
    out_v4.release()

    cv2.destroyAllWindows()
    print("✨ 四視窗行為分析、全程錄影與事件影片分類輸出完成！")


# =========================================================
# Realtime API - 初始化與逐幀分析
# =========================================================
def init_behavior_system():
    pose_model = YOLO("yolo11n-pose.pt")
    person_tracker = SimpleGeometryTracker(max_lost=25)
    object_tracker = SimpleGeometryTracker(max_lost=15)
    analyzer = BehaviorAnalyzer()

    state = {
        "BASE_COMPENSATION": 10,
        "MAX_COMPENSATION": 150,
        "TRUST_ACCUMULATION_RATE": 0.5,
        "last_object_boxes": [],
        "object_missing_count": 0,
        "object_alive_frames_dict": {}
    }

    return {
        "pose_model": pose_model,
        "person_tracker": person_tracker,
        "object_tracker": object_tracker,
        "analyzer": analyzer,
        "state": state
    }


def analyze_one_frame(frame, mask, behavior_pack):
    pose_model = behavior_pack["pose_model"]
    person_tracker = behavior_pack["person_tracker"]
    object_tracker = behavior_pack["object_tracker"]
    analyzer = behavior_pack["analyzer"]
    state = behavior_pack["state"]

    if frame is None or mask is None:
        raise ValueError("frame 或 mask 是 None")

    ori_img = cv2.resize(frame, (320, 240))

    if len(mask.shape) == 3:
        raw_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        raw_mask = mask.copy()

    raw_mask = cv2.resize(
        raw_mask,
        (320, 240),
        interpolation=cv2.INTER_NEAREST
    )

    obj_mask = postprocess_mask_for_object(
        raw_mask,
        kernel_size=3,
        min_area=20,
        border_ignore=10
    )

    kernel = np.ones((3, 3), np.uint8)

    person_bgs_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    person_bgs_mask = cv2.morphologyEx(
        person_bgs_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    foreground_only = cv2.bitwise_and(
        ori_img,
        ori_img,
        mask=person_bgs_mask
    )

    annotated_frame = ori_img.copy()
    bg_remove_render = foreground_only.copy()
    clean_frame = ori_img.copy() 

    # 🔄【改回最穩定的寫法】Realtime API 偵測分支同步將輸入改回原始有背景的「ori_img」
    pose_results = pose_model(
        ori_img,
        verbose=False,
        conf=0.3
    )

    if len(pose_results[0].boxes) > 0:
        pose_results[0].orig_img = annotated_frame
        annotated_frame = pose_results[0].plot(boxes=False)

        pose_results[0].orig_img = bg_remove_render
        bg_remove_render = pose_results[0].plot(boxes=False)

    detected_people_boxes = []
    detected_keypoints_list = []

    if len(pose_results[0].boxes) > 0:
        all_boxes = pose_results[0].boxes

        if pose_results[0].keypoints is not None:
            all_kpts_data = (
                pose_results[0]
                .keypoints
                .xy
                .cpu()
                .numpy()
                )
        else:
            all_kpts_data = None

        for idx, box in enumerate(all_boxes):
            if int(box.cls[0]) == 0:
                px1, py1, px2, py2 = map(
                    int,
                    box.xyxy[0]
                )

                detected_people_boxes.append(
                    (
                        px1,
                        py1,
                        px2 - px1,
                        py2 - py1
                    )
                )

                if (
                    all_kpts_data is not None
                    and idx < len(all_kpts_data)
                ):
                    detected_keypoints_list.append(all_kpts_data[idx])
                else:
                    detected_keypoints_list.append(None)
    frame_triggers = {
        "running": False,
        "loiter": False,
        "faint": False,
        "collision": False,
        "litter": False
    }
    active_people = person_tracker.update(detected_people_boxes)

    for pid, bbox in active_people.items():
        px, py, pw, ph = bbox
        matched_kpts = None
        best_iou = 0.5

        for idx, d_box in enumerate(detected_people_boxes):
            iou = calculate_iou(bbox, d_box)
            if iou > best_iou:
                best_iou = iou
                matched_kpts = detected_keypoints_list[idx]

        analyzer.update_person(pid, bbox, keypoints=matched_kpts)

        is_run, speed = analyzer.check_running(pid)
        is_loiter = analyzer.check_loitering(pid)
        is_fall = analyzer.check_fall(pid)

        for view in [annotated_frame, bg_remove_render, clean_frame]:
            cv2.rectangle(view, (px, py), (px + pw, py + ph), (255, 255, 255), 2)
            cv2.putText(
                view,
                f"ID:{pid}",
                (px, max(py - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

            if is_fall:
                cv2.putText(
                    view,
                    " ALERT: FAINT!",
                    (px - 10, py + ph + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    2
                )
                frame_triggers["faint"] = True
            elif is_run:
                cv2.putText(
                    view,
                    f"RUNNING ({int(speed)}%)",
                    (px, py - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )
                frame_triggers["running"] = True

            if is_loiter:
                cv2.putText(
                    view,
                    "LOITERING",
                    (px + pw - 65, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 255),
                    1
                )
                frame_triggers["loiter"] = True

    pids = list(active_people.keys())

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if analyzer.check_collision(pids[i], pids[j]):
                frame_triggers["collision"] = True
                bx1, by1, _, _ = active_people[pids[i]]
                bx2, by2, _, _ = active_people[pids[j]]

                for view in [annotated_frame, bg_remove_render, clean_frame]:
                    cv2.putText(
                        view,
                        " COLLISION!",
                        (((bx1 + bx2) // 2), ((by1 + by2) // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 75, 255),
                        2
                    )

    num_o, labels_o, stats_o, _ = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    current_frame_object_boxes = []

    for i in range(1, num_o):
        x = stats_o[i, 0]
        y = stats_o[i, 1]
        w = stats_o[i, 2]
        h = stats_o[i, 3]
        area = stats_o[i, 4]

        if area < 15 or area > 450:
            continue
        if w < 3 or h < 3:
            continue

        if (x <= 15 or y <= 15 or x + w >= 305 or y + h >= 225) and (
            h >= 50 or (h / max(w, 1)) >= 1.4 or w <= 8 or h <= 8
        ):
            continue

        if not any(box_inside_or_overlap((x, y, w, h), b, 0.8) for b in detected_people_boxes):
            current_frame_object_boxes.append((x, y, w, h))

    BASE_COMPENSATION = state["BASE_COMPENSATION"]
    MAX_COMPENSATION = state["MAX_COMPENSATION"]
    TRUST_ACCUMULATION_RATE = state["TRUST_ACCUMULATION_RATE"]
    object_alive_frames_dict = state["object_alive_frames_dict"]

    if len(current_frame_object_boxes) > 0:
        state["last_object_boxes"] = current_frame_object_boxes
        state["object_missing_count"] = 0

        active_objects = object_tracker.update(current_frame_object_boxes)

        for oid in active_objects.keys():
            object_alive_frames_dict[oid] = object_alive_frames_dict.get(oid, 0) + 1

        for oid, obbox in active_objects.items():
            analyzer.update_object(oid, obbox)
            is_litter = analyzer.check_littering(oid)
            ox, oy, ow, oh = obbox

            if is_litter:
                frame_triggers["litter"] = True
                assigned_pid = getattr(analyzer, "object_owner_memory", {}).get(oid)
                if assigned_pid is not None and assigned_pid in active_people:
                    px, py, pw, ph = active_people[assigned_pid]
                    for view in [annotated_frame, bg_remove_render, clean_frame]:
                        cv2.rectangle(view, (px, py), (px + pw, py + ph), (0, 0, 255), 3)
                        cv2.putText(view, f"LITTERER CAUGHT! (ID:{assigned_pid})", 
                                    (px, max(py - 22, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                for view in [annotated_frame, bg_remove_render, clean_frame]:
                    cv2.rectangle(view, (ox, oy), (ox + ow, oy + oh), (0, 0, 255), 2)
                    cv2.putText(view, f"LITTER OBJ:{oid}", (ox, max(oy - 5, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                for view in [annotated_frame, bg_remove_render, clean_frame]:
                    cv2.rectangle(view, (ox, oy), (ox + ow, oy + oh), (0, 255, 255), 1)
                    cv2.putText(view, f"obj:{oid}", (ox, max(oy - 5, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    else:
        state["object_missing_count"] += 1
        object_missing_count = state["object_missing_count"]
        valid_compensation_boxes = []

        for oid, alive_count in list(object_alive_frames_dict.items()):
            if alive_count >= 5:
                allowed_comp = min(
                    int(BASE_COMPENSATION + (alive_count * TRUST_ACCUMULATION_RATE)),
                    MAX_COMPENSATION
                )

                if object_missing_count <= allowed_comp:
                    if oid in object_tracker.tracked_objects:
                        tx, ty, tw, th, _ = object_tracker.tracked_objects[oid]
                        valid_compensation_boxes.append((oid, (tx, ty, tw, th), allowed_comp))
                else:
                    object_alive_frames_dict.pop(oid, None)
            else:
                object_alive_frames_dict.pop(oid, None)

        if len(valid_compensation_boxes) > 0:
            for oid, box, allowed_comp in valid_compensation_boxes:
                lox, loy, low, loh = box
                for view in [annotated_frame, bg_remove_render, clean_frame]:
                    cv2.rectangle(view, (lox, loy), (lox + low, loy + loh), (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(view, f"Tracking Lost ({object_missing_count}/{allowed_comp})", 
                                (lox, max(loy - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
        else:
            state["last_object_boxes"] = []
            object_alive_frames_dict.clear()

    analyzer.clear_dead_tracks(
        list(active_people.keys()),
        list(object_tracker.tracked_objects.keys())
    )

    display_mask = obj_mask if obj_mask.max() == 255 else obj_mask * 255
    display_mask_3ch = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)

    return {
        "view1_box_only": clean_frame.copy(),
        "view2_skeleton": annotated_frame.copy(),
        "view3_mask_overlay": bg_remove_render.copy(),
        "view4_mask": display_mask_3ch.copy(),
        "view5_original": ori_img.copy(),
        "frame_triggers": frame_triggers
    }


if __name__ == "__main__":
    main(
        base_dir="D:/subgarbage",
        input_dir="input",
        mask_dir="maskpicture",
        output_dir="11111/combined_results",
        save_mode="event"
    )