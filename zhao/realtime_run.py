import os
import cv2
import numpy as np
import queue
import threading
import time
from collections import deque
from datetime import datetime
from PIL import Image

import test_new_HyRGB
import finalsecond


def list_available_cameras(max_test=10):
    available = []

    for cam_id in range(max_test):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                available.append(cam_id)

        cap.release()

    return available


def choose_camera():
    cameras = list_available_cameras()

    if len(cameras) == 0:
        raise RuntimeError("找不到任何可用鏡頭")

    print("可用鏡頭：")

    for cam_id in cameras:
        print(f"[{cam_id}] Camera {cam_id}")

    selected = input("請輸入要使用的鏡頭編號：")

    try:
        selected = int(selected)
    except ValueError:
        raise ValueError("請輸入數字，例如 0 或 1")

    if selected not in cameras:
        raise ValueError(f"鏡頭 {selected} 不在可用清單中：{cameras}")

    return selected


def crop_stable_view(frame, margin=50):
    if margin <= 0:
        return frame

    h, w = frame.shape[:2]

    if h <= margin * 2 or w <= margin * 2:
        raise ValueError(
            f"Frame is too small for margin={margin}: {w}x{h}"
        )

    return frame[
        margin:h - margin,
        margin:w - margin
    ]


def prepare_model_frame(
    frame,
    width=320,
    height=240,
    crop_margin=50
):
    frame = crop_stable_view(
        frame,
        margin=crop_margin
    )

    return cv2.resize(
        frame,
        (width, height)
    )


def collect_calibration_frames(
    cap,
    count=30,
    width=320,
    height=240,
    crop_margin=50
):
    frames = []

    print("正在收集背景校準影格，請先讓鏡頭畫面保持乾淨...")

    while len(frames) < count:
        ret, frame = cap.read()

        if not ret:
            break

        frame = prepare_model_frame(
            frame,
            width=width,
            height=height,
            crop_margin=crop_margin
        )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_frame = Image.fromarray(frame_rgb).convert("RGB")

        frames.append(pil_frame)

        cv2.imshow("Calibration Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyWindow("Calibration Preview")

    print(f"校準影格收集完成: {len(frames)} 張")

    return frames


def ensure_background_image(
    cap,
    background_path,
    width=320,
    height=240,
    crop_margin=50,
    warmup_frames=10
):
    if os.path.exists(background_path):
        print(f"Use existing background image: {background_path}")
        return

    print("background.jpg not found.")
    input("Clear the scene, then press Enter to capture background.jpg...")

    background_frame = None

    for _ in range(warmup_frames):
        ret, frame = cap.read()

        if ret:
            background_frame = frame

    if background_frame is None:
        raise RuntimeError("Cannot capture background image from camera.")

    background_frame = prepare_model_frame(
        background_frame,
        width=width,
        height=height,
        crop_margin=crop_margin
    )

    os.makedirs(
        os.path.dirname(background_path),
        exist_ok=True
    )

    if not cv2.imwrite(background_path, background_frame):
        raise RuntimeError(
            f"Failed to save background image: {background_path}"
        )

    print(f"Saved background image: {background_path}")


def put_latest(frame_queue, item):
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass

    frame_queue.put_nowait(item)


def process_realtime_frame(
    frame_idx,
    frame,
    hyrgb_system,
    bg_img,
    behavior_pack
):
    mask_img, active_count = test_new_HyRGB.infer_one_frame(
        hyrgb_system,
        bg_img,
        frame
    )

    mask_np = np.array(mask_img)

    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(
            mask_np,
            cv2.COLOR_RGB2GRAY
        )

    result_views = finalsecond.analyze_one_frame(
        frame,
        mask_np,
        behavior_pack
    )

    return {
        "frame_idx": frame_idx,
        "view1_box_only": result_views["view1_box_only"],
        "view2_skeleton": result_views["view2_skeleton"],
        "view3_mask_overlay": result_views["view3_mask_overlay"],
        "view4_mask": result_views["view4_mask"],
        "view5_original": result_views["view5_original"],
        "frame_triggers": result_views["frame_triggers"]
    }


def save_event_frame(result, save_mode, event_dir):
    annotated_frame = result["view1_box_only"]
    frame_idx = result["frame_idx"]
    frame_triggers = result["frame_triggers"]
    has_event = any(frame_triggers.values())

    if save_mode == "all":
        save_path = os.path.join(
            event_dir,
            f"frame_{frame_idx:06d}.jpg"
        )

        cv2.imwrite(save_path, annotated_frame)

    elif save_mode == "event" and has_event:
        event_names = [
            name
            for name, triggered in frame_triggers.items()
            if triggered
        ]

        event_text = "_".join(event_names)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = os.path.join(
            event_dir,
            f"event_{event_text}_{timestamp}_frame_{frame_idx:06d}.jpg"
        )

        cv2.imwrite(save_path, annotated_frame)
        print(f"事件觸發，已存圖: {save_path}")


def inference_worker(
    input_queue,
    result_queue,
    stop_event,
    hyrgb_system,
    bg_img,
    behavior_pack,
    save_mode,
    event_dir
):
    while not stop_event.is_set() or not input_queue.empty():
        try:
            item = input_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        if item is None:
            break

        frame_idx, frame = item
        result = process_realtime_frame(
            frame_idx,
            frame,
            hyrgb_system,
            bg_img,
            behavior_pack
        )

        save_event_frame(result, save_mode, event_dir)
        put_latest(result_queue, result)


def main():

    base_dir = r"C:\Users\Asus\Downloads\videotrain\Q_test\5_23env"#基礎路徑

    roi_model_dir = (
        r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\batch_path"
    )#權重擋路徑

    background_path = os.path.join(
        base_dir,
        "background.jpg"
    )

    crop_margin = 50

    width = 320
    height = 240
    display_fps = 10
    display_delay_seconds = 0.5
    max_buffer_frames = max(
        2,
        int(display_fps * display_delay_seconds) + 2
    )

    hyrgb_params = {
        "varThreshold": 16,
        "active_ratio_threshold": 0.35,
        "active_indices_threshold": 200,
        "active_indices_limit": 8000,
        "learningRate": 0.001,
        "update_interval": 1,
        "batch_size": 2048,
        "auto_varThreshold": True,
        "varThreshold_min": 8,
        "varThreshold_max": 64,
        "varThreshold_step": 2,
        "active_ratio_spike": 0.20,
        "active_ratio_drop": 0.08,
        "active_ratio_smooth": 0.4,
        "foreground_threshold": 0.9
    }

    save_mode = "event"
    # 可選：
    # "none"  = 完全不存，只顯示
    # "event" = 只有事件發生時存圖
    # "all"   = 每一幀都存圖，不推薦

    event_dir = os.path.join(
        base_dir,
        "realtime_event_output"
    )

    os.makedirs(event_dir, exist_ok=True)

    camera_id = choose_camera()

    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟鏡頭 camera_id={camera_id}"
        )

    ensure_background_image(
        cap,
        background_path,
        width=width,
        height=height,
        crop_margin=crop_margin
    )

    calibration_frames = collect_calibration_frames(
        cap,
        count=30,
        width=width,
        height=height,
        crop_margin=crop_margin
    )

    hyrgb_system, bg_img = test_new_HyRGB.init_hyrgb_system(
        base_dir=base_dir,
        roi_model_dir=roi_model_dir,
        background_path=background_path,
        calibration_frames=calibration_frames,
        scene_id="camera_realtime",
        **hyrgb_params
    )

    behavior_pack = finalsecond.init_behavior_system()

    input_queue = queue.Queue(maxsize=max_buffer_frames)
    result_queue = queue.Queue(maxsize=max_buffer_frames)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=inference_worker,
        args=(
            input_queue,
            result_queue,
            stop_event,
            hyrgb_system,
            bg_img,
            behavior_pack,
            save_mode,
            event_dir
        ),
        daemon=True
    )
    worker.start()

    print("即時系統啟動，按 q 離開")

    frame_idx = 0
    output_buffer = deque(maxlen=max_buffer_frames)
    delay_frames = max(1, int(display_fps * display_delay_seconds))
    display_interval = 1.0 / display_fps
    last_display_time = 0.0
    last_result = None

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Cannot read camera frame.")
                break

            frame = prepare_model_frame(
                frame,
                width=width,
                height=height,
                crop_margin=crop_margin
            )

            put_latest(
                input_queue,
                (frame_idx, frame)
            )

            while True:
                try:
                    output_buffer.append(
                        result_queue.get_nowait()
                    )
                except queue.Empty:
                    break

            now = time.perf_counter()

            if now - last_display_time >= display_interval:
                if len(output_buffer) > delay_frames:
                    last_result = output_buffer.popleft()
                elif last_result is None and len(output_buffer) > 0:
                    last_result = output_buffer.popleft()

                if last_result is not None:
                    cv2.imshow(
                        "1. Boxes Only",
                        last_result["view1_box_only"]
                    )

                    cv2.imshow(
                        "2. Boxes + Skeleton",
                        last_result["view2_skeleton"]
                    )

                    cv2.imshow(
                        "3. Boxes + Skeleton + Mask",
                        last_result["view3_mask_overlay"]
                    )

                    cv2.imshow(
                        "4. Pure Mask",
                        last_result["view4_mask"]
                    )

                    cv2.imshow(
                        "5. Original Frame",
                        last_result["view5_original"]
                    )

                last_display_time = now

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        stop_event.set()
        put_latest(input_queue, None)
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()

    print("即時系統已關閉")
    return

    # print("即時系統啟動，按 q 離開")

    # frame_idx = 0

    # while True:
    #     ret, frame = cap.read()

    #     if not ret:
    #         print("讀取鏡頭失敗")
    #         break

    #     frame = prepare_model_frame(
    #         frame,
    #         width=width,
    #         height=height,
    #         crop_margin=crop_margin
    #     )

    #     # =====================================================
    #     # 1. HyRGB 即時產生 mask
    #     # 不存 mask，只存在記憶體
    #     # =====================================================

    #     mask_img, active_count = test_new_HyRGB.infer_one_frame(
    #         hyrgb_system,
    #         bg_img,
    #         frame
    #     )

    #     mask_np = np.array(mask_img)

    #     if len(mask_np.shape) == 3:
    #         mask_np = cv2.cvtColor(
    #             mask_np,
    #             cv2.COLOR_RGB2GRAY
    #         )

    #     # =====================================================
    #     # 2. finalsecond 單幀行為分析
    #     # =====================================================

    #     annotated_frame, obj_mask, bg_remove_render, frame_triggers = (
    #         finalsecond.analyze_one_frame(
    #             frame,
    #             mask_np,
    #             behavior_pack
    #         )
    #     )

    #     has_event = any(frame_triggers.values())

    #     # =====================================================
    #     # 3. 只有事件發生時才存圖
    #     # =====================================================

    #     if save_mode == "all":
    #         save_path = os.path.join(
    #             event_dir,
    #             f"frame_{frame_idx:06d}.jpg"
    #         )

    #         cv2.imwrite(save_path, annotated_frame)

    #     elif save_mode == "event" and has_event:

    #         event_names = [
    #             name
    #             for name, triggered in frame_triggers.items()
    #             if triggered
    #         ]

    #         event_text = "_".join(event_names)

    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #         save_path = os.path.join(
    #             event_dir,
    #             f"event_{event_text}_{timestamp}_frame_{frame_idx:06d}.jpg"
    #         )

    #         cv2.imwrite(save_path, annotated_frame)

    #         print(f"事件觸發，已存圖: {save_path}")

    #     # save_mode == "none" 時，不存任何圖

    #     # =====================================================
    #     # 4. 即時顯示
    #     # =====================================================

    #     cv2.imshow(
    #         "1. Realtime Behavior Monitor",
    #         annotated_frame
    #     )

    #     cv2.imshow(
    #         "2. Realtime Mask",
    #         mask_np
    #     )

    #     cv2.imshow(
    #         "3. Realtime Foreground Debug",
    #         bg_remove_render
    #     )

    #     frame_idx += 1

    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # print("即時系統已關閉")


if __name__ == "__main__":
    main()
