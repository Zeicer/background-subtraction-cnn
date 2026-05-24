import os
import cv2
import numpy as np
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


def collect_calibration_frames(
    cap,
    count=30,
    width=320,
    height=240
):
    frames = []

    print("正在收集背景校準影格，請先讓鏡頭畫面保持乾淨...")

    while len(frames) < count:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

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


def main():

    base_dir = r"C:\Users\Asus\Downloads\videotrain\Q_test\5_23env"

    roi_model_dir = (
        r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\batch_path"
    )

    background_path = os.path.join(
        base_dir,
        "background.jpg"
    )

    width = 320
    height = 240

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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    calibration_frames = collect_calibration_frames(
        cap,
        count=30,
        width=width,
        height=height
    )

    hyrgb_system, bg_img = test_new_HyRGB.init_hyrgb_system(
        base_dir=base_dir,
        roi_model_dir=roi_model_dir,
        background_path=background_path,
        calibration_frames=calibration_frames,
        scene_id="camera_realtime"
    )

    behavior_pack = finalsecond.init_behavior_system()

    print("即時系統啟動，按 q 離開")

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("讀取鏡頭失敗")
            break

        frame = cv2.resize(frame, (width, height))

        # =====================================================
        # 1. HyRGB 即時產生 mask
        # 不存 mask，只存在記憶體
        # =====================================================

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

        # =====================================================
        # 2. finalsecond 單幀行為分析
        # =====================================================

        annotated_frame, obj_mask, bg_remove_render, frame_triggers = (
            finalsecond.analyze_one_frame(
                frame,
                mask_np,
                behavior_pack
            )
        )

        has_event = any(frame_triggers.values())

        # =====================================================
        # 3. 只有事件發生時才存圖
        # =====================================================

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

        # save_mode == "none" 時，不存任何圖

        # =====================================================
        # 4. 即時顯示
        # =====================================================

        cv2.imshow(
            "1. Realtime Behavior Monitor",
            annotated_frame
        )

        cv2.imshow(
            "2. Realtime Mask",
            mask_np
        )

        cv2.imshow(
            "3. Realtime Foreground Debug",
            bg_remove_render
        )

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("即時系統已關閉")


if __name__ == "__main__":
    main()