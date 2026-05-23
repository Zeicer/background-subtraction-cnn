import os
import cv2
import numpy as np
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

        frame = cv2.resize(
            frame,
            (width, height)
        )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        pil_frame = Image.fromarray(
            frame_rgb
        ).convert("RGB")

        frames.append(pil_frame)

        cv2.imshow(
            "Calibration Preview",
            frame
        )

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
    # camera_id = 0
    # cap = cv2.VideoCapture(camera_id)
    camera_id = choose_camera()
    cap = cv2.VideoCapture(camera_id,cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"無法開啟鏡頭，請確認 camera_id={camera_id} 是否正確"
        )

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

    while True:
        ret, frame = cap.read()

        if not ret:
            print("讀取鏡頭失敗")
            break

        frame = cv2.resize(
            frame,
            (width, height)
        )

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

        annotated_frame, obj_mask, bg_remove_render = finalsecond.analyze_one_frame(
            frame,
            mask_np,
            behavior_pack
        )

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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("即時系統已關閉")


if __name__ == "__main__":
    main()