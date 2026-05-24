import os
import re
import glob
import cv2
import numpy as np
from PIL import Image
from function_monitor import PerformanceMonitor
import test_new_HyRGB
import finalsecond


def picture_name(base_dir, data_dir):

    image_dir = os.path.join(base_dir, data_dir)

    image_files = []

    extensions = ["*.jpg", "*.png", "*.jpeg"]

    for ext in extensions:

        image_files.extend(
            glob.glob(
                os.path.join(image_dir, ext)
            )
        )

    image_files.sort()

    print(f"找到圖片數量: {len(image_files)} 張")

    if len(image_files) == 0:

        raise FileNotFoundError(
            f"找不到圖片: {image_dir}"
        )

    return image_files

def new_hyrgb_run(
    base_dir,
    data_dir,
    roi_model_dir,
    input_picture,
    mask_dir="result_mask",
    save_mode="none",
    hyrgb_params=None
):
    if hyrgb_params is None:
        hyrgb_params = {}

    test_new_HyRGB.run_hyrgb(
        base_dir=base_dir,
        data_dir=data_dir,
        roi_model_dir=roi_model_dir,
        input_picture=input_picture,
        ghz_mask_dir=mask_dir,
        save_mode=save_mode,
        **hyrgb_params
    )

def behavior_analyzer(base_dir,input_dir,mask_dir,output_dir):
    finalsecond.main(base_dir,input_dir,mask_dir,output_dir,save_mode = "all")
#mode介紹
#"none"   # 完全不存，只顯示(只適合realtime_run)
#"all"    # 每幀都存
#"event"  # 只有事件時存(只適合realtime_run)

def run_streaming_dataset(
    base_dir,
    data_dir,
    roi_model_dir,
    output_dir,
    save_mode="all",
    hyrgb_params=None,
    result_callback=None,
    stop_event=None
):
    if hyrgb_params is None:
        hyrgb_params = {}

    input_picture = picture_name(base_dir, data_dir)
    background_path = os.path.join(base_dir, "background.jpg")

    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Missing background image: {background_path}")

    os.makedirs(output_dir, exist_ok=True)

    calibration_frames = []

    for img_path in input_picture[:30]:
        if os.path.exists(img_path):
            calibration_frames.append(
                Image.open(img_path).convert("RGB")
            )

    hyrgb_system, bg_img = test_new_HyRGB.init_hyrgb_system(
        base_dir=base_dir,
        roi_model_dir=roi_model_dir,
        background_path=background_path,
        calibration_frames=calibration_frames,
        scene_id=data_dir,
        **hyrgb_params
    )

    behavior_pack = finalsecond.init_behavior_system()

    for frame_idx, img_path in enumerate(input_picture):
        if stop_event is not None and stop_event.is_set():
            break

        frame = cv2.imread(img_path)

        if frame is None:
            continue

        frame = cv2.resize(frame, (320, 240))

        mask_img, active_count = test_new_HyRGB.infer_one_frame(
            hyrgb_system,
            bg_img,
            frame
        )

        mask_np = np.array(mask_img)

        if len(mask_np.shape) == 3:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

        result_views = finalsecond.analyze_one_frame(
            frame,
            mask_np,
            behavior_pack
        )

        result = {
            "frame_idx": frame_idx,
            "image_path": img_path,
            "view1_box_only": result_views["view1_box_only"],
            "view2_skeleton": result_views["view2_skeleton"],
            "view3_mask_overlay": result_views["view3_mask_overlay"],
            "view4_mask": result_views["view4_mask"],
            "view5_original": result_views["view5_original"],
            "frame_triggers": result_views["frame_triggers"]
        }

        if result_callback is not None:
            result_callback(result)

        has_event = any(result["frame_triggers"].values())

        if save_mode == "all" or (save_mode == "event" and has_event):
            name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{name}_behavior.jpg")
            cv2.imwrite(save_path, result["view2_skeleton"])

    return True


def run(
    base_dir,
    dataset,
    roi_model_dir,
    behavior_analyzer_output_dir,
    mode = "all",
    hyrgb_params=None
):
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    for name in dataset:

        print(f"\n開始處理 : {name}")

        frame_id = picture_name(
            base_dir,
            name
        )

        new_hyrgb_run(
            base_dir,
            name,
            roi_model_dir,
            input_picture=frame_id,
            mask_dir=f"{name}_result_mask",
            save_mode=mode,
            hyrgb_params=hyrgb_params
        )

        behavior_analyzer(base_dir=base_dir,
                        input_dir=name,
                        mask_dir=f"maskpicture/{name}_result_mask",
                        output_dir=behavior_analyzer_output_dir)


if __name__ == "__main__":
    base_dir=r"C:\Users\Asus\Downloads\videotrain\Q_test\5_23env"#基礎路徑
    roi_model_dir=r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\new_batch_path"#權重擋路徑
    behavior_analyzer_output_dir="11111/combined_results"#行為分析輸出資料夾位置
    #輸入資料夾名字放dataset
    dataset = ["pic_obalanuwalk"]
    hyrgb_params = {
        "varThreshold": 24,
        "active_ratio_threshold": 0.35,
        "active_indices_threshold": 225,
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

    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    monitor = PerformanceMonitor (save_path=os.path.join(
        log_dir,
        "main_run_performance_log.txt"
    ),
                                interval = 1.0)
    monitor.start()
    #mode介紹
    #"none"   # 完全不存，只顯示(只適合realtime_run)
    #"all"    # 每幀都存
    #"event"  # 只有事件時存(只適合realtime_run)
    try:
        run(
            base_dir=base_dir,
            dataset=dataset,
            roi_model_dir=roi_model_dir,
            behavior_analyzer_output_dir=behavior_analyzer_output_dir,
            mode="all",
            hyrgb_params=hyrgb_params
        )
    finally:
        monitor.stop()
    print("引用成功")
