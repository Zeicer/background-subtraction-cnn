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
    background_path=None,
    save_mode="all",
    hyrgb_params=None,
    result_callback=None,
    progress_callback=None,
    stop_event=None
):
    if hyrgb_params is None:
        hyrgb_params = {}

    input_picture = picture_name(base_dir, data_dir)

    if background_path is None:
        background_path = os.path.join(base_dir, "background.jpg")

    if not os.path.exists(background_path):
        raise FileNotFoundError(f"Missing background image: {background_path}")

    os.makedirs(output_dir, exist_ok=True)
    mask_save_dir = os.path.join(
        base_dir,
        "maskpicture",
        f"{data_dir}_result_mask"
    )
    mask_video_dir = os.path.join(
        base_dir,
        "maskpicture",
        "mask_video"
    )
    save_1111_dir = os.path.join(base_dir, "1111")
    behavior_main_dir = os.path.join(
        base_dir,
        "11111",
        "fivebehavior_videos"
    )
    combined_results_dir = os.path.join(
        base_dir,
        "11111",
        "combined_results"
    )
    behavior_dirs = {
        "running": os.path.join(behavior_main_dir, "RUNNING"),
        "loiter": os.path.join(behavior_main_dir, "LOITERING"),
        "faint": os.path.join(behavior_main_dir, "FAINT"),
        "collision": os.path.join(behavior_main_dir, "COLLISION"),
        "litter": os.path.join(behavior_main_dir, "LITTERING")
    }
    save_dirs = {
        "view1_box_only": os.path.join(output_dir, "view1_box_only"),
        "view2_skeleton": os.path.join(output_dir, "view2_skeleton"),
        "view3_mask_overlay": os.path.join(output_dir, "view3_mask_overlay"),
        "view4_mask": os.path.join(output_dir, "view4_mask"),
        "view5_original": os.path.join(output_dir, "view5_original"),
        "event": os.path.join(output_dir, "event")
    }

    for save_dir in [
        mask_save_dir,
        mask_video_dir,
        save_1111_dir,
        combined_results_dir,
        *behavior_dirs.values(),
        *save_dirs.values()
    ]:
        os.makedirs(save_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_size = (320, 240)
    mask_video = cv2.VideoWriter(
        os.path.join(mask_video_dir, "mask_video.mp4"),
        fourcc,
        10.0,
        video_size
    )
    out_v1 = cv2.VideoWriter(
        os.path.join(save_1111_dir, "1_geometry_mask.mp4"),
        fourcc,
        20.0,
        video_size
    )
    out_v2 = cv2.VideoWriter(
        os.path.join(save_1111_dir, "2_ultimate_monitor.mp4"),
        fourcc,
        20.0,
        video_size
    )
    out_v3 = cv2.VideoWriter(
        os.path.join(save_1111_dir, "3_foreground_debug.mp4"),
        fourcc,
        20.0,
        video_size
    )
    out_v4 = cv2.VideoWriter(
        os.path.join(save_1111_dir, "4_clean_monitor.mp4"),
        fourcc,
        20.0,
        video_size
    )

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

    try:
        for frame_idx, img_path in enumerate(input_picture):
            if stop_event is not None and stop_event.is_set():
                break

            frame = cv2.imread(img_path)

            if frame is None:
                continue

            frame = cv2.resize(frame, (320, 240))
            filename = os.path.basename(img_path)
            name = os.path.splitext(filename)[0]

            mask_img, active_count = test_new_HyRGB.infer_one_frame(
                hyrgb_system,
                bg_img,
                frame
            )

            mask_np = np.array(mask_img)

            if len(mask_np.shape) == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

            mask_path = os.path.join(mask_save_dir, f"{name}_mask.png")

            if save_mode in ("all", "event"):
                Image.fromarray(mask_np).save(mask_path)

            mask_video.write(cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR))

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

            out_v1.write(result["view4_mask"])
            out_v2.write(result["view2_skeleton"])
            out_v3.write(result["view3_mask_overlay"])
            out_v4.write(result["view1_box_only"])

            if result_callback is not None:
                result_callback(result)

            if progress_callback is not None:
                progress_callback(frame_idx + 1, len(input_picture))

            has_event = any(result["frame_triggers"].values())

            if save_mode == "all" or (save_mode == "event" and has_event):
                if save_mode == "all":
                    for view_key, save_dir in save_dirs.items():
                        if view_key == "event":
                            continue

                        cv2.imwrite(
                            os.path.join(save_dir, f"{name}.jpg"),
                            result[view_key]
                        )

                    cv2.imwrite(
                        os.path.join(output_dir, f"behavior_{name}.jpg"),
                        result["view2_skeleton"]
                    )
                    cv2.imwrite(
                        os.path.join(
                            combined_results_dir,
                            f"behavior_{name}.jpg"
                        ),
                        result["view2_skeleton"]
                    )

                if has_event:
                    cv2.imwrite(
                        os.path.join(save_dirs["event"], f"{name}_event.jpg"),
                        result["view2_skeleton"]
                    )
                    cv2.imwrite(
                        os.path.join(output_dir, f"event_behavior_{name}.jpg"),
                        result["view2_skeleton"]
                    )
                    cv2.imwrite(
                        os.path.join(
                            combined_results_dir,
                            f"event_behavior_{name}.jpg"
                        ),
                        result["view2_skeleton"]
                    )

                    for behavior_name, triggered in result["frame_triggers"].items():
                        if triggered and behavior_name in behavior_dirs:
                            cv2.imwrite(
                                os.path.join(
                                    behavior_dirs[behavior_name],
                                    f"{name}_{behavior_name}.jpg"
                                ),
                                result["view2_skeleton"]
                            )

    finally:
        mask_video.release()
        out_v1.release()
        out_v2.release()
        out_v3.release()
        out_v4.release()

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
