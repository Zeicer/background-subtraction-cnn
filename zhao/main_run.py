import os
import re
import glob
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
        "varThreshold": 30,
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
        "active_ratio_smooth": 0.4
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
