import os
import re
import glob
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
    mask_dir="result_mask"
):

    test_new_HyRGB.run_hyrgb(
        base_dir=base_dir,
        data_dir=data_dir,
        roi_model_dir=roi_model_dir,
        input_picture=input_picture,
        ghz_mask_dir=mask_dir
    )

def behavior_analyzer(base_dir,input_dir,mask_dir,output_dir):
    finalsecond.main(base_dir,input_dir,mask_dir,output_dir)


def run(
    base_dir,
    dataset,
    roi_model_dir,
    behavior_analyzer_output_dir
):

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
            mask_dir=f"{name}_result_mask"
        )

        behavior_analyzer(base_dir=base_dir,
                        input_dir=name,
                        mask_dir=f"maskpicture/{name}_result_mask",
                        output_dir=behavior_analyzer_output_dir)


if __name__ == "__main__":
    #輸入資料夾名字放dataset
    dataset = ["input"]

    run(
        base_dir=r"C:\Users\Asus\Downloads\videotrain\Q_test\5_23env",#基礎路徑
        dataset=dataset,
        roi_model_dir=r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\batch_path",#權重擋路徑
        behavior_analyzer_output_dir="11111/combined_results"#行為分析輸出資料夾位置
    )

    print("引用成功")