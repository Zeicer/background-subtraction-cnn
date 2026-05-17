import os
import cv2
import torch
import numpy as np
import time
import glob
import re
import subprocess
from PIL import Image
from ultralytics import YOLO

# 官方命名：PerfectCombinedSystem_ObjectDualView
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 設備啟動: {device}")

    base_dir = "D:/subgarbage"
    input_dir = os.path.join(base_dir, "input")
    mask_dir = os.path.join(base_dir, "maskpicture")
    
    # 彩色成果與黑白遮罩的輸出資料夾
    output_dir = os.path.join(base_dir, "11111/combined_results")
    mask_output_dir = os.path.join(base_dir, "11111/mask_results")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # =========================================================================
    # 🌟 背景自動執行前置任務 (new_HyRGB.py)
    # =========================================================================
    print("⏳ [ObjectDualView] 正在背景自動執行 new_HyRGB.py 以生成高品質遮罩...")
    python_exe = os.path.abspath(torch.__file__).split("lib")[0] + "python.exe"
    if not os.path.exists(python_exe):
        import sys
        python_exe = sys.executable

    hy_rgb_script = os.path.join(base_dir, "new_HyRGB.py")
    try:
        subprocess.run([python_exe, hy_rgb_script], check=True)
        print("✅ [ObjectDualView] 前置遮罩生成完畢！開始進行人體與物品整合推論...")
    except subprocess.CalledProcessError as e:
        print(f"❌ 前置腳本執行失敗，嘗試直接讀取現有遮罩... 錯誤代碼: {e}")

    # =========================================================================
    # 載入 YOLO11-Pose (它除了骨架外，也能看得到常見的物品如背包、瓶子、雨傘等)
    print("🚀 載入 YOLO11-Pose 模型...")
    pose_model = YOLO("yolo11n-pose.pt")

    # 讀取影像序列 (從第 100 幀開始)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*_aligned.jpg")))[100:]
    print(f"📊 開始雙視窗推論，共 {len(image_paths)} 幀")

    # 設定雙視窗並排
    cv2.namedWindow("YOLO Pose & Object Result", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("CNN Mask Result", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("YOLO Pose & Object Result", 100, 200) # 左邊：原圖+骨架+物品框
    cv2.moveWindow("CNN Mask Result", 450, 200)          # 右邊：純黑白遮罩

    for path in image_paths:
        filename = os.path.basename(path)
        frame_num = re.search(r"frame_(\d+)", filename).group(1)
        
        # 1. 讀取與縮放原始彩色圖
        ori_img = cv2.imread(path)
        ori_img = cv2.resize(ori_img, (320, 240))
        
        # 2. 讀取 CNN 黑白遮罩
        mask_path = os.path.join(mask_dir, f"pred_mask_{frame_num}.png")
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_NEAREST)

        # 3. 遮罩優化處理 
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. 刷黑背景：餵給 YOLO (包含人影跟隨身物品的前景都會保留)
        foreground_only = cv2.bitwise_and(ori_img, ori_img, mask=mask)

        # 5. 讓 YOLO 在全黑圖上做偵測 (人體信心度 0.355)
        results = pose_model(foreground_only, verbose=False, conf=0.355)
        
        # 6. 🛠️ 【核心變動】將骨架與物品偵測框，同時畫回「原始彩色原圖」上
        annotated_frame = ori_img.copy()
        
        if len(results[0].boxes) > 0:
            # 抽換畫布為原始彩色圖
            results[0].orig_img = annotated_frame
            
            # 💡 關鍵設定：boxes=True 代表除了畫人體骨架外，也強迫 YOLO 畫出偵測到的物體邊框與標籤！
            annotated_frame = results[0].plot(boxes=True, labels=True, conf=True)

        # 7. 處理右邊視窗的純黑白顯示
        if mask.max() == 1:
            display_mask = mask * 255
        else:
            display_mask = mask

        # 顯示整合物品判斷後的雙視窗結果
        cv2.imshow("YOLO Pose & Object Result", annotated_frame) # 左邊：彩色原圖 + 骨架 + 物品框
        cv2.imshow("CNN Mask Result", display_mask)               # 右邊：純黑白遮罩

        # 分別儲存成果與黑白遮罩
        cv2.imwrite(os.path.join(output_dir, f"pose_obj_{frame_num}.jpg"), annotated_frame)
        cv2.imwrite(os.path.join(mask_output_dir, f"mask_{frame_num}.png"), display_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("✨ [ObjectDualView] 任務完成！物品判斷與骨架已成功疊加並顯示。")

if __name__ == "__main__":
    main()