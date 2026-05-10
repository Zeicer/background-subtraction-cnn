from ultralytics import YOLO
import cv2
import numpy as np

mask_path = r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\maskpicture\pred_mask_000170.png"
img_path = r"C:\Users\Asus\Downloads\videotrain\Q_test\dataset\input\frame_000170_aligned.jpg"

model = YOLO("yolo11n-pose.pt")

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, 0)

if img is None:
    raise FileNotFoundError(f"讀不到原圖：{img_path}")

if mask is None:
    raise FileNotFoundError(f"讀不到 mask：{mask_path}")

mask = cv2.resize(
    mask,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

_, mask = cv2.threshold(
    mask,
    127,
    255,
    cv2.THRESH_BINARY
)

person_only = cv2.bitwise_and(
    img,
    img,
    mask=mask
)

cv2.imwrite("person_only.png", person_only)

results = model(person_only)

pose_img = results[0].plot()

cv2.imwrite("pose_result.png", pose_img)

print("✅ 已儲存 person_only.png 和 pose_result.png")

cv2.imshow("YOLO Pose", pose_img)
cv2.waitKey(0)
cv2.destroyAllWindows()