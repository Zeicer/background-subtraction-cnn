# background-subtraction-cnn
使用 CNN 進行影像序列背景減除的專題專案

# Dataset Folder Structure

This folder shows the expected dataset layout.

## Structure
```text
dataset/
├─ Rain/
│  ├─ input/
│  │  ├─ Rain_v2_1.jpg
│  │  ├─ Rain_v2_2.jpg
│  │  └─ ...
│  ├─ background.jpg
│  ├─ groundtruth/
│  │  ├─ Rain_v2_0250.bmp
│  │  ├─ Rain_v2_0251.bmp
│  │  └─ ...
│  └─ patches/
│     ├─ 000001_000.npy
│     ├─ 000002_001.npy
│     └─ ...
│
├─ lagi/
│  ├─ input/
│  │  ├─ frame_000000_aligned.jpg
│  │  ├─ frame_000001_aligned.jpg
│  │  └─ ...
│  ├─ background.jpg
│  ├─ groundtruth/
│  │  ├─ frame_000000_aligned_mask.png
│  │  ├─ frame_000001_aligned_mask.png
│  │  └─ ...
│  └─ patches/
│     ├─ 000000_000.npy
│     ├─ 000000_001.npy
│     └─ ...
│
└─ baseline/
   ├─ input/
   ├─ background/
   ├─ groundtruth/
   └─ patches/
```
## Notes

patch的命名規則是"照片張數_第幾份patch.npy"
lagi的groundtruth為從zecplusgg_re_code.py中的pseudo_masks獲取
Large image files, `.npy`, `.pt`, and model weights are not uploaded to GitHub.
