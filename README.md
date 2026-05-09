# background-subtraction-cnn
使用 CNN 進行影像序列背景減除的專題專案

# Dataset Folder Structure

以下展示了資料集的格式

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
│  └─ Rain_patch/
│     ├─ train/
│     │  ├─ 000000_000.npy
│     │  ├─ 000000_001.npy
│     │  └─ ...
│     └─ valid/
│        ├─ 000001_000.npy
│        ├─ 000001_001.npy
│        └─ ...
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
│  └─ lagi_patch/
│     ├─ train/
│     │  ├─ 000000_000.npy
│     │  ├─ 000000_001.npy
│     │  └─ ...
│     └─ valid/
│        ├─ 000001_000.npy
│        ├─ 000001_001.npy
│        └─ ...
│
└─ baseline/
   ├─ input/
   ├─ background.jpg
   ├─ groundtruth/
   └─ patches/
```
## Notes
```text
patch 的命名規則是「影格編號_第幾份 patch.npy」
lagi的groundtruth為從zecplusgg_re_code.py中的pseudo_masks獲取
Large image files, `.npy`, `.pt`, and model weights are not uploaded to GitHub.
```
