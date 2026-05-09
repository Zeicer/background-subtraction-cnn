import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import time
import glob
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random


# 種子（結果可重現）
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 定義完整的 background subtraction 模型架構
class BackgroundSubtractorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.Conv = nn.Sequential(
            # 改成單通道輸入
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),  # 27→27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),                 # 27→9


            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),  # 9→9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)                  # 9→3
        )
        
        # 動態算 flatten size，避免手算錯
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 27, 27)
            x = self.Conv(dummy)           # [1,16,3,3]
            flat = x.view(1, -1).size(1)   # 16*3*3=144


        self.Classes = nn.Sequential(
            nn.Linear(flat, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, num_classes)    # 用 CrossEntropy → num_classes=2
        )


    def forward(self, input):
        x = self.Conv(input)
        x = x.view(x.size(0), -1)          # [N, flat]
        x = self.Classes(x)                # [N,2]
        return x

class PatchDataset(Dataset):
    def __init__(self, data_dir, split_name, max_samples=None, cache_size=500):
        self.data_dir = os.path.join(data_dir, split_name)
        self.npy_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))

        self.index_map = []
        all_labels = []

        print("建立 index mapping...")

        for file_path in self.npy_files:
            data = np.load(file_path, allow_pickle=True).item()

            labels = data["labels"]
            num_patches = len(labels)

            all_labels.extend(labels)

            for i in range(num_patches):
                self.index_map.append((file_path, i))

        print("Dataset label 分佈:", np.bincount(all_labels))

        random.shuffle(self.index_map)

        if max_samples is not None and len(self.index_map) > max_samples:
            self.index_map = self.index_map[:max_samples]

        print(f"{split_name} patches: {len(self.index_map)}")

        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        if len(self.index_map) == 0:
            raise RuntimeError(f"{split_name} 沒有讀到任何 npy 檔，請檢查路徑: {self.data_dir}")

    def __len__(self):
        return len(self.index_map)

    def _get_npy(self, file_path):
        if file_path in self.cache:
            return self.cache[file_path]

        data = np.load(file_path, allow_pickle=True).item()

        self.cache[file_path] = data
        self.cache_order.append(file_path)

        if len(self.cache_order) > self.cache_size:
            old = self.cache_order.pop(0)
            del self.cache[old]

        return data

    def __getitem__(self, idx):
        file_path, patch_idx = self.index_map[idx]

        data = self._get_npy(file_path)

        patch = data["patches"][patch_idx]
        label = data["labels"][patch_idx]

        patch = torch.from_numpy(patch).float()
        label = torch.tensor(label, dtype=torch.long)

        return patch, label

def paper_init(module):
    """論文精確初始化: bias=0.1, weights~TruncatedNormal(0,0.01,±0.2)"""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        # Bias → 0.1
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)
        
        # Weights → Truncated N(0,0.01)
        for name, param in module.named_parameters():
            if 'bias' not in name:
                while True:  # 重抽直到 |w| ≤ 0.2
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                    if torch.max(torch.abs(param)) <= 0.2:
                        break

def get_data_loader(data_dir, split_name, batch_size=100):
    max_samples = 2560000 if split_name == "train" else 256000
    dataset = PatchDataset(data_dir, split_name, max_samples=max_samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split_name == "train"),  # train 才 shuffle
        num_workers=2,
        pin_memory=True,
    )

    return loader, dataset

# 模型訓練流程
def train_model(environment1 = 'Rain', environment2 = 'Rain_patch'):
    data_dir = f'./dataset/{environment1}/{environment2}'  # 注意路徑修正

    # 1. 兩個 DataLoader
    train_loader, train_dataset = get_data_loader(data_dir, "train", batch_size=100)
    valid_loader, valid_dataset = get_data_loader(data_dir, "valid", batch_size=100)
    
    # 2. 模型 + 最佳準確率追蹤
    model = BackgroundSubtractorCNN(num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.apply(paper_init)  # 遞迴套用到所有子模組
    model = model.to(device)
    print("✅初始化完成")
    print("使用裝置:", device)

    # labels = [label for _, label in train_dataset]
    # class_count = np.bincount(labels)

    # print("Train class count:", class_count)

    # 🔥 更穩版本（不要太極端）
    # weights = 1. / np.sqrt(class_count + 1e-6)
    # weights = weights / weights.sum()

    # weights = torch.tensor(weights, dtype=torch.float).to(device)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)
        
    loss_f = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_num = 50
    patience = 5  # 連續5輪存檔點沒改善就停
    patience_counter = 0
    best_f1 = 0.0
    stop_training = False
    save_dir = "./batch_pth"
    os.makedirs(save_dir, exist_ok=True)

    # batch 計數器
    total_batch_count = 0  # 全域 batch 計數
    check_every_batches = 10000 # 存檔點

    for epoch in range(epoch_num):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epoch_num}")
        
        # ========== Train Phase ==========
        model.train()
        epoch_train_loss = 0.0
        epoch_train_corrects = 0

        for batch_idx, (X, y) in enumerate(train_loader):
            # ★★★ 標準訓練 ★★★
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_f(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X.size(0)
            _, pred = torch.max(y_pred, 1)
            epoch_train_corrects += torch.sum(pred == y).item()

            total_batch_count += 1

            # ★★★ 每到存檔點完整驗證 + 儲存 + 早停 ★★★
            if total_batch_count % check_every_batches == 0:
                print(f"\n[檢查點] 總 batch {total_batch_count}")
                
                # 完整 Valid（移到這裡！）
                model.eval()
                valid_loss, valid_corrects = 0.0, 0
                all_labels, all_preds = [], []
                
                with torch.no_grad():
                    for Xv, yv in valid_loader:
                        Xv, yv = Xv.to(device), yv.to(device)
                        y_pred_v = model(Xv)
                        loss_v = loss_f(y_pred_v, yv)
                        
                        valid_loss += loss_v.item() * Xv.size(0)
                        _, pred_v = torch.max(y_pred_v, 1)
                        valid_corrects += torch.sum(pred_v == yv).item()
                        
                        all_labels.append(yv.cpu())
                        all_preds.append(pred_v.cpu())
                
                all_labels = torch.cat(all_labels).numpy()
                all_preds = torch.cat(all_preds).numpy()

                print("GT 分佈:", np.bincount(all_labels))
                print("Pred 分佈:", np.bincount(all_preds))

                cm = confusion_matrix(all_labels, all_preds)
                print("Confusion Matrix:\n", cm)
                
                valid_loss /= len(valid_dataset)
                valid_acc = 100.0 * valid_corrects / len(valid_dataset)
                valid_f1 = f1_score(all_labels, all_preds, zero_division=0)
                
                print(f"Valid: Loss={valid_loss:.4f} Acc={valid_acc:.4f}% F1={valid_f1:.4f}")
                
                os.makedirs(save_dir, exist_ok=True)
                # ★ 保存 + 早停判斷 ★
                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_step{total_batch_count}_{environment1}_{environment2}.pth'))
                    patience_counter = 0
                    print(f"✅ 新最佳 F1: {best_f1:.4f} → 已保存")
                else:
                    patience_counter += 1
                    print(f"⏳ 耐心計數器: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("⛔ Early stopping triggered")
                    stop_training = True
                    break
                
                model.train()  # 切回訓練模式

            if stop_training:
                    break

        # epoch 統計
        epoch_train_loss /= len(train_dataset)
        epoch_train_acc = 100.0 * epoch_train_corrects / len(train_dataset)
        print(f"Epoch {epoch+1} Train: Loss={epoch_train_loss:.4f} Acc={epoch_train_acc:.4f}%")
        
        # ★ 早停後跳出 epoch 迴圈
        if patience_counter >= patience:
            break
        
        end_time = time.time()
        print(f"本次epoch運行{end_time-start_time}s")


    # 載入最佳模型
    model_paths = glob.glob(os.path.join(save_dir, f'best_model_step*_{environment1}_{environment2}.pth'))
    if model_paths:
        latest_model_path = max(model_paths, key=os.path.getctime)  # 最新修改時間
        model.load_state_dict(torch.load(latest_model_path))
        print(f"✅ 載入最新模型: {latest_model_path}")
    else:
        print("⚠️ 警告：沒有找到 best_model_step 檔案，使用當前模型")
    model.eval()

    # 測試評估 (類似驗證但不存歷史)
    valid_loss, valid_corrects = 0.0, 0
    all_valid_labels, all_valid_preds = [], []


    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_f(y_pred, y)

            valid_loss += loss.item() * X.size(0)
            _, pred = torch.max(y_pred, 1)
            valid_corrects += torch.sum(pred == y).item()

            all_valid_labels.append(y.cpu())
            all_valid_preds.append(pred.cpu())

    print(f"\n訓練完成! 最佳 Valid F1: {best_f1:.4f}%")


if __name__ == '__main__':
    # 開始模型訓練
    total_start_time = time.time()
    train_model()
    total_end_time = time.time()
    print(f"訓練總時長: {total_end_time-total_start_time}s")
