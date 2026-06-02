import os
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf

# ==========================================
# 1. 讀取本機台積電數據 + 線上抓取美股/台股大盤
# ==========================================
csv_path = "c:/Users/user/Downloads/tsmc_5y_data.csv"
if not os.path.exists(csv_path):
    print(f"❌ 找不到台積電原始檔案！請確認 {csv_path} 是否存在。")
    exit()

print("--- [步驟 1] 讀取本機數據，並利用 yfinance 跨境捕捉美股、台股大盤數據 ---")
# 讀取本機台積電
tsmc_df = pd.read_csv(csv_path)

# 將民國年轉換為標準西元年，以便跟美股日期對齊 (例: 110/01/04 -> 2021/01/04)
def convert_date(date_str):
    parts = date_str.split('/')
    year = int(parts[0]) + 1911
    return f"{year}-{parts[1]}-{parts[2]}"

tsmc_df['Date'] = pd.to_datetime(tsmc_df['Date'].apply(convert_date))
tsmc_df = tsmc_df.sort_values('Date').reset_index(drop=True)

start_date = tsmc_df['Date'].min().strftime('%Y-%m-%d')
end_date = tsmc_df['Date'].max().strftime('%Y-%m-%d')

# 🎯 下載美股老大哥：費城半導體指數 (^SOX) & 台灣加權指數 (^TWII)
print(f"📡 正在從 Yahoo Finance 抓取 {start_date} 至 {end_date} 的大盤數據...")
sox = yf.download("^SOX", start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'SOX_Close'})
twii = yf.download("^TWII", start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'TWII_Close'})

# 清理 yfinance 帶有 MultiIndex 的欄位結構
sox.columns = ['SOX_Close']
twii.columns = ['TWII_Close']

# ==========================================
# 2. 🚀 多市場跨國特徵工程
# ==========================================
print("\n--- [步驟 2] 實作多市場聯動特徵工程 ---")

# (1) 計算昨晚美股費半指數的漲跌幅 (這對台積電是極強的先行指標！)
sox['SOX_Return_Yesterday'] = sox['SOX_Close'].pct_change().shift(1) # shift(1) 代表昨晚

# (2) 計算台股大盤的 5 日乖離率
twii['TWII_MA5'] = twii['TWII_Close'].rolling(window=5).mean()
twii['TWII_Bias5'] = (twii['TWII_Close'] - twii['TWII_MA5']) / twii['TWII_MA5']

# (3) 核心合併：把台積電、美股、大盤用「日期」黏在一起
merged_df = pd.merge(tsmc_df, twii[['TWII_Bias5']], left_on='Date', right_index=True, how='left')
merged_df = pd.merge(merged_df, sox[['SOX_Return_Yesterday']], left_on='Date', right_index=True, how='left')

# (4) 加入台積電原本的技術指標
merged_df["TSMC_Return"] = merged_df["Close"].pct_change()
merged_df["TSMC_Bias5"] = (merged_df["Close"] - merged_df["Close"].rolling(5).mean()) / merged_df["Close"].rolling(5).mean()
merged_df["TSMC_Vol_Change"] = merged_df["Volume"].pct_change()

# 預測目標：回歸本質，預測明天的台積電是漲還是跌
merged_df["Target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)

# 剔除因為移動窗格產生的空值
merged_df = merged_df.dropna().reset_index(drop=True)

# 💡 特徵大軍：融合了美股、大盤、個股價量
feature_cols = ["TSMC_Return", "TSMC_Vol_Change", "TSMC_Bias5", "TWII_Bias5", "SOX_Return_Yesterday"]
X = merged_df[feature_cols].values
y = merged_df["Target"].values

# 三階段切分 (60%, 20%, 20%)
total_len = len(X)
train_end = int(total_len * 0.6)
val_end = int(total_len * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X[val_end:], y[val_end:]

print(f"📊 特徵整合完畢！總交易日: {total_len} 天")
print(f"📝 訓練集: {len(X_train)} 天 | 驗證集: {len(X_val)} 天 | 測試集: {len(X_test)} 天")

# ==========================================
# 3. 🛠️ 訓練與動態監控
# ==========================================
print("\n--- [步驟 3] 喚醒 6 核心處理器，啟動『跨國聯動模型』訓練 ---")

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.03,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=15
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=10
)

# ==========================================
# 4. 計算具有實戰邏輯的真實勝率
# ==========================================
test_predictions = model.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)

print("\n" + "=" * 60)
print(f"🏁 【跨國聯動模型 - 終極盲測成果】")
print(f"   --> 機器人在完全陌生的千元多頭市場（2025-2026）中，")
print(f"       預測明天台積電漲跌的真實正向勝率為: {test_accuracy:.2%}")
print("=" * 60)

print("\n【大局觀架構下：各市場指標的貢獻度排名】")
for col, score in zip(feature_cols, model.feature_importances_):
    print(f" 🌟 {col}: {score:.4f}")