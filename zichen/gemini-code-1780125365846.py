import os
import numpy as np
import pandas as pd
import xgboost as xgb
from FinMind.data import DataLoader

# ==========================================
# 1. 讀取本機數據 + 線上抓取三大法人籌碼面大數據
# ==========================================
csv_path = "c:/Users/user/Downloads/tsmc_5y_data.csv"
if not os.path.exists(csv_path):
    print(f"❌ 找不到台積電原始檔案！請確認 {csv_path} 是否存在。")
    exit()

print("--- [步驟 1] 讀取本機數據，並透過 FinMind API 跨境捕捉法人實彈籌碼 ---")
tsmc_df = pd.read_csv(csv_path)

# 將民國年轉換為標準西元年
def convert_date(date_str):
    parts = date_str.split('/')
    year = int(parts[0]) + 1911
    return f"{year}-{parts[1]}-{parts[2]}"

tsmc_df['Date'] = pd.to_datetime(tsmc_df['Date'].apply(convert_date))
tsmc_df = tsmc_df.sort_values('Date').reset_index(drop=True)

start_date = tsmc_df['Date'].min().strftime('%Y-%m-%d')
end_date = tsmc_df['Date'].max().strftime('%Y-%m-%d')

# 🎯 調用 FinMind 下載台積電三大法人買賣超數據 (使用最新函式名)
print(f"📡 正在從台灣證交所下載 {start_date} 至 {end_date} 三大法人籌碼數據...")
dl = DataLoader()
chip_df = dl.taiwan_stock_institutional_investors(
    stock_id='2330', start_date=start_date, end_date=end_date
)

# 💡 安全檢查：自動適應官方欄位改名
if 'buy_sell' not in chip_df.columns and 'buy' in chip_df.columns:
    # 如果沒有 buy_sell 欄位，我們就用 (買進 - 賣出) 自行計算出淨買賣超張數
    chip_df['buy_sell'] = chip_df['buy'] - chip_df['sell']
elif 'value' in chip_df.columns:
    # 有些版本會直接叫做 value
    chip_df['buy_sell'] = chip_df['value']

# 整理籌碼數據：將外資、投信、自營商的淨買賣超欄位橫向展開 (Pivot)
chip_pivot = chip_df.pivot(index='date', columns='name', values='buy_sell').fillna(0)
chip_pivot.index = pd.to_datetime(chip_pivot.index)

# 自動檢查展開後的欄位，確保更名對齊
rename_dict = {}
for col in chip_pivot.columns:
    if '外資' in col or 'Foreign' in col:
        rename_dict[col] = 'Foreign_BuySell'
    elif '投信' in col or 'Investment' in col:
        rename_dict[col] = 'Investment_Trust_BuySell'
    elif '自營商' in col or 'Dealer' in col:
        rename_dict[col] = 'Dealer_BuySell'

chip_pivot = chip_pivot.rename(columns=rename_dict)

# 確保三個核心籌碼欄位都存在，沒有的話補 0
for col in ['Foreign_BuySell', 'Investment_Trust_BuySell', 'Dealer_BuySell']:
    if col not in chip_pivot.columns:
        chip_pivot[col] = 0

# ==========================================
# 2. 🚀 籌碼面進階特徵工程 (後續程式碼接回原本的...)
# ==========================================
print("\n--- [步驟 2] 實作法人籌碼特徵工程 ---")
# ...後面 Foreign_Streak 的計算和 XGBoost 訓練完全不用動...
foreign_net = chip_pivot['Foreign_BuySell']
is_buy = (foreign_net > 0).astype(int)
is_sell = (foreign_net < 0).astype(int)

# 利用累積加總魔法計算連續天數
buy_streak = is_buy.groupby((is_buy != is_buy.shift()).cumsum()).cumsum()
sell_streak = is_sell.groupby((is_sell != is_sell.shift()).cumsum()).cumsum()
chip_pivot['Foreign_Streak'] = np.where(foreign_net > 0, buy_streak, -sell_streak)

# (2) 將籌碼特徵與台積電股價用「日期」黏在一起
merged_df = pd.merge(tsmc_df, chip_pivot, left_on='Date', right_index=True, how='left')

# (3) 加入基礎漲跌特徵
merged_df["TSMC_Return"] = merged_df["Close"].pct_change()
merged_df["Target"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)

# 剔除空值
merged_df = merged_df.dropna().reset_index(drop=True)

# 💡 特徵大軍：全面換上籌碼面黃金武器
feature_cols = [
    "Foreign_BuySell",          # 外資今天買賣超張數
    "Investment_Trust_BuySell", # 投信今天買賣超張數
    "Dealer_BuySell",           # 自營商今天買賣超張數
    "Foreign_Streak",           # 外資連續買超/賣超天數
    "TSMC_Return"               # 個股當日漲跌幅
]
X = merged_df[feature_cols].values
y = merged_df["Target"].values

# 三階段切分 (60%, 20%, 20%)
total_len = len(X)
train_end = int(total_len * 0.6)
val_end = int(total_len * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
X_test, y_test   = X[val_end:], y[val_end:]

print(f"📊 籌碼面整合完畢！總交易日: {total_len} 天")
print(f"📝 訓練集: {len(X_train)} 天 | 驗證集: {len(X_val)} 天 | 測試集: {len(X_test)} 天")

# ==========================================
# 3. 🛠️ 訓練與動態監控
# ==========================================
print("\n--- [步驟 3] 喚醒 6 核心處理器，啟動『法人籌碼流模型』訓練 ---")

model = xgb.XGBClassifier(
    n_estimators=200,
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
# 4. 計算最終大考成果
# ==========================================
test_predictions = model.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)

print("\n" + "=" * 60)
print(f"🏁 【法人籌碼流模型 - 終極盲測成果】")
print(f"   --> 機器人透過追蹤主力資金動向，")
print(f"       預測明天台積電漲跌的真實盲測勝率為: {test_accuracy:.2%}")
print("=" * 60)

print("\n【籌碼大局觀下：各項法人指標的貢獻度排名】")
for col, score in zip(feature_cols, model.feature_importances_):
    print(f" 🌟 {col}: {score:.4f}")