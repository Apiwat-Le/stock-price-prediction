# ==============================================================================
#    Stock Price Prediction - Simple Version
#    ทำนายว่าราคาหุ้นพรุ่งนี้จะ "ขึ้น" หรือ "ลง"
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# STEP 1: สร้างข้อมูลหุ้นจำลอง (5 ปี)
# ------------------------------------------------------------------------------
print("=" * 50)
print("STEP 1: สร้างข้อมูลหุ้น")
print("=" * 50)

np.random.seed(42)

# สร้างวันที่ 5 ปี (เฉพาะวันทำการ)
dates = pd.date_range('2019-01-01', '2024-12-31', freq='B')

# สร้างราคาหุ้นจำลอง (เริ่มที่ 100 บาท)
price = 100
prices = []
for _ in range(len(dates)):
    change = np.random.randn() * 2  # เปลี่ยนแปลง ±2%
    price = price * (1 + change/100)
    prices.append(price)

# สร้าง DataFrame
df = pd.DataFrame({
    'date': dates,
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, len(dates))
})

print(f"จำนวนข้อมูล: {len(df)} วัน")
print(df.head())

# ------------------------------------------------------------------------------
# STEP 2: สร้าง Features (ตัวแปรที่ใช้ทำนาย)
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2: สร้าง Features")
print("=" * 50)

# 2.1 Moving Average - ค่าเฉลี่ยราคา 5 วัน และ 20 วัน
df['ma_5'] = df['close'].rolling(5).mean()
df['ma_20'] = df['close'].rolling(20).mean()

# 2.2 Price Change - ราคาเปลี่ยนแปลงกี่ %
df['price_change_1d'] = df['close'].pct_change(1) * 100   # เทียบเมื่อวาน
df['price_change_5d'] = df['close'].pct_change(5) * 100   # เทียบ 5 วันก่อน

# 2.3 RSI - ดูว่าซื้อ/ขายมากเกินไปหรือยัง
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain/loss))

# 2.4 Volume Change - ปริมาณซื้อขายเปลี่ยนแปลงกี่ %
df['volume_change'] = df['volume'].pct_change(1) * 100

# 2.5 Lag Features - ราคาย้อนหลัง 1-3 วัน
df['close_lag_1'] = df['close'].shift(1)
df['close_lag_2'] = df['close'].shift(2)
df['close_lag_3'] = df['close'].shift(3)

print("Features ที่สร้าง:")
print("- ma_5, ma_20 (Moving Average)")
print("- price_change_1d, price_change_5d (% เปลี่ยนแปลง)")
print("- rsi (Relative Strength Index)")
print("- volume_change (% Volume เปลี่ยนแปลง)")
print("- close_lag_1/2/3 (ราคาย้อนหลัง)")

# ------------------------------------------------------------------------------
# STEP 3: สร้าง Target (สิ่งที่จะทำนาย)
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3: สร้าง Target")
print("=" * 50)

# ราคาพรุ่งนี้ขึ้น = 1, ลง = 0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

print("Target: ราคาพรุ่งนี้ขึ้น(1) หรือ ลง(0)")
print(f"วันที่ราคาขึ้น: {df['target'].sum()} วัน")
print(f"วันที่ราคาลง: {len(df) - df['target'].sum()} วัน")

# ------------------------------------------------------------------------------
# STEP 4: เตรียมข้อมูลสำหรับ Train
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4: เตรียมข้อมูล")
print("=" * 50)

# ลบแถวที่มีค่าว่าง
df = df.dropna()

# เลือก Features
features = ['ma_5', 'ma_20', 'price_change_1d', 'price_change_5d', 
            'rsi', 'volume_change', 'close_lag_1', 'close_lag_2', 'close_lag_3']

X = df[features]
y = df['target']

# แบ่งข้อมูล: 80% train, 20% test (แบ่งตามเวลา ไม่ใช่ random!)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"ข้อมูล Train: {len(X_train)} วัน")
print(f"ข้อมูล Test: {len(X_test)} วัน")

# ------------------------------------------------------------------------------
# STEP 5: Train Model
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5: Train Model (Random Forest)")
print("=" * 50)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Training เสร็จสิ้น!")

# ------------------------------------------------------------------------------
# STEP 6: ทดสอบและประเมินผล
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 6: ประเมินผล")
print("=" * 50)

# ทำนาย
y_pred = model.predict(X_test)

# วัดความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print(f"\n★ Accuracy: {accuracy:.2%}")
print(f"  (ทำนายถูก {int(accuracy * len(y_test))} จาก {len(y_test)} วัน)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ลง', 'ขึ้น']))

# ------------------------------------------------------------------------------
# STEP 7: ดู Feature Importance
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 7: Feature ไหนสำคัญที่สุด?")
print("=" * 50)

importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.to_string(index=False))

# ------------------------------------------------------------------------------
# STEP 8: สร้างกราฟ
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 8: สร้าง Visualization")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# กราฟ 1: ราคาหุ้น
axes[0, 0].plot(df['date'], df['close'])
axes[0, 0].set_title('Stock Price')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price')

# กราฟ 2: Moving Averages
axes[0, 1].plot(df['date'][-100:], df['close'][-100:], label='Price')
axes[0, 1].plot(df['date'][-100:], df['ma_5'][-100:], label='MA 5')
axes[0, 1].plot(df['date'][-100:], df['ma_20'][-100:], label='MA 20')
axes[0, 1].set_title('Price vs Moving Averages (Last 100 days)')
axes[0, 1].legend()

# กราฟ 3: RSI
axes[1, 0].plot(df['date'][-100:], df['rsi'][-100:])
axes[1, 0].axhline(70, color='r', linestyle='--', label='Overbought')
axes[1, 0].axhline(30, color='g', linestyle='--', label='Oversold')
axes[1, 0].set_title('RSI (Last 100 days)')
axes[1, 0].legend()

# กราฟ 4: Feature Importance
axes[1, 1].barh(importance['feature'], importance['importance'])
axes[1, 1].set_title('Feature Importance')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('stock_analysis.png', dpi=150)
print("บันทึกกราฟเป็น stock_analysis.png")

# ------------------------------------------------------------------------------
# สรุป
# ------------------------------------------------------------------------------
print("\n" + "=" * 50)
print("สรุป")
print("=" * 50)
print(f"• ข้อมูล: {len(df)} วัน")
print(f"• Features: {len(features)} ตัว")
print(f"• Model: Random Forest")
print(f"• Accuracy: {accuracy:.2%}")
print("=" * 50)