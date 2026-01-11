# Stock Price Movement Prediction

A machine learning project demonstrating how to predict stock price movements (UP/DOWN) using technical indicators and Random Forest classifier.

## 📌 Project Overview

This project builds an end-to-end ML pipeline to predict whether a stock price will go **up** or **down** the next day. It demonstrates key concepts in financial machine learning including feature engineering, time-series data handling, and model evaluation.

> **Note:** This project uses **simulated stock data** for learning and demonstration purposes.

## 🔧 Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning
- **matplotlib** - Data visualization

## 📊 Features Engineered

| Feature | Description |
|---------|-------------|
| `ma_5`, `ma_20` | Moving Averages (5-day and 20-day) |
| `price_change_1d` | Daily price change (%) |
| `price_change_5d` | 5-day price change (%) |
| `rsi` | Relative Strength Index (momentum indicator) |
| `volume_change` | Daily volume change (%) |
| `close_lag_1/2/3` | Previous 1-3 days closing prices |

## 🔄 Pipeline Steps

```
1. Generate Data    → Simulated 5 years of stock prices
2. Feature Engineering → Create 9 technical indicators
3. Create Target    → Label: price goes UP(1) or DOWN(0)
4. Train/Test Split → 80/20 split (time-based, not random)
5. Train Model      → Random Forest Classifier
6. Evaluate         → Accuracy & Classification Report
7. Visualize        → Price charts, RSI, Feature Importance
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Run the script
```bash
python stock_prediction_simple.py
```

### 3. Output
- Console: Model accuracy and classification report
- File: `stock_analysis.png` (visualization charts)

## 📈 Sample Output

```
★ Accuracy: 50.65%
  (Predicted correctly 157 out of 310 days)

Feature Importance:
        feature  importance
price_change_1d    0.131951
  volume_change    0.125256
price_change_5d    0.117772
            rsi    0.115936
          ma_20    0.108180
```

## 📉 Visualization

The script generates 4 charts:
1. **Stock Price** - Overall price trend
2. **Moving Averages** - Price vs MA5 & MA20
3. **RSI** - Relative Strength Index with overbought/oversold levels
4. **Feature Importance** - Which features matter most

## 💡 Key Learnings

- **Time-series split** is crucial to prevent data leakage (no random split!)
- **Technical indicators** (MA, RSI) are commonly used in quantitative finance
- Stock prediction is inherently difficult; ~50% accuracy is expected for random walk data
- **Feature engineering** significantly impacts model performance

## 📁 Project Structure

```
stock-price-prediction/
├── README.md
├── stock_prediction_simple.py
├── stock_analysis.png
└── requirements.txt
```

## 🔮 Future Improvements

- [ ] Use real stock data from Yahoo Finance API (`yfinance`)
- [ ] Add more technical indicators (MACD, Bollinger Bands)
- [ ] Compare multiple models (XGBoost, LSTM)
- [ ] Implement backtesting framework

## 📝 License

This project is for educational purposes only. Not financial advice.

---

*Built as a learning project to demonstrate ML pipeline for financial data*
