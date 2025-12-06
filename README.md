# 📈 Crypto Price Prediction (BTC & ETH) using LSTM

## Description
This project uses **Deep Learning** (specifically a Bidirectional LSTM neural network) to predict the closing prices of Bitcoin (BTC) and Ethereum (ETH).
The program downloads historical data from Yahoo Finance, calculates technical indicators (like RSI and MACD), and trains a model to forecast future prices.

## 🛠 Features
* **Data Source:** Automatically downloads data using `yfinance`.
* **Data Processing:** Cleans data and adds technical indicators:
    * Moving Averages (7 days)
    * Volatility
    * RSI (Relative Strength Index)
    * MACD (Moving Average Convergence Divergence)
* **AI Model:** Uses TensorFlow/Keras with a **Bidirectional LSTM** architecture to learn from time sequences.
* **Visualization:** Plots the "Real" vs "Predicted" prices for visual comparison.

## 📦 Requirements
To run this code, you need Python installed with the following libraries:

```bash
pip install yfinance numpy matplotlib scikit-learn tensorflow
```

## 🧠 Model Architecture
* **Input:** Sequences of 90 days of data (prices + indicators).
* **Hidden Layers:** Two Bidirectional LSTM layers with Dropout (to prevent overfitting).
* **Output:** A Dense layer with 2 units (BTC price and ETH price).

## 📊 Results & Analysis

### 1. Performance Metrics
The model was evaluated using **RMSE** (Root Mean Squared Error) and **MAPE** (Mean Absolute Percentage Error).

* **Bitcoin (BTC):**
    * **MAPE:** `10.68%`
    * *Analysis:* The model is relatively accurate for Bitcoin. An error of ~10% is acceptable for a highly volatile asset like crypto, but it is clearly not perfect for precise trading.
* **Ethereum (ETH):**
    * **MAPE:** `19.88%`
    * *Analysis:* The model struggles more with Ethereum. The error is almost double the BTC error. This suggests ETH is harder to predict or follows different patterns than BTC.

### 2. Visual Analysis (Graphs)

**General Observation:**
![BTC Prediction Graph](images/btc_prediction.png)
![ETH Prediction Graph](images/eth_prediction.png)
The Predicted line (orange) successfully follows the general **trend** of the Real line (blue). When the market crashes or rallies, the model understands the direction.

**Issues:**
1.  **Lag (Delay):** The prediction often reacts *after* the real price moves. The model is "chasing" the price rather than predicting it in advance.
2.  **Smoothing:** The model is "conservative."
    * For **BTC**, it failed to reach the very high peak (around index 450).
    * For **ETH**, the gap is wider; it missed the depth of the massive drop (around index 350).

The model is **good at identifying general market trends** (bull market vs. bear market) but is **not accurate enough for high-frequency trading**. It tends to smooth out extreme spikes and drops. To improve this, we could try adding more external data (like news sentiment) or adjusting the sequence length.
