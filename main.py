import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf

cfg = {
    "start_date": "2018-01-01",
    "end_date": "2025-12-06",
    "seq_length": 90,
    "epochs": 50,
    "batch_size": 64,
    "train_split_ratio": 0.8
}


# Load data
btc_df = yf.download("BTC-USD", start=cfg['start_date'], end=cfg['end_date'])[["Close", "Volume"]]
eth_df = yf.download("ETH-USD", start=cfg['start_date'], end=cfg['end_date'])[["Close", "Volume"]]

btc_df.rename(columns={"Close": "BTC_Close", "Volume": "BTC_Volume"}, inplace=True)
eth_df.rename(columns={"Close": "ETH_Close", "Volume": "ETH_Volume"}, inplace=True)

df = btc_df.join(eth_df, how="inner")

# Return
df['BTC_Return'] = df['BTC_Close'].pct_change()
df['ETH_Return'] = df['ETH_Close'].pct_change()

# 7 last days mean
df['BTC_MA7'] = df['BTC_Close'].rolling(7).mean()
df['ETH_MA7'] = df['ETH_Close'].rolling(7).mean()

# Volatility
df['BTC_Volatility'] = df['BTC_Return'].rolling(7).std()
df['ETH_Volatility'] = df['ETH_Return'].rolling(7).std()

# Relative Strenght Index
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10) # 1e-10 avoid division by 0
    rsi = 100 - (100 / (1 + rs))   
    return rsi

df['BTC_RSI'] = compute_rsi(df['BTC_Close'])
df['ETH_RSI'] = compute_rsi(df['ETH_Close'])

# Moving Average Convergence Divergence
def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

df['BTC_MACD'], df['BTC_MACD_Signal'] = compute_macd(df['BTC_Close'])
df['ETH_MACD'], df['ETH_MACD_Signal'] = compute_macd(df['ETH_Close'])



df.dropna(inplace=True)
if df.empty:
    print("ERROR: df is empty !")
else:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, [0, 2]])  # BTC_Close and ETH_Close
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, seq_length=cfg['seq_length'])

# Train and test
split_idx = int(cfg['train_split_ratio'] * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=cfg['epochs'],
    batch_size=cfg['batch_size'],
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# PREDICT
y_pred = model.predict(X_test)

def inverse_transform_columns(scaler, y_scaled, column_indices):
    dummy = np.zeros((len(y_scaled), scaler.n_features_in_))
    dummy[:, column_indices] = y_scaled
    return scaler.inverse_transform(dummy)[:, column_indices]

y_test_actual = inverse_transform_columns(scaler, y_test, [0, 2])
y_pred_actual = inverse_transform_columns(scaler, y_pred, [0, 2])

# Metrics
btc_rmse = np.sqrt(mean_squared_error(y_test_actual[:, 0], y_pred_actual[:, 0]))
eth_rmse = np.sqrt(mean_squared_error(y_test_actual[:, 1], y_pred_actual[:, 1]))
btc_mape = mean_absolute_percentage_error(y_test_actual[:, 0], y_pred_actual[:, 0])
eth_mape = mean_absolute_percentage_error(y_test_actual[:, 1], y_pred_actual[:, 1])

print(f"BTC RMSE (Direct): {btc_rmse:.2f}, BTC MAPE: {btc_mape:.2%}")
print(f"ETH RMSE (Direct): {eth_rmse:.2f}, ETH MAPE: {eth_mape:.2%}")

# PLOT
plt.figure(figsize=(14,6))
plt.plot(y_test_actual[:, 0], label='BTC Real')
plt.plot(y_pred_actual[:, 0], label='BTC Direct')
plt.legend()
plt.title("BTC prediction")

plt.figure(figsize=(14,6))
plt.plot(y_test_actual[:, 1], label='ETH Real')
plt.plot(y_pred_actual[:, 1], label='ETH Direct')
plt.legend()
plt.title("ETH prediction")

plt.show()