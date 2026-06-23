import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model("models/lstm_model.keras")

# -----------------------------
# Download Latest Bitcoin Data
# -----------------------------
print("Downloading latest Bitcoin prices...")

data = yf.download(
    "BTC-USD",
    period="120d",
    auto_adjust=True
)

# Keep only Close Price
close_prices = data[['Close']]

# -----------------------------
# Scale Data
# -----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# -----------------------------
# Last 60 Days
# -----------------------------
window = 60

last_60_days = scaled_data[-window:]

X_test = []

X_test.append(last_60_days)

X_test = np.array(X_test)

X_test = np.reshape(
    X_test,
    (X_test.shape[0], X_test.shape[1], 1)
)

# -----------------------------
# Predict Tomorrow Price
# -----------------------------
prediction = model.predict(X_test)

prediction = scaler.inverse_transform(prediction)

print("\n==============================")
print("Predicted Bitcoin Price")
print("==============================")
print(f"Tomorrow's Predicted Price : ${prediction[0][0]:,.2f}")