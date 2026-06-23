
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# Create folders
# -----------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
csv_path = "dataset/bitcoin.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        "bitcoin.csv not found! Run fetch_data.py first."
    )

# Read dataset
data = pd.read_csv(csv_path)

print("\nDataset Preview:")
print(data.head())

print("\nColumns:")
print(data.columns)

# -----------------------------
# Handle different CSV formats
# -----------------------------
if "Close" in data.columns:
    close_price = data["Close"]

else:
    # Find any column containing "Close"
    close_columns = [col for col in data.columns if "Close" in str(col)]

    if len(close_columns) == 0:
        raise Exception(
            "Close column not found in bitcoin.csv"
        )

    close_price = data[close_columns[0]]

# Convert to numeric
close_price = pd.to_numeric(close_price, errors="coerce")

# Remove missing values
close_price = close_price.dropna()

close_price = close_price.values.reshape(-1, 1)

# -----------------------------
# Normalize
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(close_price)

# -----------------------------
# Sliding Window
# -----------------------------
window = 60

X_train = []
y_train = []

for i in range(window, len(scaled_data)):
    X_train.append(scaled_data[i-window:i, 0])
    y_train.append(scaled_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

print("\nTraining Samples:", X_train.shape)

# -----------------------------
# Build LSTM
# -----------------------------
model = Sequential()

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(window, 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=64
    )
)

model.add(Dropout(0.2))

model.add(Dense(25))
model.add(Dense(1))

# -----------------------------
# Compile
# -----------------------------
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# -----------------------------
# Train
# -----------------------------
print("\nTraining Model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    verbose=1
)

# -----------------------------
# Save Model
# -----------------------------
model.save("models/lstm_model.keras")

print("\nModel Saved Successfully!")

# -----------------------------
# Prediction
# -----------------------------
predicted = model.predict(X_train)

predicted = scaler.inverse_transform(predicted)

actual = scaler.inverse_transform(
    y_train.reshape(-1, 1)
)

# -----------------------------
# RMSE
# -----------------------------
rmse = np.sqrt(mean_squared_error(actual, predicted))

print(f"\nRMSE : {rmse:.2f}")

# -----------------------------
# Graph
# -----------------------------
plt.figure(figsize=(15, 6))

plt.plot(actual, label="Actual Price")

plt.plot(predicted, label="Predicted Price")

plt.title("Bitcoin Price Prediction using LSTM")

plt.xlabel("Time")

plt.ylabel("Bitcoin Price (USD)")

plt.legend()

plt.grid(True)

plt.savefig(
    "graphs/prediction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPrediction graph saved in graphs/prediction.png")