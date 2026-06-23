import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Cryptocurrency Price Predictor")
st.write("Deep Learning Project using LSTM")

# -----------------------------------
# Load Model
# -----------------------------------
model = load_model("models/lstm_model.keras")

# -----------------------------------
# Download Latest Data
# -----------------------------------
data = yf.download(
    "BTC-USD",
    period="180d",
    auto_adjust=True
)

close = data[['Close']]

# -----------------------------------
# Show Dataset
# -----------------------------------
st.subheader("Latest Bitcoin Prices")

st.dataframe(data.tail())

# -----------------------------------
# Plot Historical Price
# -----------------------------------
st.subheader("Bitcoin Closing Price")

fig = plt.figure(figsize=(12,5))

plt.plot(close)

plt.xlabel("Days")

plt.ylabel("Price (USD)")

plt.title("Historical Bitcoin Price")

st.pyplot(fig)

# -----------------------------------
# Scale Data
# -----------------------------------
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(close)

window = 60

last_60 = scaled_data[-window:]

X = np.array([last_60])

X = np.reshape(
    X,
    (X.shape[0], X.shape[1], 1)
)

prediction = model.predict(X)

prediction = scaler.inverse_transform(prediction)

st.subheader("Tomorrow Prediction")

st.success(
    f"${prediction[0][0]:,.2f}"
)

# -----------------------------------
# Predict Button
# -----------------------------------
if st.button("Predict Again"):

    data = yf.download(
        "BTC-USD",
        period="180d",
        auto_adjust=True
    )

    close = data[['Close']]

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(close)

    last60 = scaled[-60:]

    X = np.array([last60])

    X = np.reshape(
        X,
        (X.shape[0], X.shape[1], 1)
    )

    pred = model.predict(X)

    pred = scaler.inverse_transform(pred)

    st.success(
        f"New Prediction : ${pred[0][0]:,.2f}"
    )

st.markdown("---")
st.caption("Developed using TensorFlow, Keras, Streamlit and Yahoo Finance API")