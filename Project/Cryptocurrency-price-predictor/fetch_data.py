import os
import yfinance as yf

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

print("Downloading Bitcoin Historical Data...")

# Download Bitcoin historical data
btc = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True
)

# Save dataset
btc.to_csv("dataset/bitcoin.csv")

print("Dataset saved successfully!")
print(btc.tail())