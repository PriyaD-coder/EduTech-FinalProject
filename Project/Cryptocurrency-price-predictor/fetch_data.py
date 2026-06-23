import os
import yfinance as yf

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

print("Downloading Bitcoin Historical Data...")

# Download Bitcoin historical data
btc = yf.download(
    "BTC-USD",
    start="2018-01-01",
    end="2026-01-01",
    auto_adjust=True
)

# Convert index to column
btc.reset_index(inplace=True)

# Save dataset
btc.to_csv("dataset/bitcoin.csv", index=False)

print("Dataset saved successfully!")
print(btc.tail())
