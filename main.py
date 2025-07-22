# main.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import os

# --- Configuration ---
START_DATE = '2015-01-01'
SEQUENCE_LENGTH = 60
MODEL_DIR = "trained_models"  # Directory to save models

STOCKS = {
    "Apple Inc. (AAPL)": "AAPL",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Microsoft Corporation (MSFT)": "MSFT",
    # ... add the rest of your stocks
}

# Create directory for models if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_save(ticker):
    """Trains a model and saves it to a local directory."""
    print(f"--- Processing {ticker} ---")
    data = yf.download(ticker, start=START_DATE, end=pd.to_datetime('today'), progress=False)
    if data.empty:
        print(f"No data for {ticker}, skipping.")
        return

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # ... (Data preparation code is the same) ...
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i - SEQUENCE_LENGTH:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ... (Model definition and training is the same) ...
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=25, batch_size=32, verbose=0)

    # Save files to the local MODEL_DIR
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model and scaler for {ticker} saved to {MODEL_DIR}.")


# --- Main execution loop ---
if __name__ == "__main__":
    for name, ticker in STOCKS.items():
        try:
            train_and_save(ticker)
        except Exception as e:
            print(f"Failed to process {ticker}. Error: {e}")