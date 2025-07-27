# main.py (USD Stocks Only)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import os
import json
import sys

# --- Configuration ---
START_DATE = '2015-01-01'
SEQUENCE_LENGTH = 60
MODEL_DIR = "trained_models"
TRAIN_TEST_SPLIT = 0.9

# --- NEW, USD-ONLY STOCK LIST ---
STOCKS = {
    "Apple Inc. (AAPL)": "AAPL",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Microsoft Corporation (MSFT)": "MSFT",
    "Amazon.com, Inc. (AMZN)": "AMZN",
    "Tesla, Inc. (TSLA)": "TSLA",
    "Meta Platforms, Inc. (META)": "META",
    "JPMorgan Chase & Co. (JPM)": "JPM"
}

os.makedirs(MODEL_DIR, exist_ok=True)

def create_and_evaluate_model(ticker):
    """
    Trains a high-accuracy model in an isolated session.
    """
    tf.keras.backend.clear_session()
    print(f"--- Starting new isolated session for {ticker} ---")
    
    data = yf.download(ticker, start=START_DATE, end=pd.to_datetime('today'), progress=False)
    
    if data.empty:
        print(f"ERROR: No data downloaded for {ticker}. Skipping.")
        return False

    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split_index = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print(f"Training high-accuracy model for {ticker}...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    print(f"Evaluating model precision for {ticker}...")
    predictions_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"Precision (RMSE) for {ticker}: {rmse}")

    print(f"Re-training model on full dataset for {ticker}...")
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    rmse_path = os.path.join(MODEL_DIR, f"{ticker}_rmse.json")
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(rmse_path, 'w') as f:
        json.dump({'rmse': rmse}, f)
    
    print(f"✅ High-accuracy model for {ticker} saved successfully.")
    return True

# --- Main execution loop ---
if __name__ == "__main__":
    success_count = 0
    for name, ticker in STOCKS.items():
        try:
            if create_and_evaluate_model(ticker):
                success_count += 1
        except Exception as e:
            print(f"CRITICAL ERROR while processing {ticker}. Error: {e}")
    
    if success_count < len(STOCKS):
        print(f"🔴 WARNING: Only {success_count} out of {len(STOCKS)} models were trained.")
    else:
        print(f"✅ SUCCESS: All {success_count} models were trained.")
