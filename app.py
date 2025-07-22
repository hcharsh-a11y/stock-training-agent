# app.py (Final Definitive Solution)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
import joblib
import json

# --- App Configuration ---
st.set_page_config(page_title="AI Stock Forecaster", page_icon="📈", layout="wide")

# --- Configuration ---
SEQUENCE_LENGTH = 60
FORECAST_DAYS = 60
MODEL_DIR = "trained_models"

# --- Stock List ---
STOCKS = {
    "Apple Inc. (AAPL)": "AAPL",
    "NVIDIA Corporation (NVDA)": "NVDA",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Microsoft Corporation (MSFT)": "MSFT",
}

# --- Core Functions ---

@st.cache_resource(show_spinner="Loading AI model and assets...")
def load_assets(_ticker):
    """Loads the model, scaler, and precision score from the repository."""
    model_path = os.path.join(MODEL_DIR, f"{_ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{_ticker}_scaler.joblib")
    rmse_path = os.path.join(MODEL_DIR, f"{_ticker}_rmse.json")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, rmse_path]):
        return None, None, None
        
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        with open(rmse_path, 'r') as f:
            precision = json.load(f)
        return model, scaler, precision['rmse']
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

def generate_forecast(model, data, scaler):
    """Generates future predictions."""
    last_sequence_unscaled = data['Close'][-SEQUENCE_LENGTH:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence_unscaled)
    
    current_sequence_list = list(last_sequence_scaled.flatten())
    future_predictions = []

    for _ in range(FORECAST_DAYS):
        current_sequence_array = np.array(current_sequence_list).reshape(1, SEQUENCE_LENGTH, 1)
        next_pred_scaled = model.predict(current_sequence_array, verbose=0)
        prediction_value = next_pred_scaled[0, 0]
        future_predictions.append(prediction_value)
        current_sequence_list.pop(0)
        current_sequence_list.append(prediction_value)

    future_forecast = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_forecast

def plot_forecast(stock_data, future_forecast, ticker_name):
    """Creates the main forecast chart."""
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, FORECAST_DAYS + 1)]
    
    historical_prices = np.array(stock_data['Close']).flatten()
    forecast_prices = np.array(future_forecast).flatten()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=stock_data.index, y=historical_prices, mode='lines', name='Historical Price', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_prices, mode='lines', name='Forecast', line=dict(color='darkorange', width=2, dash='dash')))
    
    all_values = np.concatenate([historical_prices, forecast_prices])
    y_min = all_values.min() * 0.95
    y_max = all_values.max() * 1.05
    
    fig.update_layout(title=f"{ticker_name} - Historical Price and {FORECAST_DAYS}-Day Forecast", xaxis_title="Date", yaxis_title="Stock Price", yaxis=dict(range=[y_min, y_max]), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    return fig

# --- Streamlit UI ---
st.title("🤖 AI Stock Forecaster")
st.markdown("This website uses an automated agent to provide daily updated stock forecasts.")

selected_stock_name = st.selectbox("Choose a stock to forecast:", list(STOCKS.keys()))

if selected_stock_name:
    ticker = STOCKS[selected_stock_name]
    
    model, scaler, rmse = load_assets(ticker)
    
    if model and scaler and rmse:
        try:
            data = yf.download(ticker, period="1y", progress=False)
            
            if data.empty:
                st.error("Could not download stock data.")
            else:
                st.subheader("Today's Snapshot & Model Precision")
                
                latest = data.iloc[-1]
                
                currency_symbol = "$"
                try:
                    stock_info = yf.Ticker(ticker)
                    currency = stock_info.info.get('currency', 'USD')
                    if currency == "INR":
                        currency_symbol = "₹"
                except Exception:
                    pass

                # --- THE DEFINITIVE FIX IS HERE ---
                # Convert Pandas Series to a simple float before formatting
                last_close_val = float(latest['Close'])
                high_val = float(latest['High'])
                low_val = float(latest['Low'])
                # --- END FIX ---

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Close", f"{currency_symbol}{last_close_val:.2f}")
                col2.metric("Day High", f"{currency_symbol}{high_val:.2f}")
                col3.metric("Day Low", f"{currency_symbol}{low_val:.2f}")
                col4.metric("Model Precision (RMSE)", f"{currency_symbol}{rmse:.2f}", help="Root Mean Square Error. Lower is better. This shows the model's average prediction error on historical test data.")
                
                st.divider()

                st.subheader(f"Forecast for {selected_stock_name}")
                
                with st.spinner("Generating forecast..."):
                    future_forecast = generate_forecast(model, data, scaler)

                st.plotly_chart(plot_forecast(data, future_forecast, selected_stock_name), use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning(f"The model for {selected_stock_name} is not available yet. The agent may still be training it. Please run the training agent and check back later.")
