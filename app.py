# app.py (Definitive Solution)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import os
import joblib

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

@st.cache_resource(show_spinner="Loading AI model and scaler...")
def load_assets(_ticker):
    """Loads the pre-trained model and scaler from the repository."""
    model_path = os.path.join(MODEL_DIR, f"{_ticker}_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{_ticker}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

def generate_forecast(model, data, scaler):
    """
    Uses the loaded model to forecast future prices with robust, manual array handling
    to prevent dimension errors.
    """
    # Start with the last known sequence of data
    last_sequence_unscaled = data['Close'][-SEQUENCE_LENGTH:].values.reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence_unscaled)
    
    # Use a simple Python list for manipulation to avoid NumPy dimension issues
    current_sequence_list = list(last_sequence_scaled.flatten())
    future_predictions = []

    for _ in range(FORECAST_DAYS):
        # Convert the list to the 3D NumPy array shape the model expects for prediction
        current_sequence_array = np.array(current_sequence_list).reshape(1, SEQUENCE_LENGTH, 1)
        
        # Predict the next value
        next_pred_scaled = model.predict(current_sequence_array, verbose=0)
        
        # Extract the single predicted value (scalar)
        prediction_value = next_pred_scaled[0, 0]
        future_predictions.append(prediction_value)
        
        # Manually update the sequence: remove the first element and append the new prediction
        current_sequence_list.pop(0)
        current_sequence_list.append(prediction_value)

    # Convert the list of predictions back to a NumPy array for inverse scaling
    future_forecast = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_forecast

def plot_forecast(stock_data, future_forecast, ticker_name):
    """Creates a robust Plotly chart."""
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, FORECAST_DAYS + 1)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=stock_data.index, 
        y=stock_data['Close'], 
        mode='lines', 
        name='Historical Price',
        line=dict(color='royalblue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_forecast.flatten(), 
        mode='lines',
        name='Forecast', 
        line=dict(color='darkorange', width=2, dash='dash')
    ))
    
    all_values = np.concatenate([stock_data['Close'].values, future_forecast.flatten()])
    y_min = all_values.min() * 0.95
    y_max = all_values.max() * 1.05
    
    fig.update_layout(
        title=f"{ticker_name} - Historical Price and {FORECAST_DAYS}-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Stock Price",
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

# --- Streamlit UI ---
st.title("🤖 AI Stock Forecaster")
st.markdown("This website uses an automated agent to provide daily updated stock forecasts.")

selected_stock_name = st.selectbox("Choose a stock to forecast:", list(STOCKS.keys()))

if selected_stock_name:
    ticker = STOCKS[selected_stock_name]
    model, scaler = load_assets(ticker)
    
    if model and scaler:
        try:
            data = yf.download(ticker, period="1y", progress=False)
            if data.empty:
                st.error("Could not download stock data.")
            else:
                st.subheader(f"Forecast for {selected_stock_name}")
                
                with st.spinner("Generating forecast..."):
                    future_forecast = generate_forecast(model, data, scaler)

                st.plotly_chart(plot_forecast(data, future_forecast, selected_stock_name), use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning(f"Model for {selected_stock_name} not available.")
