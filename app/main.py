import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title
st.title("📈 Stock Price Prediction using LSTM")

# Load data
df = pd.read_csv("data/historical_data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Show recent data
st.subheader("Recent Closing Prices")
st.line_chart(df["Close"].tail(100))

# Load model
model_path = "model/lstm_model.h5"
if not os.path.exists(model_path):
    st.error("LSTM model not found. Please train the model using `train_model.py` first.")
    st.stop()

model = load_model(model_path)

# Scale data
data = df.filter(["Close"])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Prepare last 60 days for prediction
last_60_days = scaled_data[-60:]
X_test = np.reshape(last_60_days, (1, 60, 1))

# Predict
predicted_price_scaled = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# Get latest actual price
latest_price = data["Close"].iloc[-1]

# Display side-by-side cards
col1, col2 = st.columns(2)
col1.metric("📉 Latest Actual Closing Price", f"${latest_price:.2f}")
col2.metric("📈 Predicted Next Day Price", f"${predicted_price[0][0]:.2f}")

# Plot actual vs predicted (trend)
st.subheader("📊 Trend Visualization (Past vs Predicted)")

# Add predicted point to plot
predicted_df = pd.DataFrame(index=[df.index[-1] + pd.Timedelta(days=1)],
                            data={"Close": predicted_price[0][0]})

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.tail(100).index, y=df["Close"].tail(100),
                         name="Actual Closing Price", line=dict(color='royalblue')))
fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df["Close"],
                         name="Predicted Next Day Price", mode='markers+text',
                         marker=dict(color='red', size=10),
                         text=[f"${predicted_price[0][0]:.2f}"], textposition="top center"))

fig.update_layout(title="Stock Price Trend with Prediction",
                  xaxis_title="Date", yaxis_title="Price (USD)", legend_title="Legend")

st.plotly_chart(fig, use_container_width=True)

st.info("Prediction made using an LSTM neural network trained on the last 60 days of closing price data.")
