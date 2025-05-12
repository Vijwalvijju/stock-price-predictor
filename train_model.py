import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Download data
df = yf.download("AAPL", start="2015-01-01", end="2024-12-31")

# Save CSV
os.makedirs("data", exist_ok=True)
data.to_csv("data/historical_data.csv", columns=["Date", "Close", "High", "Low", "Open", "Volume"], index=False)

# Preprocessing
data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Save model
model.save("model/lstm_model.h5")
print("✅ Model saved to model/lstm_model.h5")
