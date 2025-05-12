import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pandas as pd

def get_data(stock):
    df = yf.download(stock, period="5y")
    return df

def preprocess_data(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def predict_next_day(data, model):
    last_60 = data[-60:]
    input_data = np.reshape(last_60, (1, 60, 1))
    prediction = model.predict(input_data)
    return prediction
