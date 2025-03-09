import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load model
model = load_model(r'C:\Users\ankit\Stock-Trend-Prediction\Stock prediction Model.keras')

st.header('Stock Price Prediction')
stock = st.text_input('Enter the stock symbol', 'AAPL')
start = '2012-01-01'
end = '2022-01-01'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

train_data = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
test_data = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pst_100_days = train_data.tail(100)
test_data = pd.concat([pst_100_days, test_data], ignore_index=True)
test_data_scale = scaler.fit_transform(test_data)


x = []
y = []

for i in range(100, test_data_scale.shape[0]):
    x.append(test_data_scale[i-100:i])
    y.append(test_data_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale


st.subheader('MACD 50')
ma_50_days = data.Close.rolling(50).mean()
macd50 = plt.figure(figsize=(12, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(macd50)

st.subheader('MACD 100')
ma_100_days = data.Close.rolling(100).mean()
macd100 = plt.figure(figsize=(12, 6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

st.pyplot(macd100)

st.subheader('MACD 200')
ma_200_days = data.Close.rolling(200).mean()
macd200 = plt.figure(figsize=(12, 6))
plt.plot(ma_200_days, 'r')
plt.plot(data.Close, 'g')
plt.show()

st.pyplot(macd200)

st.subheader('Price vs MA100 vs MA200')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(data.Close, 'g')
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'y')  
plt.show()
st.pyplot(fig4)

st.subheader('Original Price vs Predicted Price')
fig5 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig5)
  
