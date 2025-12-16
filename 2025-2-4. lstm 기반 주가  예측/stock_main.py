import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import os


TICKER = '005930.KS'
START_DATE = '2020-01-01'
END_DATE = '2024-12-31' # 최근까지
WINDOW_SIZE = 50  # 과거 50일을 보고 내일을 예측

base_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(base_dir, 'stock_lstm_model.h5')


print(f">> {TICKER} 주가 데이터 다운로드 중...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)


data = df['Close'].values.reshape(-1, 1)

print(f">> 데이터 로드 완료: 총 {len(data)}일 치")


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


x_data, y_data = [], []

for i in range(WINDOW_SIZE, len(scaled_data)):
    x_data.append(scaled_data[i-WINDOW_SIZE:i, 0])
    y_data.append(scaled_data[i, 0])

x_data, y_data = np.array(x_data), np.array(y_data)


x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))


train_size = int(len(x_data) * 0.8)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

print(f">> 학습 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")


model = Sequential()


model.add(LSTM(units=50, return_sequences=True, input_shape=(x_data.shape[1], 1)))
model.add(Dropout(0.2)) # 과적합 방지


model.add(LSTM(units=50, return_sequences=False)) # 마지막 층이므로 return_sequences=False
model.add(Dropout(0.2))


model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()



model.fit(x_train, y_train, epochs=25, batch_size=32, verbose=1)
model.save(model_save_path)
print(" 모델 저장 완료")



predictions = model.predict(x_test)


predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, color='blue', label='Actual Price (Samsung)')
plt.plot(predictions, color='red', label='Predicted Price (AI)')
plt.title('Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Price (KRW)')
plt.legend()
plt.savefig(os.path.join(base_dir, 'stock_result.png'))
print(">> 결과 그래프 저장 완료: stock_result.png")
plt.show()