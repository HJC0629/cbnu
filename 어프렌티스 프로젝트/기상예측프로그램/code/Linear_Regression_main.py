## Linear Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드 및 전처리 (파일 경로에 맞게 수정 필요)
weather_data = pd.read_csv('2020년도 기상데이터_청주.csv', encoding='euc-kr')
weather_data['일시'] = pd.to_datetime(weather_data['일시'], errors='coerce')
weather_data.dropna(subset=['기온(°C)', '습도(%)', '이슬점온도(°C)', '시정(10m)'], inplace=True)

# 특성과 타겟 정의
X = weather_data[['기온(°C)', '습도(%)', '이슬점온도(°C)']]
y = weather_data['시정(10m)']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# 시각화 1: 실제 시정거리 vs. 예측 시정거리 (산포도)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='skyblue', edgecolor='k', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Ideal Fit")
plt.xlabel("Actual Visibility (10m)")       #실제
plt.ylabel("Predicted Visibility (10m)")    #예측
plt.title("Actual vs Predicted Visibility")
plt.legend()
plt.grid(True)
plt.show()

# 시각화 2: 일자별 실제 시정거리 및 예측 시정거리 비교
weather_data['Predicted Visibility'] = model.predict(X)
plt.figure(figsize=(14, 6))
plt.plot(weather_data['일시'], weather_data['시정(10m)'], label='Actual Visibility', color='blue')
plt.plot(weather_data['일시'], weather_data['Predicted Visibility'], label='Predicted Visibility', color='orange', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Visibility (10m)")      
plt.title("Actual vs Predicted Visibility Over Time") #실제 vs 예측
plt.legend()
plt.xticks(rotation=45)
plt.show()
