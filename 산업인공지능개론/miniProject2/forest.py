from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# 데이터 불러오기
file_path = r'C:\Users\hjc\PycharmProjects\AI_MiniProject2\venv\Scripts\weekData_2.xlsx'
data = pd.read_excel(file_path, sheet_name=0)

# 특성과 타겟 변수 분리
X = data.drop(columns=["week"])
y = data["productionAmount"]
print(data.head())

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("평균 제곱 오차(MSE):", mse)
print("평균 절대 오차(MAE):", mae)
print("결정 계수(R^2):", r2)

# 다음 주의 생산량 예측
next_week_feature = [[54]]  # 다음 주의 주차를 입력하세요.
next_week_production_prediction = model.predict(next_week_feature)
print("다음 주의 생산량 예측:", next_week_production_prediction[0])

