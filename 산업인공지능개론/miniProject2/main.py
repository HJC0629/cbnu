
# 필요한 라이브러리 불러오기

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 데이터 불러오기
file_path = r'C:\Users\hjc\PycharmProjects\AI_MiniProject2\venv\Scripts\weekData_2.xlsx'
data = pd.read_excel(file_path, sheet_name=0)
#data = pd.read_excel('생산량데이터.xlsx')

# 특성 이름을 정의
#feature_names = ['week']  # 주어진 데이터의 첫 번째 열이 주차 정보를 나타내는 경우

# 입력 특성과 출력 값 분리
X = data[['week']]
y = data['productionAmount']



# 선형 회귀 모델 생성
#model = LinearRegression()
model = LinearRegression(fit_intercept=False)

# 모델 훈련
model.fit(X, y)

# 다음 주의 주차 값을 예측
다음_주_주차 = 27  # 예측할 다음 주의 주차
다음_주_생산량_예측 = model.predict([[다음_주_주차]])
다음_주_생산량_예측 = int(다음_주_생산량_예측[0])
print(f"다음 주의 생산량 예측: {다음_주_생산량_예측}")

# 훈련 데이터를 사용하여 예측
y_pred_train = model.predict(X)

# 모델 평가
mse_train = mean_squared_error(y, y_pred_train)
mae_train = mean_absolute_error(y, y_pred_train)
r2_train = r2_score(y, y_pred_train)

print("=== 훈련 데이터 성능평가 ===")
print(f"평균 제곱 오차(MSE): {mse_train}")
print(f"평균 절대 오차(MAE): {mae_train}")
print(f"결정 계수(R^2): {r2_train}")

# 테스트 데이터를 사용하여 예측
# 이 부분에서는 테스트 데이터를 따로 가지고 있어야 합니다.
# 여기서는 테스트 데이터가 없으므로 생략하겠습니다.