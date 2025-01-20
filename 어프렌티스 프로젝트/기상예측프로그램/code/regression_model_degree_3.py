import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import re  # 정규 표현식을 사용하여 특수문자 제거

# 데이터 불러오기
data = pd.read_csv('weather_data.csv', encoding='cp949')

# 컬럼 이름의 앞뒤 공백 제거
data.columns = data.columns.str.strip()

# 정확한 컬럼명에 맞게 y 정의
columns_with_sijung = [col for col in data.columns if '시정' in col]
y = data[columns_with_sijung[0]]  # '시정(10m)' 또는 적절한 컬럼명

# 특성 정의
X = data[['기온(°C)', '풍속(m/s)', '일조(hr)', '일사(MJ/m2)', '지면온도(°C)']]
# X의 결측값 처리: 평균값으로 대체
X.fillna(X.mean(), inplace=True)

# y의 결측값 처리: 평균값으로 대체 (필요한 경우)
y.fillna(y.mean(), inplace=True)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 다항회귀 파이프라인 정의
degree = 3  # 다항식 차수
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('linear', LinearRegression())
])

# 하이퍼파라미터 그리드 정의
param_grid = {
    'poly__degree': [2, 3],  # 다항식 차수 (2차와 3차 중 선택)
}

# GridSearchCV를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 최적의 모델
best_model = grid_search.best_estimator_

# 교차 검증 점수
best_score = -grid_search.best_score_  # MSE는 음수로 계산되므로 음수 부호 제거
print(f"Best Cross-validation MSE: {best_score}")

# 테스트 데이터에서 예측
y_pred = best_model.predict(X_test)

# 성능 평가
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# 실제 값과 예측 값 비교
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head())

# 성능 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

# 모델 저장
best_params_str = str(grid_search.best_params_)
best_params_str = re.sub(r'[^a-zA-Z0-9_]', '_', best_params_str)  # 특수문자 제거
model_filename = f"polynomial_regression_model_{best_params_str}.pkl"
joblib.dump(best_model, model_filename)
