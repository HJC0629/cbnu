import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import os
import joblib  


base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'car data.csv')


model_save_path = os.path.join(base_dir, 'car_price_model.h5')
graph_save_path = os.path.join(base_dir, 'real_car_price_result.png')
preprocessor_save_path = os.path.join(base_dir, 'car_preprocessor.pkl') # [New]


if not os.path.exists(csv_path):
    print(f"[Error] 데이터 파일 없음: {csv_path}")
    exit()

df = pd.read_csv(csv_path)


X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(">> 학습 시작 ")
history = model.fit(X_train, y_train, epochs=200, validation_split=0.2, verbose=0)
print(">> 학습 완료")



model.save(model_save_path)
print(f">> 모델 저장 완료: {os.path.basename(model_save_path)}")


joblib.dump(preprocessor, preprocessor_save_path)
print(f">> 전처리기 저장 완료: {os.path.basename(preprocessor_save_path)}")


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(graph_save_path)
print(">> 그래프 저장 완료")