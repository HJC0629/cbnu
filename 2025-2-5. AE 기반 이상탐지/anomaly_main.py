import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os


base_dir = os.path.dirname(os.path.abspath(__file__))

model_save_path = os.path.join(base_dir, 'anomaly_autoencoder.weights.h5')
dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"



dataframe = pd.read_csv(dataset_url, header=None)
raw_data = dataframe.values


labels = raw_data[:, -1]
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)

# 정규화 (MinMax)
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)


train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]
anomalous_test_data = test_data[~test_labels]

print(f">> 정상 학습 데이터: {len(normal_train_data)}개")
print(f">> 비정상 테스트 데이터: {len(anomalous_test_data)}개")



class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")])


        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(140, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = AnomalyDetector()
model.compile(optimizer='adam', loss='mae')



history = model.fit(normal_train_data, normal_train_data,
                    epochs=20,
                    batch_size=512,
                    validation_data=(test_data, test_data),
                    shuffle=True, verbose=1)


model.save_weights(model_save_path)
print(f">> 모델 가중치 저장 완료: {os.path.basename(model_save_path)}")


reconstructions = model.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
threshold = np.mean(train_loss) + np.std(train_loss)
print(f"\n>> 이상 탐지 임계값(Threshold): {threshold:.4f}")



def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)


def plot_prediction(data, title, filename):
    encoded_data = model.encoder(data)
    decoded_data = model.decoder(encoded_data)

    plt.figure(figsize=(10, 5))
    plt.plot(data[0], 'b', label='Input (Original)')
    plt.plot(decoded_data[0], 'r', label='Reconstruction (AI)')
    plt.fill_between(np.arange(140), data[0], decoded_data[0], color='lightcoral', label='Error')
    plt.legend()
    plt.title(title)

    # 그래프 저장
    save_full_path = os.path.join(base_dir, filename)
    plt.savefig(save_full_path)
    print(f">> 그래프 저장 완료: {filename}")




plot_prediction(normal_test_data, "Normal Data Reconstruction", "anomaly_result_normal.png")


plot_prediction(anomalous_test_data, "Anomalous Data Reconstruction", "anomaly_result_abnormal.png")


preds = predict(model, test_data, threshold)
print(f"\n>> 최종 정확도(Accuracy): {accuracy_score(test_labels, preds):.4f}")