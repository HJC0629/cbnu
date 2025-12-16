import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os


train_dir = './casting_data/train'
test_dir = './casting_data/test'

img_size = (300, 300)
batch_size = 32

if not os.path.exists(train_dir):
    print(f"[Error] 학습 데이터 폴더를 찾을 수 없습니다: {train_dir}")
    exit()


train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)


test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n>> 학습을 시작합니다")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'b-', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'b-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.grid(True)


graph_save_name = 'training_result_graph.png'
plt.savefig(graph_save_name, dpi=300)  # dpi=300은 고화질 저장
print(f"\n>> [성공] 그래프 이미지가 저장되었습니다: {graph_save_name}")


plt.show()


model_save_name = 'casting_inspection_model.h5'
model.save(model_save_name)
print(f"\n>> [성공] 모델 파일이 저장되었습니다: {model_save_name}")