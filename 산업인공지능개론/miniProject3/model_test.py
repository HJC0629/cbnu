from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# 새로운 분류명 설정
class_labels = ['car', 'cat', 'dog']

# 저장된 모델 불러오기
loaded_model = load_model('my_model.h5')

# 검증 이미지 디렉토리 설정
validation_dir = r'C:\Users\hjc\PycharmProjects\miniproject3\venv\Scripts\val2'

# 검증 이미지 파일 목록 가져오기
image_files = os.listdir(validation_dir)

# 각 이미지에 대해 예측 수행하고 결과 출력
for image_file in image_files:
    # 이미지 경로 설정
    image_path = os.path.join(validation_dir, image_file)

    # 이미지 불러오기 및 전처리
    img = image.load_img(image_path, target_size=(128, 128))  # 이미지를 128x128 크기로 조정
    img_array = image.img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원을 추가하여 모델에 입력할 수 있는 형태로 변환
    img_array /= 255.  # 이미지를 0과 1 사이의 값으로 정규화

    # 모델 예측
    prediction = loaded_model.predict(img_array)

    # 예측 결과 출력
    predicted_class_index = np.argmax(prediction)  # 가장 높은 확률을 가진 클래스 인덱스
    predicted_class_label = class_labels[predicted_class_index]  # 새로운 분류명으로 변경
    confidence = prediction[0][predicted_class_index]  # 예측된 클래스의 신뢰도

    # 결과 출력
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.barh(class_labels, prediction[0], color='skyblue')
    plt.xlabel('Probability')
    plt.title(f'Predicted class: {predicted_class_label}\nConfidence: {confidence:.2f}')
    plt.tight_layout()
    plt.show()
