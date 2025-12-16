import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os


model_path = 'casting_inspection_model.h5'


test_image_path = './predict_img/cast_def_0_87.jpeg'

if not os.path.exists(model_path):
    print(f"모델 파일이 없습니다: {model_path}")
    exit()


model = tf.keras.models.load_model(model_path)
print(">> 모델 로드 완료")


img = image.load_img(test_image_path, target_size=(300, 300), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)  


prediction = model.predict(img_array)
score = prediction[0][0] # 0~1 사이의 값




print(f"예측 수치: {score:.4f}")

if score < 0.5:
    result_text = "불량 (Defective)"
    confidence = (1 - score) * 100
else:
    result_text = "정상 (OK)"
    confidence = score * 100

print(f"판단: {result_text}")
print(f"확신도: {confidence:.2f}%")


plt.imshow(image.load_img(test_image_path, color_mode='grayscale'), cmap='gray')
plt.title(f"AI Result: {result_text} ({confidence:.1f}%)")
plt.axis('off')
plt.show()