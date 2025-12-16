import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kiwipiepy import Kiwi
import os


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'spam_model.h5')
tokenizer_path = os.path.join(base_dir, 'spam_tokenizer.pkl')


max_length = 25
trunc_type = 'post'
padding_type = 'post'


# 파일 존재 여부 확인
if not os.path.exists(model_path):
    print(f"[Error] 모델 파일이 없습니다: {model_path}")
    print("먼저 spam_filter_main.py를 실행해서 학습을 완료해주세요.")
    exit()

print(">> 시스템 로딩 중...")


model = tf.keras.models.load_model(model_path, compile=False)


with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


kiwi = Kiwi()

print(">> 준비 완료! 스팸인지 궁금한 메시지를 입력하세요.")
print(">> 종료하려면 'quit' 입력")
print("=" * 40)



def preprocess_input(text):

    if not text: return ""


    result = kiwi.tokenize(text)
    keywords = []
    for token in result:
        # 명사(N), 동사/형용사(V), 어근(XR)만 추출
        if token.tag.startswith('N') or token.tag.startswith('V') or token.tag.startswith('XR'):
            keywords.append(token.form)


    return " ".join(keywords)



while True:

    user_input = input("\n[메시지 입력]: ")

    if user_input.lower() == 'quit':
        print(">> 프로그램을 종료합니다.")
        break

    if not user_input.strip():
        continue


    clean_text = preprocess_input(user_input)



    seq = tokenizer.texts_to_sequences([clean_text])


    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


    score = model.predict(padded, verbose=0)[0][0]
    percentage = score * 100


    print("-" * 35)
    if score >= 0.5:
        print(f"[스팸]입니다! (확률: {percentage:.2f}%)")
        print("   -> 주의: 사기, 광고, 피싱 의심")
    else:
        print(f" [정상]입니다. (스팸 확률: {percentage:.2f}%)")
    print("-" * 35)