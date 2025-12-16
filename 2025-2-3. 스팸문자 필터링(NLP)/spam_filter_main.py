import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from kiwipiepy import Kiwi
import matplotlib.pyplot as plt
import os
import pickle


base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'spam_data.csv')
model_save_path = os.path.join(base_dir, 'spam_model.h5')
tokenizer_save_path = os.path.join(base_dir, 'spam_tokenizer.pkl')
graph_save_path = os.path.join(base_dir, 'spam_training_result.png')


vocab_size = 2000
embedding_dim = 32
max_length = 25
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


if not os.path.exists(csv_path):
    print(f"[Error] 데이터 파일이 없습니다: {csv_path}")
    exit()


data = pd.read_csv(csv_path, encoding='cp949')
print(f">> 전체 데이터 개수: {len(data)}개")


kiwi = Kiwi()

print(">> 형태소 분석 수행 중... (명사/동사/어근 추출)")
def extract_keywords(text):
    if not isinstance(text, str): return ""
    result = kiwi.tokenize(text)
    keywords = []
    for token in result:

        if token.tag.startswith('N') or token.tag.startswith('V') or token.tag.startswith('XR'):
            keywords.append(token.form)
    return " ".join(keywords)

data['processed_text'] = data['text'].apply(extract_keywords)

sentences = data['processed_text'].tolist()
labels = data['label'].tolist()


train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print(f">> 학습용 데이터: {len(train_sentences)}개")
print(f">> 검증용 데이터: {len(test_sentences)}개 (AI가 처음 보는 문제)")


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences) 


train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sequences = tokenizer.texts_to_sequences(test_sentences)


train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


train_padded = np.array(train_padded)
train_labels = np.array(train_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)


model = tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3), # 과적합 방지

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


print("\n>> 학습 시작 ")

history = model.fit(train_padded, train_labels,
                    epochs=100,
                    validation_data=(test_padded, test_labels),
                    verbose=0) # 로그 숨김


loss, acc = model.evaluate(test_padded, test_labels, verbose=0)
print("\n" + "="*30)
print(f" 최종 테스트 정확도: {acc*100:.2f}%")
print("="*30)

if acc < 0.8:
    print("Tip: 데이터가 더 필요하거나 학습 횟수(Epochs)를 늘려야 합니다.")
else:
    print("Tip: 훌륭한 성능입니다.")


model.save(model_save_path)
with open(tokenizer_save_path, 'wb') as handle:
    pickle.dump(tokenizer, handle)
print(f">> 모델 저장 완료: {os.path.basename(model_save_path)}")

# 결과 그래프
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r-', label='Test Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-', label='Train Loss')
plt.plot(history.history['val_loss'], 'r-', label='Test Loss')
plt.title('Loss')
plt.legend()

plt.savefig(graph_save_path)
plt.show()