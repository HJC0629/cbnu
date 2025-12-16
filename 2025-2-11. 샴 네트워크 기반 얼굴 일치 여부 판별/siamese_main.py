import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import os

IMG_SIZE = 64
base_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_dir, 'siamese_model.weights.h5')


def make_pairs(images, labels):
    pair_images = []
    pair_labels = []

    unique_classes = np.unique(labels)
    idx_map = {label: np.where(labels == label)[0] for label in unique_classes}

    print(f">> Generating pairs from {len(images)} images...")

    for idxA in range(len(images)):
        current_img = images[idxA]
        label = labels[idxA]

        if len(idx_map[label]) > 1:
            possibles = list(idx_map[label])
            if idxA in possibles: possibles.remove(idxA)
            idxB = random.choice(possibles)
        else:
            idxB = idxA

        pos_img = images[idxB]

        pair_images.append([current_img, pos_img])
        pair_labels.append(1)

        possible_neg_classes = list(unique_classes)
        if label in possible_neg_classes:
            possible_neg_classes.remove(label)

        if len(possible_neg_classes) > 0:
            neg_label = random.choice(possible_neg_classes)
            idxC = random.choice(idx_map[neg_label])
            neg_img = images[idxC]

            pair_images.append([current_img, neg_img])
            pair_labels.append(0)

    return np.array(pair_images), np.array(pair_labels).astype('float32')


print(">> Loading Olivetti Faces dataset...")
data = fetch_olivetti_faces()
images = data.images
labels = data.target

images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.1, random_state=42, stratify=labels
)

pairs_train, labels_train = make_pairs(x_train, y_train)
pairs_test, labels_test = make_pairs(x_test, y_test)

print(f">> Train Pairs: {pairs_train.shape}, Test Pairs: {pairs_test.shape}")


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_base_network(input_shape):
    input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation=None)(x)
    return models.Model(input, x)


input_shape = (IMG_SIZE, IMG_SIZE, 1)
base_network = build_base_network(input_shape)

img_a = layers.Input(shape=input_shape)
img_b = layers.Input(shape=input_shape)

feat_a = base_network(img_a)
feat_b = base_network(img_b)

distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])

model = models.Model(inputs=[img_a, img_b], outputs=distance)


def contrastive_loss(y_true, y_pred):
    margin = 1
    sq_pred = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sq_pred + (1 - y_true) * margin_sq)


model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])

print(">> Start Training...")
history = model.fit(
    [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
    validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test),
    batch_size=64,
    epochs=15,
    verbose=1
)

model.save_weights(save_path)
print(">> Model saved.")

print(">> Visualizing results...")
idx = np.random.choice(len(pairs_test), 4, replace=False)
sample_pairs = pairs_test[idx]
sample_labels = labels_test[idx]

preds = model.predict([sample_pairs[:, 0], sample_pairs[:, 1]])

plt.figure(figsize=(10, 5))
for i in range(4):
    ax = plt.subplot(2, 2, i + 1)

    img1 = sample_pairs[i, 0].reshape(IMG_SIZE, IMG_SIZE)
    img2 = sample_pairs[i, 1].reshape(IMG_SIZE, IMG_SIZE)
    combined = np.hstack((img1, img2))

    plt.imshow(combined, cmap='gray')

    dist = preds[i][0]
    is_same = dist < 0.5
    true_label = "Same" if sample_labels[i] == 1 else "Diff"
    pred_result = "Same" if is_same else "Diff"

    color = 'green' if (sample_labels[i] == 1) == is_same else 'red'
    plt.title(f"True: {true_label} | Pred: {pred_result}\nDist: {dist:.4f}", color=color)
    plt.axis('off')

plt.tight_layout()
save_img_path = os.path.join(base_dir, 'siamese_result.png')
plt.savefig(save_img_path)
print(f">> Saved to {save_img_path}")