import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random

# Config
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 1
NUM_SAMPLES = 2000
EPOCHS = 10
BATCH_SIZE = 32

base_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(base_dir, 'unet_model.weights.h5')


def generate_data(num_samples):
    # Init arrays
    images = np.zeros((num_samples, IMG_HEIGHT, IMG_WIDTH, CHANNELS), dtype=np.float32)
    masks = np.zeros((num_samples, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

    print(f">> Generating {num_samples} synthetic images...")

    for n in range(num_samples):
        # Background noise
        noise = np.random.randint(0, 50, (IMG_HEIGHT, IMG_WIDTH))
        img = np.full((IMG_HEIGHT, IMG_WIDTH), 100, dtype=np.uint8)
        img = cv2.add(img, noise.astype(np.uint8))

        # Defect mask
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

        # Random scratches
        num_scratches = random.randint(1, 4)
        for _ in range(num_scratches):
            x1, y1 = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
            x2, y2 = random.randint(0, IMG_WIDTH), random.randint(0, IMG_HEIGHT)
            thickness = random.randint(1, 2)

            # Draw scratch (dark on img, white on mask)
            cv2.line(img, (x1, y1), (x2, y2), (50, 50, 50), thickness)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

        # Normalize
        images[n] = img.reshape(IMG_HEIGHT, IMG_WIDTH, 1) / 255.0
        masks[n] = mask.reshape(IMG_HEIGHT, IMG_WIDTH, 1) / 255.0

    return images, masks


def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c4)

    # Decoder
    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u3 = layers.concatenate([u3, c3])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u3)
    c5 = layers.Dropout(0.2)(c5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c5)

    u2 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
    c6 = layers.Dropout(0.1)(c6)
    c6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    u1 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = layers.concatenate([u1, c1])
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
    c7 = layers.Dropout(0.1)(c7)
    c7 = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    # Output (Pixel-wise classification)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)

    return models.Model(inputs=[inputs], outputs=[outputs])


if __name__ == "__main__":
    # Prepare Data
    X, Y = generate_data(NUM_SAMPLES)

    split_idx = int(NUM_SAMPLES * 0.9)
    x_train, x_val = X[:split_idx], X[split_idx:]
    y_train, y_val = Y[:split_idx], Y[split_idx:]

    print(f">> Train shape: {x_train.shape}, Val shape: {x_val.shape}")

    # Train
    model = build_unet((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(">> Start training...")
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save_weights(save_path)
    print(">> Model saved.")

    # Visualize Inference
    print(">> Saving result image...")
    test_indices = random.sample(range(len(x_val)), 3)
    preds = model.predict(x_val[test_indices])

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(test_indices):
        # Input
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(x_val[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Input")
        plt.axis('off')

        # Ground Truth
        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(y_val[idx].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

        # Prediction
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(preds[i].reshape(IMG_HEIGHT, IMG_WIDTH) > 0.5, cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

    save_img_path = os.path.join(base_dir, 'segmentation_result.png')
    plt.savefig(save_img_path)
    print(f">> Saved to {save_img_path}")