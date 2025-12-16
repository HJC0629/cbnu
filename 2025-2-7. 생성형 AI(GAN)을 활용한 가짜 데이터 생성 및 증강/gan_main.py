import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time


base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, 'generated_images')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

EPOCHS = 50
BATCH_SIZE = 256
BUFFER_SIZE = 60000
NOISE_DIM = 100


print(">> Loading Dataset...")
(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)




def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))


    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

 
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model



def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


generator = make_generator_model()
discriminator = make_discriminator_model()


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss



def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch:04d}.png'))
    plt.close()



seed = tf.random.normal([16, NOISE_DIM])  


gen_loss_history = []
disc_loss_history = []

print(">> Start Training...")

for epoch in range(EPOCHS):
    start = time.time()

    epoch_gen_loss = 0
    epoch_disc_loss = 0
    batch_count = 0

    for image_batch in train_dataset:
        g_loss, d_loss = train_step(image_batch)

        epoch_gen_loss += g_loss
        epoch_disc_loss += d_loss
        batch_count += 1


    gen_loss_history.append(epoch_gen_loss / batch_count)
    disc_loss_history.append(epoch_disc_loss / batch_count)


    if (epoch + 1) % 5 == 0 or epoch == 0:
        generate_and_save_images(generator, epoch + 1, seed)
        print(
            f'Epoch {epoch + 1} | Time: {time.time() - start:.1f}s | Gen Loss: {gen_loss_history[-1]:.4f} | Disc Loss: {disc_loss_history[-1]:.4f}')

print(">> Training Finished.")


plt.figure(figsize=(10, 5))
plt.plot(gen_loss_history, label='Generator Loss', color='blue')
plt.plot(disc_loss_history, label='Discriminator Loss', color='red')
plt.title('GAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(base_dir, 'gan_loss_graph.png'))
print(f">> Saved Loss Graph: gan_loss_graph.png")