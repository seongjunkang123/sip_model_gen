from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

#import data
(train_data, _), (_, _) = mnist.load_data()

# normalize
train_data = train_data.astype('float32') / 255.0
train_data = np.expand_dims(train_data, -1)
print(train_data[0].shape)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ],
    name='discriminator'
)

# print(discriminator.summary())

latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(7 * 7 * 128),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, kernel_size=5, padding="same", activation="sigmoid"),
    ],
    name='generator'
)

# print(generator.summary())

discriminator.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = keras.Model(gan_input, gan_output)

gan.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    loss="binary_crossentropy"
)

batch_size = 128
epochs = 500
half_batch = batch_size // 2

for epoch in range(epochs):

    # Sample real images
    idx = np.random.randint(0, train_data.shape[0], half_batch)
    real_images = train_data[idx]
    real_labels = np.ones((half_batch, 1))

    # Generate fake images
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_images = generator.predict(noise, verbose=0)
    fake_labels = np.zeros((half_batch, 1))

    # Train discriminator on real and fake
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    misleading_labels = np.ones((batch_size, 1))  # generator wants "real"

    g_loss = gan.train_on_batch(noise, misleading_labels)

    print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {d_loss[1] * 100:.2f}%] [G loss: {g_loss:.4f}]")

def show_generated_images(generator, epoch):
    noise = np.random.normal(0, 1, (16, latent_dim))
    images = generator.predict(noise)
    images = images.reshape(16, 28, 28)

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.show()

show_generated_images(generator, epochs - 1)