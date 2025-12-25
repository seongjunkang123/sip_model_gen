import time
from tqdm import tqdm
import tensorflow as tf 
from tensorflow import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

tags = [0,1,2,3,4,5,6,7,8,9]
epochs = 10
batch_size = 16
latent_dim = 128
classes = len(tags)
img_size = 28

#import data
(train_data, train_labels), (_, _) = mnist.load_data()

# normalize
train_data = train_data.astype('float32') / 255.0
train_data = np.expand_dims(train_data, -1)
# print(train_data[0].shape)

dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

# show figure
plt.figure(figsize=(2,2))
idx = np.random.randint(0,len(train_data))
img = image.array_to_img(train_data[idx], scale=True)
plt.imshow(img)
plt.axis('off')
plt.title(tags[train_labels[idx]])
plt.show()

bce = tf.keras.losses.BinaryCrossentropy()

#loss function and optimizers
def d_loss(real, fake):
    real_loss = bce(tf.ones_like(real), real)
    fake_loss = bce(tf.zeros_like(fake), fake)
    total_loss = real_loss + fake_loss
    return total_loss

def g_loss(pred):
    return bce(tf.ones_like(pred), pred)

d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

def build_generator():
    label = keras.Input(shape=(1,))
    li = layers.Embedding(classes, 50)(label)

    nodes = 7 * 7
    li = layers.Dense(nodes)(li)
    li = layers.Reshape((7, 7, 1))(li)
    lat = layers.Input(shape=(latent_dim,))

    nodes = 128 * 7 * 7
    gen = layers.Dense(nodes)(lat)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((7, 7, 128))(gen)
    gen = layers.Concatenate()([gen, li])

    gen = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Conv2D(1, kernel_size=5, padding="same", activation="sigmoid")(gen)

    model = keras.Model([lat, label], gen)
    return model

def build_discriminator():
    label = keras.Input(shape=(1,))
    li = layers.Embedding(classes, 50)(label)

    nodes = img_size * img_size
    li = layers.Dense(nodes)(li)
    li = layers.Reshape((img_size, img_size, 1))(li)

    image = layers.Input(shape=(img_size, img_size, 1))
    dis = layers.Concatenate()([image, li])

    dis = layers.Conv2D(128, kernel_size=5, padding="same")(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(dis)
    dis = layers.LeakyReLU(alpha=0.2)(dis)
    dis = layers.Flatten()(dis)
    dis = layers.Dropout(0.4)(dis)
    dis = layers.Dense(1, activation="sigmoid")(dis)

    model = keras.Model([image, label], dis)
    return model

generator = build_generator()
discriminator = build_discriminator()

@tf.function
def train_step(dataset):
    real_images, real_labels = dataset

    real_labels = tf.reshape(real_labels, (-1, 1))

    latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    generated_images = generator([latent_vectors, real_labels])

    with tf.GradientTape() as tape:
        pred_fake = discriminator([generated_images, real_labels])
        pred_real = discriminator([real_images, real_labels])

        d_loss_value = d_loss(pred_real, pred_fake)

    grads = tape.gradient(d_loss_value, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

    latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))

    with tf.GradientTape() as tape:
        generated_images = generator([latent_vectors, real_labels])
        pred_fake = discriminator([generated_images, real_labels])

        g_loss_value = g_loss(pred_fake)

    grads = tape.gradient(g_loss_value, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return d_loss_value, g_loss_value

def show_samples(num_samples, n_class, g_model):
    fig, axes = plt.subplots(10,num_samples, figsize=(10,20)) 
    fig.tight_layout()
    fig.subplots_adjust(wspace=None, hspace=0.2)

    # Fix: ensure axes is always 2D
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for l in np.arange(10):
      random_noise = tf.random.normal(shape=(num_samples, latent_dim))
      label = tf.ones(num_samples) * l  # Fix: shape (num_samples, 1)
      gen_imgs = g_model.predict([random_noise, label])
      for j in range(gen_imgs.shape[0]):
        img = image.array_to_img(gen_imgs[j], scale=True)
        axes[l,j].imshow(img)
        axes[l,j].yaxis.set_ticks([])
        axes[l,j].xaxis.set_ticks([])

        if j ==0:
          axes[l,j].set_ylabel(tags[l])
    plt.show()

def train(dataset, epochs=epochs):
   for epoch in range(epochs):
    print(f"epoch {epoch+1}/{epochs}")

    d_loss_list = []
    g_loss_list = []

    start = time.time()

    i = 0
    for image_batch in tqdm(dataset):
      d_loss_value, g_loss_value = train_step(image_batch)
      d_loss_list.append(d_loss_value.numpy())
      g_loss_list.append(g_loss_value.numpy())
      i += 1

    if (epoch + 1) % 10 == 0:
        show_samples(3, classes, generator)

    print(f"Time for epoch {epoch + 1}: {int(time.time() - start)}s")
    print(f"Discriminator loss: {np.mean(d_loss_list):.4f}")
    print(f"Generator loss: {np.mean(g_loss_list):.4f}")
    
train(dataset, epochs=epochs)