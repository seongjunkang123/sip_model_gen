import tensorflow as tf
from keras import Model
import keras

class CTGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, num_classes):
        super(CTGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(CTGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

        # metrics
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data):
        real_voc, real_labels = data
        batch_size = tf.shape(real_voc)[0]

        # random latent vector
        latent_vec = tf.random.normal(shape=(batch_size, self.latent_dim))

        # train discriminator
        with tf.GradientTape() as tape:
            fake_vocs = self.generator([latent_vec, real_labels], training=True)

            # discriminator results
            real_pred = self.discriminator([real_voc, real_labels], training=True)
            fake_pred = self.discriminator([fake_vocs, real_labels], training=True)

            # discriminator loss
            real_loss = self.loss_fn(tf.ones_like(real_pred), real_pred)
            fake_loss = self.loss_fn(tf.zeros_like(fake_pred), fake_pred)
            d_loss = (real_loss + fake_loss) / 2

        # edit weights
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        # train generator
        latent_vec = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_vocs = self.generator([latent_vec, real_labels], training=True)
            fake_pred = self.discriminator([fake_vocs, real_labels], training=True)

            g_loss = self.loss_fn(tf.ones_like(fake_pred), fake_pred)

        # edit weights
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        # update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }

    def generate(self, labels, num_samples=None):
        if num_samples is None:
            num_samples = tf.shape(labels)[0]

        noise = tf.random.normal(shape=(num_samples, self.latent_dim))
        generated = self.generator([noise, labels], training=False)
        return generated
