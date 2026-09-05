import tensorflow as tf
from keras import Model
import keras


class CTGAN(Model):
    def __init__(self, discriminator, generator, latent_dim, num_classes, n_critic=5, gp_weight=10.0):
        super(CTGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.n_critic = n_critic        # critic updates per generator update
        self.gp_weight = gp_weight      # gradient penalty weight

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super(CTGAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        # metrics
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.gp_metric = keras.metrics.Mean(name="gp")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric, self.gp_metric]

    def _gradient_penalty(self, real_voc, fake_voc, labels):
        # calculate gradient penalty on top of wasserstein loss
        batch_size = tf.shape(real_voc)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_voc + (1 - alpha) * fake_voc

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)

        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_voc, real_labels = data
        batch_size = tf.shape(real_voc)[0]

        # train discriminator
        for _ in range(self.n_critic):
            latent_vec = tf.random.normal(shape=(batch_size, self.latent_dim))

            with tf.GradientTape() as tape:
                fake_vocs = self.generator([latent_vec, real_labels], training=True)

                real_pred = self.discriminator([real_voc, real_labels], training=True)
                fake_pred = self.discriminator([fake_vocs, real_labels], training=True)

                # wasserstein loss: maximize E[D(real)] - E[D(fake)]
                # minimizing: E[D(fake)] - E[D(real)] + GP
                d_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

                # gradient penalty
                gp = self._gradient_penalty(real_voc, fake_vocs, real_labels)
                d_loss = d_loss + self.gp_weight * gp

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # train generator
        latent_vec = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            fake_vocs = self.generator([latent_vec, real_labels], training=True)
            fake_pred = self.discriminator([fake_vocs, real_labels], training=True)

            # maximize D(fake) → minimize -E[D(fake)]
            g_loss = -tf.reduce_mean(fake_pred)

        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )

        # update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        self.gp_metric.update_state(gp)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "gp": self.gp_metric.result(),
        }

    def generate(self, labels, num_samples=None):
        if num_samples is None:
            num_samples = tf.shape(labels)[0]

        noise = tf.random.normal(shape=(num_samples, self.latent_dim))
        generated = self.generator([noise, labels], training=False)
        return generated
