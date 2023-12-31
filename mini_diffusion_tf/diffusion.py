import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from mini_diffusion_tf.ops import get_dtype, render_samples

layers = tf.keras.layers


class DiffusionModel(keras.Model):
    def __init__(
        self,
        network,
        ema_network,
        img_size: int,
        max_signal_rate: float = 0.95,
        min_signal_rate: float = 0.01,
        ema: float = 0.995,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.normalizer = layers.Normalization(dtype=tf.float32)
        self.network = network
        self.ema_network = ema_network
        self.max_signal_rate = max_signal_rate
        self.min_signal_rate = min_signal_rate
        self.img_size = img_size
        self.ema = ema

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(self.max_signal_rate)
        end_angle = tf.acos(self.min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(
        self, noisy_images, noise_rates, signal_rates, training, class_labels=None
    ):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        inputs = [noisy_images, tf.reshape(noise_rates, [-1]) ** 2]
        if class_labels is not None:
            inputs.append(class_labels)

        pred_noises = network(inputs, training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(
        self, initial_noise, diffusion_steps, start_step=0, class_labels=None
    ):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(start_step, diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            noisy_images = tf.cast(noisy_images, get_dtype())
            noise_rates = tf.cast(noise_rates, get_dtype())
            signal_rates = tf.cast(signal_rates, get_dtype())

            pred_noises, pred_images = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=False,
                class_labels=class_labels,
            )
            pred_noises = tf.cast(pred_noises, tf.float32)
            pred_images = tf.cast(pred_images, tf.float32)
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps, class_labels=None):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, self.img_size, self.img_size, 3)
        )
        if class_labels is None:
            class_labels = tf.zeros((num_images,), dtype=tf.int32)

        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps, class_labels=class_labels
        )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, data):
        if "class_labels" in self.network.input_names:
            images, class_labels = data
            print(f"Using class labels: {class_labels}")
        else:
            images, class_labels = data, None

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        images = tf.cast(images, tf.float32)
        noises = tf.random.normal(shape=tf.shape(images))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(tf.shape(images)[0], 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        images = tf.cast(images, get_dtype())
        noises = tf.cast(noises, get_dtype())
        noisy_images = tf.cast(noisy_images, get_dtype())
        noise_rates = tf.cast(noise_rates, get_dtype())
        signal_rates = tf.cast(signal_rates, get_dtype())

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=True,
                class_labels=class_labels,
            )

            pred_noises = tf.cast(pred_noises, tf.float32)
            noise_loss = self.loss(noises, pred_noises)  # used for training

            image_loss = self.loss(images, pred_images)  # only used as metric
            scaled_noise_loss = self.optimizer.get_scaled_loss(noise_loss)

        scaled_gradients = tape.gradient(
            scaled_noise_loss, self.network.trainable_weights
        )
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)

        # gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        return self.train_step(data)

    def plot_images(
        self,
        epoch=None,
        logs=None,
        num_rows=3,
        num_cols=6,
        num_diffusion_steps=25,
        class_labels=None,
    ):

        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=num_diffusion_steps,
            class_labels=class_labels,
        )
        title = None
        if epoch is not None:
            title = f"Epoch {epoch} logs={logs}"
        return render_samples(
            num_cols,
            num_rows,
            generated_images,
            as_pillow=True,
            title=title,
            class_labels=class_labels,
        )
