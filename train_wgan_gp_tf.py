import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE


def make_mnist_dataset(batch_size: int, buffer_size: int = 60000):
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 127.5) - 1.0
    x_train = np.expand_dims(x_train, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_generator(latent_dim: int, base_channels: int = 64):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(7 * 7 * base_channels, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((7, 7, base_channels)),
            layers.Conv2DTranspose(base_channels // 2, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(base_channels // 4, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(1, 5, padding="same", activation="tanh"),
        ],
        name="generator",
    )
    return model


def build_discriminator(base_channels: int = 64):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(base_channels, 5, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(base_channels * 2, 5, strides=2, padding="same"),
            layers.LayerNormalization(),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    return model


class WGANGPTrainer:
    def __init__(
        self,
        latent_dim: int,
        batch_size: int,
        n_critic: int,
        gp_weight: float,
        gen: tf.keras.Model,
        disc: tf.keras.Model,
        g_opt: tf.keras.optimizers.Optimizer,
        d_opt: tf.keras.optimizers.Optimizer,
    ):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.gp_weight = gp_weight
        self.generator = gen
        self.discriminator = disc
        self.g_opt = g_opt
        self.d_opt = d_opt

    def gradient_penalty(self, real_images, fake_images):
        epsilon = tf.random.uniform([self.batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = tape.gradient(pred, interpolated)
        grads = tf.reshape(grads, [self.batch_size, -1])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-12)
        return tf.reduce_mean((slopes - 1.0) ** 2)

    @tf.function
    def critic_train_step(self, real_images):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_images = tf.stop_gradient(fake_images)
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            gp = tf.constant(0.0)
            if self.gp_weight > 0:
                gp = self.gradient_penalty(real_images, fake_images)
            d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + self.gp_weight * gp
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))
        wasserstein = tf.reduce_mean(real_logits) - tf.reduce_mean(fake_logits)
        return d_loss, wasserstein

    @tf.function
    def generator_train_step(self):
        noise = tf.random.normal([self.batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_logits)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        return g_loss

    def sample_images(self, num_samples: int, out_path: Path):
        noise = tf.random.normal([num_samples, self.latent_dim])
        generated = self.generator(noise, training=False)
        generated = (generated + 1.0) * 0.5
        generated = tf.clip_by_value(generated, 0.0, 1.0)
        n = int(math.sqrt(num_samples))
        canvas = np.zeros((28 * n, 28 * n))
        imgs = generated.numpy().reshape(num_samples, 28, 28)
        for i in range(n):
            for j in range(n):
                canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = imgs[i * n + j]
        plt.figure(figsize=(n, n))
        plt.axis("off")
        plt.imshow(canvas, cmap="gray")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()


def run_training(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.eager:
        tf.config.run_functions_eagerly(True)

    dataset = make_mnist_dataset(args.batch_size)

    generator = build_generator(args.latent_dim, args.base_channels)
    discriminator = build_discriminator(args.base_channels)

    g_opt = tf.keras.optimizers.Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    d_opt = tf.keras.optimizers.Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)

    trainer = WGANGPTrainer(
        args.latent_dim,
        args.batch_size,
        args.n_critic,
        args.gp_weight,
        generator,
        discriminator,
        g_opt,
        d_opt,
    )

    history: Dict[str, List[float]] = {"g_loss": [], "d_loss": [], "wasserstein": []}
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        g_epoch_losses = []
        d_epoch_losses = []
        w_epoch = []
        for batch_idx, real_images in enumerate(dataset):
            for _ in range(args.n_critic):
                d_loss, wasserstein = trainer.critic_train_step(real_images)
                d_epoch_losses.append(float(d_loss))
                w_epoch.append(float(wasserstein))
            g_loss = trainer.generator_train_step()
            g_epoch_losses.append(float(g_loss))
            global_step += 1
            if args.log_interval and (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"  Step {batch_idx + 1:04d} | G loss: {g_epoch_losses[-1]:.4f} "
                    f"| D loss: {d_epoch_losses[-1]:.4f} | W-dist: {w_epoch[-1]:.4f}",
                    flush=True,
                )
            if args.max_batches and (batch_idx + 1) >= args.max_batches:
                break
        history["g_loss"].append(float(np.mean(g_epoch_losses)))
        history["d_loss"].append(float(np.mean(d_epoch_losses)))
        history["wasserstein"].append(float(np.mean(w_epoch)))
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_path = Path(args.output_dir) / f"samples_epoch_{epoch:03d}.png"
            trainer.sample_images(args.sample_grid ** 2, sample_path)
        print(
            f"Epoch {epoch}/{args.epochs} | G loss: {history['g_loss'][-1]:.4f} "
            f"| D loss: {history['d_loss'][-1]:.4f} | W-dist: {history['wasserstein'][-1]:.4f}",
            flush=True,
        )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    trainer.sample_images(args.sample_grid ** 2, Path(args.output_dir) / "final_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="outputs/wgan_gp_run")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--sample-grid", type=int, default=6)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Limit number of batches per epoch for faster experiments")
    parser.add_argument("--log-interval", type=int, default=0,
                        help="Print mini-batch metrics every N steps")
    parser.add_argument("--eager", action="store_true", help="Run training without tf.function tracing")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    run_training(args)
