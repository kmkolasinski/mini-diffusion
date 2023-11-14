import io
from typing import Optional

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

layers = tf.keras.layers


def get_dtype():
    if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
        return tf.float16
    return tf.float32


def buffer_plot_and_get(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    image = Image.open(buf)
    image = image.convert("RGB")
    return image


def render_samples(
    num_cols: int,
    num_rows: int,
    generated_images: list,
    as_pillow: bool = False,
    title: Optional[str] = None,
):
    fig = plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
    if title is not None:
        fig.suptitle(title)
    for row in range(num_rows):
        for col in range(num_cols):
            index = row * num_cols + col
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(generated_images[index])
            plt.axis("off")
    plt.tight_layout()
    if as_pillow:
        image = buffer_plot_and_get(fig)

        plt.cla()
        plt.clf()
        return image
    else:
        plt.show()
        plt.close()
