import tensorflow as tf

AUTO = tf.data.AUTOTUNE


def create_dataset_from_paths(
    image_paths: list,
    batch_size: int,
    image_size: tuple[int, int],
    random_crop_size: int,
    shuffle_buffer_size: int = 10000,
) -> tf.data.Dataset:

    def process_image(image_path: str) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.image.random_crop(image, (random_crop_size, random_crop_size, 3))
        image = image / 255.0
        return image

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.repeat(-1).shuffle(shuffle_buffer_size)
    ds = ds.map(process_image, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size).prefetch(AUTO)
    return ds

