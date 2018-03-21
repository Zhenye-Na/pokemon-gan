"""Input and output helpers to load in data."""

import tensorflow as tf


def process_data():
    """Process data.

    Args:
        None.

    Returns:
        images_batch(list): A list of image tensors with the types as tensors

        num_images(int): Number of images in training set.

    """
    # current_dir = os.getcwd()
    current_dir = '../'
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'data/preprocessed_data')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir, each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype=tf.string)

    images_queue = tf.train.slice_input_producer(
        [all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels=CHANNEL)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise'))
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT, WIDTH, CHANNEL])

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    images_batch = tf.train.shuffle_batch(
        [image], batch_size=BATCH_SIZE,
        num_threads=4, capacity=200 + 3 * BATCH_SIZE,
        min_after_dequeue=200)
    num_images = len(images)

    return images_batch, num_images
