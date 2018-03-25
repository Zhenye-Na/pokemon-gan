"""High level pipeline for Pokemon WGAN."""

from __future__ import print_function
from __future__ import absolute_import

import os
import time
import random
import scipy.misc
import numpy as np
import tensorflow as tf
from funcs import *
from train_eval_model import *
from utils.io_tools import read_dataset
from utils.data_tools import process_data

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 2e-4, 'Initial learning rate.')
# flags.DEFINE_float('w_decay_factor', 0.001, 'Weight decay factor.')
flags.DEFINE_integer('HEIGHT', 128, 'Height of images.')
flags.DEFINE_integer('WIDTH', 128, 'Width of images.')
flags.DEFINE_integer('CHANNEL', 3, 'Number of channels of images.')
flags.DEFINE_integer('EPOCH', 5000, 'Number of Epochs to run.')
flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of batch in update.')
flags.DEFINE_string('VERSION', 'newPokemon', 'Directory of output')
flags.DEFINE_string('process_method', 'default',
                    'Preprocess methods, supports [rgb, default, hsv]')
FLAGS = flags.FLAGS
# flags.DEFINE_string('opt_method', 'qp', 'Supports ["iter", "qp"]')


def main(_):
    """High level pipeline.

    This script performs the trainsing, evaling and testing state of the model.
    """
    pp.pprint(flags.FLAGS.__flags)

    # w_decay_factor = FLAGS.w_decay_factor
    HEIGHT = FLAGS.HEIGHT
    WIDTH = FLAGS.WIDTH
    EPOCH = FLAGS.EPOCH
    BATCH_SIZE = FLAGS.BATCH_SIZE
    CHANNEL = FLAGS.CHANNEL
    VERSION = FLAGS.VERSION
    learning_rate = FLAGS.learning_rate
    process_method = FLAGS.process_method
    # newPoke_path = './' + VERSION
    newPoke_path = VERSION

    # Preprocess all of the Pokemon images
    # read_dataset("./data/image_data",
    #              "./data/preprocessed_data", process_method)

    # Process data
    # process_data(HEIGHT, WIDTH, BATCH_SIZE, CHANNEL)

    train(
        HEIGHT=FLAGS.HEIGHT,
        WIDTH=FLAGS.WIDTH,
        EPOCH=FLAGS.EPOCH,
        BATCH_SIZE=FLAGS.BATCH_SIZE,
        CHANNEL=FLAGS.CHANNEL,
        VERSION=FLAGS.VERSION,
        learning_rate=FLAGS.learning_rate,
        process_method=FLAGS.process_method,
        newPoke_path=VERSION
    )

    # Test
    # random_dim = 100
    # with tf.variable_scope('input'):
    #     real_image = tf.placeholder(
    #         tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
    #     random_input = tf.placeholder(
    #         tf.float32, shape=[None, random_dim], name='rand_input')
    #     is_train = tf.placeholder(tf.bool, name='is_train')

    # # wgan
    # fake_image = generator(random_input, random_dim, is_train, CHANNEL)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)
    # ckpt = tf.train.latest_checkpoint('./model/' + VERSION)
    # saver.restore(sess, ckpt)


if __name__ == '__main__':
    tf.app.run()
