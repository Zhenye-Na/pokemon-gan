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


def main(_):
    """High level pipeline.

    This script performs the trainsing, evaling and testing state of the model.
    """
    pp.pprint(flags.FLAGS.__flags)

    HEIGHT = FLAGS.HEIGHT
    WIDTH = FLAGS.WIDTH
    EPOCH = FLAGS.EPOCH
    BATCH_SIZE = FLAGS.BATCH_SIZE
    CHANNEL = FLAGS.CHANNEL
    VERSION = FLAGS.VERSION
    learning_rate = FLAGS.learning_rate
    process_method = FLAGS.process_method
    # newPoke_path = './' + VERSION
    newPoke_path = './' + FLAGS.VERSION

    # Training
    train(
        HEIGHT=FLAGS.HEIGHT,
        WIDTH=FLAGS.WIDTH,
        EPOCH=FLAGS.EPOCH,
        BATCH_SIZE=FLAGS.BATCH_SIZE,
        CHANNEL=FLAGS.CHANNEL,
        VERSION=FLAGS.VERSION,
        learning_rate=FLAGS.learning_rate,
        process_method=FLAGS.process_method,
        newPoke_path=newPoke_path
    )


if __name__ == '__main__':
    tf.app.run()
