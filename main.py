"""High level pipeline for Pokemon WGAN."""

from __future__ import print_function
from __future__ import absolute_import

import os
import random
import scipy.misc
import numpy as np
import tensorflow as tf
from utils.io_tools import read_dataset
from utils.data_tools import process_data

slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
# flags.DEFINE_float('w_decay_factor', 0.001, 'Weight decay factor.')
flags.DEFINE_integer('HEIGHT', 128, 'Height of images.')
flags.DEFINE_integer('WIDTH', 128, 'Width of images.')
flags.DEFINE_integer('CHANNEL', 3, 'Number of channels of images.')
flags.DEFINE_integer('EPOCH', 5000, 'Number of Epochs to run.')
flags.DEFINE_integer('BATCH_SIZE', 64, 'Size of batch in update.')
# flags.DEFINE_string(
#     'feature_type',
#     'default',
#     'Feature type, supports [raw, default, custom]')
# flags.DEFINE_string('opt_method', 'qp', 'Supports ["iter", "qp"]')



def main(_):
    """High level pipeline.

    This script performs the trainsing, evaling and testing state of the model.
    """
    # learning_rate = FLAGS.learning_rate
    # w_decay_factor = FLAGS.w_decay_factor
    HEIGHT = FLAGS.HEIGHT
    WIDTH = FLAGS.WIDTH
    EPOCH = FLAGS.EPOCH
    BATCH_SIZE = FLAGS.BATCH_SIZE
    CHANNEL = FLAGS.CHANNEL

    outputdir = './output'
    # If dest directory not exists, create dir
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    # Read image into dictionary
    # data = train_set = read_dataset("data/train.txt", "data/image_data/")

    # Preprocess all of the Pokemon images
    read_dataset("./data/image_data", "./data/preprocessed_data", 'default')











if __name__ == '__main__':
    tf.app.run()