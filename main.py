"""High level pipeline for Pokemon WGAN."""

from __future__ import print_function
from __future__ import absolute_import

import os
import random
import scipy.misc
import numpy as np
import tensorflow as tf
from utills import *
from train_eval_model import *
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
flags.DEFINE_string('VERSION', 'newPokemon', 'Directory of output')


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
    VERSION = FLAGS.VERSION

    newPoke_path = './' + VERSION
    outputdir = './output'
    # If dest directory not exists, create dir
    # if not os.path.isdir(outputdir):
    #     os.mkdir(outputdir)

    # Read image into dictionary
    # data = train_set = read_dataset("data/train.txt", "data/image_data/")

    # Preprocess all of the Pokemon images
    read_dataset("./data/image_data_demo",
                 "./data/preprocessed_data", 'default')

    # Process data
    process_data(HEIGHT, WIDTH, BATCH_SIZE, CHANNEL)

    random_dim = 100
    # print(os.environ['CUDA_VISIBLE_DEVICES'])

    with tf.variable_scope('input'):
        # real and fake image placholders
        real_image = tf.placeholder(
            tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(
            tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train, CHANNEL)

    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)

    # This optimizes the discriminator.
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(
        learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(
        learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data(HEIGHT, WIDTH, BATCH_SIZE, CHANNEL)

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + VERSION)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' %
          (batch_size, batch_num, EPOCH))
    print('start training...')
    for i in range(EPOCH):
        print(i)
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0,
                                            size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                # wgan clip weights
                sess.run(d_clip)

                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                # train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            print('train:[%d/%d], d_loss:%f, g_loss:%f' % (i, j, dLoss, gLoss))

        # save check point every 500 epoch
        if i % 500 == 0:
            if not os.path.exists('./model/' + VERSION):
                os.makedirs('./model/' + VERSION)
            saver.save(sess, './model/' + VERSION + '/' + str(i))
        if i % 50 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0,
                                             size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={
                               random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [8, 8], newPoke_path +
                        '/epoch' + str(i) + '.jpg')

            # print('train:[%d], d_loss:%f, g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)

    # def test():
    random_dim = 100
    with tf.variable_scope('input'):
        real_image = tf.placeholder(
            tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(
            tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train, CHANNEL)
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.latest_checkpoint('./model/' + VERSION)
    saver.restore(sess, ckpt)


if __name__ == '__main__':
    tf.app.run()
