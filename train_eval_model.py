"""Train model and eval model helpers."""
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from funcs import *
from utils.data_tools import process_data


def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)


def generator(input, random_dim, is_train, CHANNEL, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32  # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable(
            'w1',
            shape=[
                random_dim,
                s4 * s4 * c4],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
        # Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(
            conv1,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        # Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(
            act1, c8, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(
            conv2,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(
            act2, c16, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(
            conv3,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(
            act3, c32, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(
            conv4,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(
            act4, c64, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv5')
        bn5 = tf.contrib.layers.batch_norm(
            conv5,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        # 128*128*3
        conv6 = tf.layers.conv2d_transpose(
            act5, output_dim, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
    return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        # Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(
            input, c2, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv1')
        bn1 = tf.contrib.layers.batch_norm(
            conv1,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn1')
        act1 = lrelu(conv1, n='act1')
        # Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(
            act1, c4, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv2')
        bn2 = tf.contrib.layers.batch_norm(
            conv2,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn2')
        act2 = lrelu(bn2, n='act2')
        # Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(
            act2, c8, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv3')
        bn3 = tf.contrib.layers.batch_norm(
            conv3,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn3')
        act3 = lrelu(bn3, n='act3')
        # Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(
            act3, c16, kernel_size=[
                5, 5], strides=[
                2, 2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(
            conv4,
            is_training=is_train,
            epsilon=1e-5,
            decay=0.9,
            updates_collections=None,
            scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

        w2 = tf.get_variable('w2',
                             shape=[fc1.shape[-1],
                                    1],
                             dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        # acted_out = tf.nn.sigmoid(logits)
    return logits  # , acted_out


def load(checkpoint_dir):
    """Load model."""
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0


def train(HEIGHT, WIDTH, EPOCH, BATCH_SIZE, CHANNEL, VERSION, learning_rate, process_method, newPoke_path):
    """Train model."""
    random_dim = 100

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
    # This optimizes the generator.
    g_loss = -tf.reduce_mean(fake_result)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate).minimize(g_loss, var_list=g_vars)
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

    # Prepare training
    save_path = saver.save(sess, "./tmp/model.ckpt")
    # save_path = saver.save(sess, "/output/tmp/model.ckpt")
    save_path2 = './model/' + VERSION
    ckpt = tf.train.latest_checkpoint(save_path2)
    saver.restore(sess, save_path2)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num: %d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' %
          (batch_size, batch_num, EPOCH))
    print('Start training...')

    # Training
    for i in range(EPOCH):
        print("EPOCH: %d" % (i))
        for j in range(batch_num):
            t = time.time()
            # print("batch_num: %d" % (j))
            d_iters = 5  # Iterations for discriminator
            g_iters = 1  # Iterations for generator

            train_noise = np.random.uniform(-1.0, 1.0,
                                            size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                # print("d_iters: %d" % (k))
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

            elapsed = time.time() - t
            # print({"metric": "discriminator loss", "value": dLoss})
            # print({"metric": "generator loss", "value": gLoss})

            print("Training: [%d / %d], d_loss: %f, g_loss: %f, elapsed time: %5f seconds" %
                  (i, j, dLoss, gLoss, elapsed))

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


def test():
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
