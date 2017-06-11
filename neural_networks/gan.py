""" Generative Adversarial Network implementation.

This is in fact the Least-Squares GAN, as I found it
yields best results so far. However, the GAN market
is developing rapidly.
"""
import sys
import time
import itertools
from typing import List
from tempfile import NamedTemporaryFile

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf

from neural_networks import nn


class GAN(object):
    def __init__(self, x_dim, noise_dim, g_arch=[128, 128],
        g_ntype='plain', d_arch=[128, 128], d_ntype='plain', **kwargs):
        # Bookkeeping.
        self.x_dim = x_dim
        self.noise_dim = noise_dim
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='x_input')
        self.z_tf = tf.placeholder(
            tf.float32, [None, noise_dim], name='z_input')
        self.lr_tf = tf.placeholder(
            tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(
            tf.float32, name='dropout_rate')
        self.g_arch = g_arch + [x_dim]
        self.g_ntype = g_ntype
        self.d_arch = d_arch + [1]
        self.d_ntype = d_ntype
        
        # Define the Generator, Discriminator, and their losses.
        self.y_from_x, self.y_from_z, self.x_from_z = self.define_gan()
        with tf.variable_scope('g_loss'):
            self.g_loss_tf = self.define_gloss()
            self.g_train_tf = self.define_gtrain()
        with tf.variable_scope('d_loss'):
            self.d_loss_tf = self.define_dloss()
            self.d_train_tf = self.define_dtrain()

        # Define the data scalers.
        self.scaler_x = self.define_scalers()

    def define_scalers(self):
        return MinMaxScaler(feature_range=(-1, 1))

    def define_nn(self, arch, ntype, in_tf):
        out = in_tf
        # First, define how noise propagates through the generator.
        if ntype not in ['residual', 'highway', 'plain']:
            raise ValueError('Network type not available.')
        if ntype != 'plain':
            # Check that (for Highway and Residual nets) layers have equal size.
            for layer_id in range(len(arch)-1):
                if arch[layer_id] != arch[0]:
                    raise ValueError('Highway and Residual nets require all' 
                        'layers to be the same size.')
            # Make sure the input has the same size too.
            with tf.variable_scope('layerreshape'):
                out = nn.fully_connected(out, arch[0])

        for layer_id in range(len(arch)):
            n_out = arch[layer_id]
            with tf.variable_scope('layer{}'.format(layer_id)):
                out_transform = out

                # Transform the input.
                with tf.variable_scope('transform'):
                    if layer_id < len(arch) - 1:
                        out_transform = tf.nn.elu(out_transform)
                    out_transform = nn.fully_connected(out_transform, n_out)

                # Propagate the original input in Residual and Highway nets.
                if (layer_id < len(arch)-1 and ntype != 'plain'):
                    if ntype == 'highway':
                        with tf.variable_scope('highway'):
                            gate = nn.fully_connected(out, n_out,
                                biases_initializer=tf.constant_initializer(.3),
                                activation_fn=tf.nn.sigmoid, name='gate')
                            out = out_transform * gate + out * (1 - gate)
                    elif ntype == 'residual':
                        with tf.variable_scope('residual'):
                            if layer_id < len(arch) - 1:
                                out_transform = tf.nn.elu(out_transform)
                            out_transform = nn.fully_connected(out_transform, n_out)
                            out += out_transform
                else:
                    out = out_transform
        return out

    def define_gan(self):
        with tf.variable_scope('generator'):
            x_from_z = tf.nn.tanh(self.define_nn(
                self.g_arch, self.g_ntype, self.z_tf))
        with tf.variable_scope('discriminator') as scope:
            y_from_x = self.define_nn(
                self.d_arch, self.d_ntype, self.x_tf)
            scope.reuse_variables()
            y_from_z = self.define_nn(
                self.d_arch, self.d_ntype, x_from_z)
        return y_from_x, y_from_z, x_from_z
    
    def define_gloss(self):
        # return -tf.reduce_mean(self.y_from_z)
        return .5 * tf.reduce_mean(tf.pow(self.y_from_z - 1, 2))

    def define_gtrain(self):
        all_vars = tf.trainable_variables()
        var_list = [v for v in all_vars if v.name.startswith('generator/')]
        return tf.train.AdamOptimizer(self.lr_tf).minimize(
            self.g_loss_tf, var_list=var_list)

    def define_dloss(self, flip=False):
        # return -tf.reduce_mean(self.y_from_x - self.y_from_z)
        return .5 * (tf.reduce_mean(tf.pow(self.y_from_x - 1, 2)
            + tf.pow(self.y_from_z, 2)))

    def define_dtrain(self):
        all_vars = tf.trainable_variables()
        var_list = [v for v in all_vars if v.name.startswith('discriminator/')]
        return tf.train.AdamOptimizer(self.lr_tf).minimize(
            self.d_loss_tf,  var_list=var_list)
    
    def fit(self, x, sess, epochs=1000,
            batch_size=32, lr=1e-3, n_diters=100, nn_verbose=True, **kwargs):
        start_time = time.time()
        batch_size = min(batch_size, x.shape[0])
        x = self.scaler_x.fit_transform(x)
        for epoch in range(epochs):

            # Train the discriminator.
            for k in range(n_diters):
                z_noise = self.sample_noise(batch_size)
                x_real = x[np.random.choice(x.shape[0], batch_size)]
                _, dloss = sess.run([self.d_train_tf, self.d_loss_tf],
                    {self.x_tf: x_real,
                     self.z_tf: z_noise,
                     self.keep_prob: 1.,
                     self.lr_tf: lr})

            # Train the generator.
            z_noise = self.sample_noise(batch_size)
            _, gloss = sess.run([self.g_train_tf, self.g_loss_tf],
                {self.z_tf: z_noise,
                 self.lr_tf: lr,
                 self.keep_prob: 1.})
            
            # Bookkeeping.
            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. '
                    'Discriminator loss {:.4g}. '
                    'Generator loss {:.4g}.').format(
                        epoch, int(tr_time), dloss, gloss))
                sys.stdout.flush()

    def sample(self, n_samples):
        z_noise = self.sample_noise(n_samples)
        x = sess.run(self.x_from_z,
            {self.z_tf: z_noise,
                self.keep_prob: 1.})
        return self.scaler_x.inverse_transform(x)

    def sample_noise(self, n_samples):
        return np.random.randn(n_samples, self.noise_dim)

if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    n_samples = 100000
    kwargs = {
              'epochs': 1000,
              'g_arch': [128]*10,
              'g_ntype': 'plain',
              'd_arch': [128]*10,
              'd_ntype': 'plain',
              'n_diters': 100,
              'lr': 1e-4,
              'noise_dim': 1,
              'batch_size': 32,
             }
    X = np.vstack([np.random.randn(int(n_samples/2), 1)*.1+6,
                   np.random.randn(int(n_samples/2), 1)*.2+3])
    X = np.random.uniform(low=-5, high=-2, size=(n_samples, 1))
    X *= np.random.rand(n_samples, 1)
    X *= -4
    X += 10
    X2 = -X
    small = min(X.min(), X2.min())
    large = max(X.max(), X2.max())
    X3 = np.random.uniform(small, large, size=(n_samples, 1))
    X = np.vstack([X, X2, X3])[np.random.choice(n_samples*3, n_samples)]
    gan = GAN(x_dim=X.shape[1], **kwargs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/{}'.format('gan'))
        writer.add_graph(sess.graph)

        # Fit the net.
        gan.fit(X, sess=sess, nn_verbose=True, **kwargs)

        # Generate some samples.
        X_samples = gan.sample(n_samples)

    vmin = min(X.min(), X_samples.min())
    vmax = max(X.max(), X_samples.max())
    fig = plt.figure(figsize=(10, 10))
    plt.hist(X.flatten(), bins=np.linspace(vmin, vmax, 100),
        label='true', alpha=.5)
    plt.hist(X_samples.flatten(), bins=np.linspace(vmin, vmax, 100),
        label='samples', alpha=.5)
    plt.legend(loc=0)
    plt.show()

