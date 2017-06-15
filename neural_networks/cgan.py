""" Conditional Generative Adversarial Network.

This is in fact the Least-Squares CGAN, as I found it
yields best results so far. However, the GAN market
is developing rapidly.
"""
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from neural_networks import nn


class SkeletonNN(nn.NN):
    __doc__ = """ A neural net skeleton.
    
    Used by the GAN class to create neural net graphs. """
    __doc__ += nn.NN.__doc__

    def __init__(self, x_tf, dropout_tf, **kwargs):
       super().__init__(x_tf=x_tf, dropout_tf=dropout_tf,**kwargs)

    def define_loss(self):
        return None

    def define_training(self):
        return None

    def define_scalers(self):
        return None, None

    def predict(self):
        pass

    def fit(self):
        pass


class CGAN(object):
    """ A Least-Squares CGAN.

    Args:
        x_dim (int): Data dimensionality.
        cond_dim (int): Conditioning data dimensionality.
        noise_dim (int): Generator random input dimensionality.
        g_arch ([int, int]): Generator architecture.
        g_ntype (str): Generator net type (see neural_networks.nn).
        d_arch ([int, int]): Discriminator architecture.
        d_ntype (str): Discriminator net type (see neural_networks.nn).
    """
    def __init__(self, x_dim, cond_dim, noise_dim, g_arch=[128, 128],
        g_ntype='plain', d_arch=[128, 128], d_ntype='plain', **kwargs):
        # Bookkeeping.
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='x_input')
        self.cond_tf = tf.placeholder(
            tf.float32, [None, cond_dim], name='cond_input')
        self.z_tf = tf.placeholder(
            tf.float32, [None, noise_dim], name='z_input')
        self.lr_tf = tf.placeholder(
            tf.float32, name='learning_rate')
        self.dropout_tf = tf.placeholder(
            tf.float32, name='dropout')
        self.g_arch = g_arch + [x_dim]
        self.g_ntype = g_ntype
        self.d_arch = d_arch + [1]
        self.d_ntype = d_ntype
        
        # Define the Generator, Discriminator, and their losses.
        self.y_from_x, self.y_from_z, self.x_from_z =\
            self.define_gan()
        with tf.variable_scope('g_loss'):
            self.g_loss_tf = self.define_gloss()
            self.g_train_tf = self.define_gtrain()
        with tf.variable_scope('d_loss'):
            self.d_loss_tf = self.define_dloss()
            self.d_train_tf = self.define_dtrain()

        # Define the data scalers.
        self.scaler_x = self.define_scalers()

    def define_scalers(self):
        """ Use the MinMax scaler, as generator will have tanh outputs. """
        return MinMaxScaler(feature_range=(-1, 1))

    def define_gan(self):
        with tf.variable_scope('generator'):
            gen_net = SkeletonNN(x_dim=self.cond_dim + self.noise_dim,
                y_dim=self.x_dim,  arch=self.g_arch, ntype=self.g_ntype,
                x_tf=tf.concat([self.cond_tf, self.z_tf], axis=1),
                dropout_tf=self.dropout_tf)
            x_from_z = tf.nn.tanh(gen_net.y_pred)
        with tf.variable_scope('discriminator') as scope:
            disc_net = SkeletonNN(x_dim=self.cond_dim + self.x_dim,
                y_dim=1,  arch=self.d_arch, ntype=self.d_ntype,
                x_tf=tf.concat([self.cond_tf, self.x_tf], axis=1),
                dropout_tf=self.dropout_tf)
            y_from_x = disc_net.y_pred
            scope.reuse_variables()
            y_from_z = SkeletonNN(x_dim=self.cond_dim + self.x_dim,
                y_dim=1, arch=self.d_arch, ntype=self.d_ntype,
                x_tf=tf.concat([self.cond_tf, x_from_z], axis=1),
                dropout_tf=self.dropout_tf).y_pred
        return y_from_x, y_from_z, x_from_z

    def define_gloss(self):
        return .5 * tf.reduce_mean(tf.pow(self.y_from_z - 1, 2))

    def define_gtrain(self):
        all_vars = tf.trainable_variables()
        var_list = [v for v in all_vars if v.name.startswith('generator/')]
        return tf.train.AdamOptimizer(self.lr_tf).minimize(
            self.g_loss_tf, var_list=var_list)

    def define_dloss(self, flip=False):
        return .5 * (tf.reduce_mean(tf.pow(self.y_from_x - 1, 2)
                                    + tf.pow(self.y_from_z, 2)))

    def define_dtrain(self):
        all_vars = tf.trainable_variables()
        var_list = [v for v in all_vars if v.name.startswith('discriminator/')]
        return tf.train.AdamOptimizer(self.lr_tf).minimize(
            self.d_loss_tf, var_list=var_list)

    def fit(self, x, cond, sess, epochs=1000, batch_size=32, lr=1e-3,
            n_diters=100, nn_verbose=True, **kwargs):
        start_time = time.time()
        batch_size = min(batch_size, x.shape[0])
        x = self.scaler_x.fit_transform(x)
        for epoch in range(epochs):
            # Train the discriminator.
            for k in range(n_diters):
                x_ids = np.random.choice(x.shape[0], batch_size)
                feed_dict = {self.x_tf: x[x_ids],
                             self.cond_tf: cond[x_ids],
                             self.z_tf: self.sample_noise(batch_size),
                             self.dropout_tf: 1.,
                             self.lr_tf: lr}
                _, dloss = sess.run([self.d_train_tf, self.d_loss_tf],
                                    feed_dict)

            # Train the generator.
            z_noise = self.sample_noise(batch_size)
            x_ids = np.random.choice(x.shape[0], batch_size)
            feed_dict = {self.z_tf: self.sample_noise(batch_size),
                         self.cond_tf: cond[x_ids],
                         self.lr_tf: lr,
                         self.dropout_tf: 1.}
            _, gloss = sess.run([self.g_train_tf, self.g_loss_tf], feed_dict)

            # Bookkeeping.
            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. '
                                  'Discriminator loss {:.4g}. '
                                  'Generator loss {:.4g}.').format(
                                  epoch, int(tr_time), dloss, gloss))
                sys.stdout.flush()

    def sample(self, n_samples, cond):
        """ Sample from the distribution defined by the generator.

        Args:
            n_samples (int): Number of samples to create *per each cond data*.
            cond (n_data, cond_dim): Conditioning data.

        Returns:
            samples (n_data, n_samples, x_dim): Samples defined by the
                generator's distribution.
        """
        res = np.zeros((cond.shape[0], n_samples, self.x_dim))
        for cond_id, cond_data in enumerate(cond):
            z_noise = self.sample_noise(n_samples)
            feed_dict = {self.z_tf: z_noise,
                         self.cond_tf: np.array([cond_data] * n_samples),
                         self.dropout_tf: 1.}
            x = sess.run(self.x_from_z, feed_dict)
            res[cond_id] = self.scaler_x.inverse_transform(x)
        return res

    def sample_noise(self, n_samples):
        """ Sample inputs to the generator.

        Args:
            n_samples (int): Number of noise-samples.

        Returns:
            noise (n_samples, self.noise_dim): Noise samples.
        """
        return np.random.randn(n_samples, self.noise_dim)


if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    n_samples = 10000
    kwargs = {
              'epochs': 10000,
              'g_arch': [32]*10,
              'g_ntype': 'plain',
              'd_arch': [32]*10,
              'd_ntype': 'plain',
              'n_diters': 10,
              'lr': 1e-4,
              'noise_dim': 10,
              'batch_size': 32,
             }
    # Make test data: a noisy mix of functions sampled with different probs.
    cond = np.linspace(-1, 1, n_samples ).reshape(-1, 1)
    x1 = cond**2 + np.random.randn(*cond.shape) * .3
    x2 = -cond/2-1 + np.random.randn(*cond.shape) * .1
    x3 = -cond-2 + np.random.randn(*cond.shape) * .1
    x4 = cond-3 + np.random.randn(*cond.shape) * .1
    x = np.hstack([x1, x2, x3, x4])
    ids = np.random.choice(4, x.shape[0], p=[.7, .1, .1, .1])
    x = cond + x[np.arange(x.shape[0]), ids].reshape(-1, 1)

    cgan = CGAN(x_dim=x.shape[1], cond_dim=cond.shape[1], **kwargs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/{}'.format('gan'))
        writer.add_graph(sess.graph)

        # Fit the net.
        cgan.fit(x=x, cond=cond, sess=sess, nn_verbose=True, **kwargs)

        # Generate some samples.
        samples = cgan.sample(n_samples=1, cond=cond)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('Samples real data.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([x.min(), x.max()])
    plt.plot(cond, x, 'k.', alpha=.2)

    plt.subplot(1, 2, 2)
    plt.title('CGAN samples.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([x.min(), x.max()])
    plt.plot(cond, np.squeeze(samples), 'k.')

    plt.show()
