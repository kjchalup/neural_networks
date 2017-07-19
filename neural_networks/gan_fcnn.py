""" Generative Adversarial Network implementation.

This is in fact the Least-Squares GAN, as I found it
yields best results so far. However, the GAN market
is developing rapidly.
"""
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from neural_networks.scalers import MinMaxScaler
import tensorflow as tf

from neural_networks import fcnn


class SkeletonFCNN(fcnn.FCNN):
    __doc__ = """ A fcnn skeleton.
    
    Used by the GAN class to create fcnn graph shared between
    the discriminator and generator training ops.

    Only the fcnn feedforward graph is used: loss, training, scalers,
    prediction and fit methods are implemented by the GAN class.
    """
    __doc__ += fcnn.FCNN.__doc__

    def __init__(self, x_tf, x_shape, reuse=False, **kwargs):
       super(SkeletonFCNN, self).__init__(x_tf=x_tf, x_shape=x_shape,
            reuse=reuse, **kwargs)

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


class GAN_FCNN(object):
    """ A Least-Squares GAN using an FCNN to generate images.

    Args:
        x_shape (int, int, int): Input dimensionality (h, w, c).
    """
    def __init__(self, x_shape, **kwargs):
        # Bookkeeping.
        self.x_shape = x_shape
        self.x_tf = tf.placeholder(tf.float32,
            [None, x_shape[0], x_shape[1], x_shape[2]], name='x_input')
        self.z_tf = tf.placeholder(tf.float32,
            [None, x_shape[0], x_shape[1], x_shape[2]], name='z_input')
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')
        
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
            gen_net = SkeletonFCNN(
                x_shape=self.x_shape, x_tf=self.z_tf, n_layers=4,
                n_filters=np.array([32] * 4))
            x_from_z = tf.nn.tanh(gen_net.y_pred)

        with tf.variable_scope('discriminator') as scope:
            # Transform the (h x w x c) feature map to a scalar by avg pooling.
            disc_net = SkeletonFCNN(x_shape=self.x_shape, x_tf=self.x_tf,
                n_layers=6, n_filters=np.array([32] * 6))
            y_from_x = tf.reduce_mean(disc_net.y_pred, axis=[1, 2, 3])
            y_from_x = tf.reshape(y_from_x, [-1, 1])
            scope.reuse_variables()

            y_from_z = SkeletonFCNN(x_shape=self.x_shape, x_tf=x_from_z,
                n_layers=6, n_filters=np.array([32] * 6), reuse=True).y_pred
            y_from_z = tf.reduce_mean(y_from_z, axis=[1, 2, 3])
            y_from_z = tf.reshape(y_from_z, [-1, 1])

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
            self.d_loss_tf,  var_list=var_list)

    def fit(self, x, sess, epochs=1000, batch_size=32, lr=1e-3,
            n_diters=100, nn_verbose=True, **kwargs):
        start_time = time.time()
        batch_size = min(batch_size, x.shape[0])
        x = self.scaler_x.fit_transform(x)
        for epoch in range(epochs):
            # Train the discriminator.
            for k in range(n_diters):
                z_noise = self.sample_noise(batch_size)
                x_real = x[np.random.choice(x.shape[0], batch_size)]
                feed_dict = {self.x_tf: x_real,
                             self.z_tf: z_noise,
                             self.lr_tf: lr}
                _, dloss = sess.run([self.d_train_tf, self.d_loss_tf],
                    feed_dict)

            # Train the generator.
            z_noise = self.sample_noise(batch_size)
            feed_dict = {self.z_tf: z_noise,
                         self.lr_tf: lr}
            _, gloss = sess.run([self.g_train_tf, self.g_loss_tf], feed_dict)
            
            # Bookkeeping.
            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. '
                    'Discriminator loss {:.4g}. '
                    'Generator loss {:.4g}.').format(
                        epoch, int(tr_time), dloss, gloss))
                sys.stdout.flush()

    def sample(self, n_samples):
        """ Sample from the distribution defined by the generator.

        Args:
            n_samples (int): Number of samples to create.

        Returns:
            samples (n_samples, x_dim): Samples defined by the
                generator's distribution.
        """
        z_noise = self.sample_noise(n_samples)
        feed_dict = {self.z_tf: z_noise}
        x = sess.run(self.x_from_z, feed_dict)
        return self.scaler_x.inverse_transform(x)

    def sample_noise(self, n_samples):
        """ Sample inputs to the generator.

        Args:
            n_samples (int): Number of noise-samples.

        Returns:
            noise (n_samples, self.noise_dim): Noise samples.
        """
        return np.random.randn(n_samples, *self.x_shape)


if __name__=="__main__":
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    lbls = mnist.train.labels
    ims_tr = mnist.train.images[np.where(lbls[:, 7].flatten())[0]].reshape(-1, 28, 28, 1)

    kwargs = {
              'epochs': 1000,
              'lr': 1e-4,
              'batch_size': 32
             }
    gan = GAN_FCNN(x_shape=ims_tr.shape[1:])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/{}'.format('gan'))
        writer.add_graph(sess.graph)

        # Fit the net.
        gan.fit(ims_tr, sess=sess, nn_verbose=True, **kwargs)

        # Generate some samples.
        X_samples = gan.sample(100)

    # Show the results.
    plt.figure(figsize=(16, 4))
    for im_id_id, im_id in enumerate(np.random.choice(X_samples.shape[0], 8)):
        plt.subplot2grid((2, 8), (0, im_id_id))
        plt.title('Generated')
        plt.axis('off')
        plt.imshow(X_samples[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')

        plt.subplot2grid((2, 8), (1, im_id_id))
        plt.title('Nearest-Neighbor')
        plt.axis('off')
        plt.imshow(X_samples[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')
    plt.savefig('gan_fcnn.png')
    plt.show()
