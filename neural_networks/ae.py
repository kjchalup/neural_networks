""" Tensorflow autoencoders. """
from typing import List
import tensorflow as tf


def define_ae(
        x_tf: (tf.placeholder, 'Input data.'),
        arch: (List[int], 'Encoder architecture.')):
    """ Define the autoencoder. 

    Creates the Tensorflow graph and variable initializers
    that define the encoder and decoder.
    """


class Autoencoder(object):
    """ A basic autoencoder.

    The interface mimics that of sklearn: first,
    initialize the autoencoder. Fit with the `fit` 
    method. Encode and decode with `encode` and `decode`
    methods.

    Args:
        x_dim: Input data dimensionality.
        arch: Architecture of the encoder. The decoding
            is symmetric.
    """
    def __init__(self,
            x_dim: int,
            arch: List[int] = [32]):
        self.arch = arch
        self.x_tf = tf.placeholder(tf.float32, [None, x_dim], name='input')
        self.lr_tf = tf.placeholder(tf.float32, name='lr')

        # Create the graph.
        self.x_pred_tf = define_ae(self.x_tf, self.arch)

        # Define loss (MSE).
        self.loss_tf = tf.losses.mean_squared_error(self.x_tf, self.x_pred_tf)

        # Define the optimizer.
        self.train_op_tf = tf.train.AdamOptimizer(
                self.lr_tf).minimize(self.loss_tf)

        # Initialize data rescalers.
        self.scaler_x = StandardScaler()

        # Define a saver object.
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create the Tensorflow session and its initializer op.
        self.sess = tf.Session()
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def fit(self, 
            x: (np.ndarray, '(n_samples, x_dim): The data.'),
            epochs: (int, 'Number of training epochs.'),
            lr: (float, 'Learning rate.'),
            ae_verbose: (bool, 'Display training progress messages.')
            **kwargs):
        """ Train the autoencoder. """
        pass

    def encode(self,
            x: (np.ndarray, '(n_samples, x_dim): The data.')):
        """ Encode x. """
        pass

    def decode(self,
            y (np.ndarray, '(n_samples, y_dim): Encoded data.')):
        """ Decode y. Note that y_dim must equal self.arch[-1]. """
        if y.shape[1] != self.arch[-1]:
            raise ValueError('Encoded data dimension mismatch.')
        pass

