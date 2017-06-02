""" Tensorflow autoencoders. """
import sys
import time
from typing import List
from tempfile import NamedTemporaryFile

from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
import numpy as np
import tensorflow as tf


def define_ae(
        x_tf: (tf.placeholder, 'Input data.'),
        arch: (List[int], 'Encoder architecture.')):
    """ Define the autoencoder. 

    Creates the Tensorflow graph and variable initializers
    that define the encoder and decoder.

    Returns:
        encode: Tensofrlow encoding of x_tf.
        decode: Tensorflow decoding of encoded inputs.
        enc_tf: Tensorflow placeholder for encoded inputs.
    """
    x_dim = x_tf.get_shape().as_list()[1]
    nonlin = tf.nn.tanh
    #nonlin_inv = lambda x: .5 * tf.log((1 + x) / (1 - x))
    nonlin_inv = nonlin
    enc_tf = tf.placeholder(tf.float32, [None, arch[-1]], name='encoded_input')
    encode = x_tf

    # Define the encoder and decoder.
    with tf.variable_scope('ae'):
        for layer_id in range(len(arch)):
            n_in = x_dim if layer_id == 0 else arch[layer_id-1]
            n_out = arch[layer_id]
            with tf.variable_scope('Layer{}'.format(layer_id)):
                with tf.variable_scope('Weights'):
                    W = tf.get_variable(
                        'W', [n_in, n_out], tf.float32,
                        tf.truncated_normal_initializer(
                            stddev=1./n_in, dtype=tf.float32))
                with tf.variable_scope('Biases'):
                    b = tf.get_variable(
                        'b_enc', [1, 1], tf.float32,
                        tf.constant_initializer(0.))
                encode = tf.add(tf.matmul(encode, W), b)
                encode = nonlin(encode)

        predict = encode
        for layer_id in range(len(arch)):
            layer_id = len(arch) - layer_id - 1
            with tf.variable_scope('Layer{}'.format(layer_id)):
                with tf.variable_scope('Weights', reuse=True):
                    W = tf.get_variable('W')
                with tf.variable_scope('Biases'):
                    b = tf.get_variable(
                        'b_dec', [1, 1], tf.float32,
                        tf.constant_initializer(0.))
                predict = tf.add(tf.matmul(predict, tf.transpose(W)), b)
                predict = nonlin_inv(predict)

        decode = enc_tf
        for layer_id in range(len(arch)):
            layer_id = len(arch) - layer_id - 1
            with tf.variable_scope('Layer{}'.format(layer_id)):
                with tf.variable_scope('Weights', reuse=True):
                    W = tf.get_variable('W')
                with tf.variable_scope('Biases', reuse=True):
                    b = tf.get_variable('b_dec')
                decode= tf.add(tf.matmul(decode, tf.transpose(W)), b)
                decode = nonlin_inv(decode)

    return encode, decode, predict, enc_tf


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
        self.encode_tf, self.decode_tf, self.predict_tf, self.enc_tf =\
                define_ae(self.x_tf, self.arch)

        # Define loss (MSE).
        self.loss_tf = tf.losses.mean_squared_error(
                self.x_tf, self.predict_tf)

        # Define the optimizer.
        self.train_op_tf = tf.train.AdamOptimizer(
                self.lr_tf).minimize(self.loss_tf)

        # Use the minmax scaler, as decoder uses tanh.
        self.scaler_x = MinMaxScaler(feature_range=(-1, 1))

        # Define a saver object.
        self.tmpfile = NamedTemporaryFile()
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create the Tensorflow session and its initializer op.
        self.sess = tf.Session()
        writer = tf.summary.FileWriter('logs', self.sess.graph)
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def close(self):
        """ Close the session and reset the graph. Note: this
        will make this neural net unusable. """
        self.sess.close()
        tf.reset_default_graph()

    def restart(self):
        """ Re-initialize the network. """
        self.sess.run(self.init_op)

    def fit(self, 
            x: (np.ndarray, '(n_samples, x_dim): The data.'),
            epochs: (int, 'Number of training epochs.') = 1000,
            lr: (float, 'Learning rate.') = 1e-3,
            batch_size: (int, 'Training minibatch size.') = 32,
            ae_verbose: (bool, 'Display training progress messages.') = True,
            **kwargs):
        
        # Split data into a training and validation set.
        n_samples = x.shape[0]
        n_val = int(n_samples * .1)
        ids_perm = np.random.permutation(n_samples)
        self.valid_ids = ids_perm[:n_val]
        x_val = x[self.valid_ids]
        x_tr = x[ids_perm[n_val:]]
        x_tr = self.scaler_x.fit_transform(x_tr)
        x_val = self.scaler_x.transform(x_val)

        # Train the neural net.
        tr_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        best_val = np.inf
        start_time = time.time()
        for epoch_id in range(epochs):
            batch_ids = np.random.choice(n_samples-n_val,
                    batch_size, replace=False)
            tr_loss = self.sess.run(
                self.loss_tf, {self.x_tf: x_tr[batch_ids]})
            self.sess.run(self.train_op_tf,
                          {self.x_tf: x_tr[batch_ids], self.lr_tf: lr})
            val_loss = self.sess.run(self.loss_tf, {self.x_tf: x_val})
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
                last_improved = epoch_id
                best_val = val_loss
                model_path = self.saver.save(
                    self.sess, self.tmpfile.name)

            tr_time = time.time() - start_time
            if ae_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()

        self.saver.restore(self.sess, model_path)
        if ae_verbose:
            print('Trainig done in {} epochs, {}s. Validation loss {:.4g}.'.format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses

    def encode(self,
            x: (np.ndarray, '(n_samples, x_dim): The data.')):
        """ Encode x. """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        encoded = self.sess.run(self.encode_tf, {self.x_tf: x})
        return encoded

    def decode(self,
            y: (np.ndarray, '(n_samples, y_dim): Encoded data.')):
        """ Decode y. Note that y_dim must equal self.arch[-1]. """
        if y.shape[1] != self.arch[-1]:
            raise ValueError('Encoded data dimension mismatch.')
        decoded = self.sess.run(self.decode_tf, {self.enc_tf: y})
        return self.scaler_x.inverse_transform(decoded)

if __name__=="__main__":
    """ Encode/decode MNIST digits. """
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X = mnist.train.images
    Xts = mnist.test.images
    ae = Autoencoder(x_dim=X.shape[1], arch=[1024])
    ae.fit(X, lr=1e-3, epochs=1000)
    Xts_pred = ae.decode(ae.encode(Xts))
    fig = plt.figure(figsize=(2, 10), facecolor='white')
    for im_id_id, im_id in enumerate(np.random.choice(Xts.shape[0], 10)):
        plt.subplot2grid((10, 2), (im_id_id, 0))
        plt.imshow(Xts[im_id].reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.subplot2grid((10, 2), (im_id_id, 1))
        plt.imshow(Xts_pred[im_id].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.savefig('res.png')

