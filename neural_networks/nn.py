""" Neural network routines. """
import sys
import time
from typing import List
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf


def define_nn(
        x_tf: tf.placeholder,
        arch: List[int],
        highway: bool = False,
        keep_prob: tf.placeholder = 1.):
    """ Define a Neural Network.

    The architecture of the network is deifned by the Ws, list of weight
    matrices. The last matrix must be of shape (?, y_dim). If the number
    of layers is lower than 3, use the sigmoid nonlinearity.
    Otherwise, use the relu.

    Args:
        x_tf: Input data.
        arch: Architecture, including output layer. 
        highway: If True, add highway net gating.
        keep_prob: Dropout probability of keeping a unit on.

    Returns:
        out: Predicted y.

    Raises:
        ValueError: When the last weight tensor's output is not compatible
            with the input shape.
    """
    x_dim = x_tf.get_shape().as_list()[1]
    y_pred = x_tf
    if highway:
        for layer_id in range(len(arch)-1):
            if arch[layer_id] != arch[0]:
                raise ValueError('Highway nets require all' 
                    'layers to be the same size.')

    with tf.variable_scope('nn'):
        for layer_id in range(len(arch)):
            n_in = x_dim if layer_id == 0 else arch[layer_id-1]
            n_out = arch[layer_id]
            with tf.variable_scope('layer{}'.format(layer_id)):
                W = tf.get_variable('W', [n_in, n_out], tf.float32,
                    tf.truncated_normal_initializer(
                        stddev=1. / n_in, dtype=tf.float32))
                b = tf.get_variable('b', [1, 1], tf.float32,
                    tf.constant_initializer(0.))
                if highway and 0 < layer_id < len(arch) - 1:
                    W_gate = tf.get_variable('W_gate', [n_in, n_out], tf.float32,
                        tf.truncated_normal_initializer(
                            stddev=1. / n_in, dtype=tf.float32))
                    b_gate = tf.get_variable('b_gate', [1, 1], tf.float32,
                        tf.constant_initializer(-3.))
                    transform = tf.add(tf.matmul(y_pred, W_gate), b_gate)
                    transform = tf.nn.sigmoid(transform)
                    y_transform = tf.add(tf.matmul(y_pred, W), b)
                    y_carry = y_pred
                    y_pred = y_transform * transform + y_carry * (1 - transform)
                else:
                    y_pred = tf.add(tf.matmul(y_pred, W), b)
                if layer_id < len(arch) - 1:
                    y_pred = tf.nn.relu(y_pred)
    return y_pred


class NN(object):
    """ A Neural Net object.

    The interface mimics that of sklearn: first,
    initialize the NN. Fit it with NN.fit(),
    and predict with NN.predict().

    Args:
        x_dim: Input data dimensionality.
        y_dim: Output data dimensionality.
        arch: A list of integers, each corresponding to the number
            of units in the NN's hidden layer.
        highway: If True, add highway-network connections.
    """
    def __init__(self, x_dim, y_dim, arch=[1024], highway=False, **kwargs):
        # Bookkeeping.
        self.arch = arch
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='inputs')
        self.y_tf = tf.placeholder(
            tf.float32, [None, y_dim], name='outputs')
        self.lr_tf = tf.placeholder(tf.float32, name='learningrate')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.y_dim = y_dim

        # Inference.
        self.y_pred = define_nn(self.x_tf, arch + [y_dim], highway, self.keep_prob)

        # Loss.
        self.loss_tf = tf.losses.mean_squared_error(self.y_tf, self.y_pred)

        # Training.
        self.train_op_tf = tf.train.AdamOptimizer(
            self.lr_tf).minimize(self.loss_tf)

        # Define the data scaler.
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Define the saver object for model persistence.
        self.tmpfile = NamedTemporaryFile()
        self.saver = tf.train.Saver(max_to_keep=1)

        # Define the Tensorflow session, and its initializer op.
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def close(self):
        """ Close the session and reset the graph. Note: this
        will make this neural net unusable. """
        self.sess.close()

    def restart(self):
        """ Re-initialize the network. """
        self.sess.run(self.init_op)

    def predict(self, x):
        """ Compute the output for given data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        y_pred = self.sess.run(self.y_pred, {self.x_tf: x, self.keep_prob: 1.})
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, epochs=1000, batch_size=32,
            lr=1e-3, nn_verbose=False,
            dropout_keep_prob=1., **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            epochs (int): How many batches to train for.
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            nn_verbose (bool): Display training progress messages (or not).
            dropout_keep_prob (float): Probability of keeping a unit on.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).
        """
        # Split data into a training and validation set.
        n_samples = x.shape[0]
        n_val = int(n_samples * .1)
        ids_perm = np.random.permutation(n_samples)
        self.valid_ids = ids_perm[:n_val]
        x_val = x[self.valid_ids]
        y_val = y[self.valid_ids]
        x_tr = x[ids_perm[n_val:]]
        y_tr = y[ids_perm[n_val:]]
        x_tr = self.scaler_x.fit_transform(x_tr)
        y_tr = self.scaler_y.fit_transform(y_tr)
        x_val = self.scaler_x.transform(x_val)
        y_val = self.scaler_y.transform(y_val)

        # Train the neural net.
        tr_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        best_val = np.inf
        start_time = time.time()
        for epoch_id in range(epochs):
            batch_ids = np.random.choice(n_samples-n_val,
                    batch_size, replace=False)
            tr_loss = self.sess.run(
                self.loss_tf, {self.x_tf: x_tr[batch_ids],
                               self.y_tf: y_tr[batch_ids],
                               self.keep_prob: 1.})
            self.sess.run(self.train_op_tf,
                          {self.x_tf: x_tr[batch_ids],
                           self.y_tf: y_tr[batch_ids],
                           self.keep_prob: dropout_keep_prob,
                           self.lr_tf: lr})
            val_loss = self.sess.run(self.loss_tf,
                                     {self.x_tf: x_val,
                                      self.y_tf: y_val,
                                      self.keep_prob: 1.})
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
                last_improved = epoch_id
                best_val = val_loss
                model_path = self.saver.save(
                    self.sess, self.tmpfile.name)

            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()

        self.saver.restore(self.sess, model_path)
        if nn_verbose:
            print('Trainig done in {} epochs, {}s. Validation loss {:.4g}.'.format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses

if __name__ == "__main__":
    """ Check that everything works as expected. Should take several 
    seconds on a CPU machine. """
    import numpy as np
    from matplotlib import pyplot as plt
    x = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y = np.sin(5 * x ** 2) + x + np.random.randn(*x.shape) * .1
    nn = NN(x_dim=x.shape[1], y_dim=y.shape[1], arch=[1024] * 4, highway=False)
    nn.fit(x, y, nn_verbose=True, lr=1e-3)
    y_pred = nn.predict(x)
    plt.plot(x, y, x, y_pred)
    plt.savefig('res.png')
    plt.show()
