""" Neural network routines. """
import sys
import time
from typing import List
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.contrib.layers import summaries 


def summarize_var(variable):
    name = variable.name
    tf.summary.histogram(name, variable)
    mean = tf.reduce_mean(variable)
    tf.summary.scalar('{}_mean'.format(name), mean)
    std = tf.reduce_mean((variable - mean)**2)
    tf.summary.scalar('{}_std'.format(name), std)
    tf.summary.scalar('{}_min'.format(name), tf.reduce_min(variable))
    tf.summary.scalar('{}_max'.format(name), tf.reduce_max(variable))


def fully_connected(invar, n_out, activation_fn=None,
    weights_initializer=tf.contrib.layers.xavier_initializer,
    biases_initializer=tf.constant_initializer, name='fc'):
    """ A fully-connected layer. Use this instead of tf.contrib.layers
    to retain fine control over summaries. """
    with tf.variable_scope(name):
        n_in = invar.get_shape().as_list()[1]
        W = tf.get_variable('W', [n_in, n_out], tf.float32,
            weights_initializer())
        summarize_var(W)
        b = tf.get_variable('b', [1, 1], tf.float32,
            biases_initializer(0.1))
        tf.summary.scalar('b', tf.reduce_mean(b))
        out = tf.add(tf.matmul(invar, W), b, name='preactivation')
        summarize_var(out)
        if activation_fn is not None:
            out = activation_fn(out, name='postactivation')
            summarize_var(out)
    return out


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
        ntype: 'plain', 'highway', or 'residual'.
        name: Name prepended to all variables in the graph.
    """
    def __init__(self, x_dim, y_dim, arch=[1024],
            ntype='plain', name='nn', **kwargs):
        # Bookkeeping.
        self.name = name
        self.arch = arch + [y_dim]
        self.ntype = ntype
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='inputs')
        self.y_tf = tf.placeholder(
            tf.float32, [None, y_dim], name='outputs')
        self.lr_tf = tf.placeholder(tf.float32, name='learningrate')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.y_dim = y_dim
        self.is_training_tf = tf.placeholder(tf.bool, name='trainflag')

        # Inference.
        self.y_pred = self.define_nn()

        # Loss.
        with tf.name_scope('loss'):
            self.loss_tf = self.define_loss()
        tf.summary.scalar('loss', self.loss_tf)

        # Training.
        self.train_op_tf = self.define_training()

        # Define the saver object for model persistence.
        self.tmpfile = NamedTemporaryFile()
        self.saver = tf.train.Saver(max_to_keep=1)

        # Define the data scaler.
        self.scaler_x, self.scaler_y = self.define_scalers()


    def define_nn(self):
        """ Define a Neural Network.

        The architecture of the network is deifned by the Ws, list of weight
        matrices. The last matrix must be of shape (?, y_dim). If the number
        of layers is lower than 3, use the sigmoid nonlinearity.
        Otherwise, use the relu.

        Returns:
            out: Predicted y.

        Raises:
            ValueError: When the last weight tensor's output is not compatible
                with the input shape.
        """
        x_dim = self.x_tf.get_shape().as_list()[1]
        y_pred = self.x_tf

        if self.ntype != 'plain':
            # Check that (for Highway and Residual nets) layers have equal size.
            for layer_id in range(len(self.arch)-1):
                if self.arch[layer_id] != self.arch[0]:
                    raise ValueError('Highway and Residual nets require all' 
                        'layers to be the same size.')
            # Make sure the input has the same size too.
            with tf.variable_scope('layerreshape'):
                y_pred = fully_connected(y_pred, self.arch[0])
                y_pred = tf.nn.dropout(y_pred, keep_prob=self.keep_prob)

        for layer_id in range(len(self.arch)):
            n_out = self.arch[layer_id]
            with tf.variable_scope('layer{}'.format(layer_id)):
                y_transform = y_pred

                # Transform the input.
                with tf.variable_scope('transform'):
                    if layer_id < len(self.arch) - 1:
                        y_transform = tf.nn.elu(y_transform)
                        y_transform = tf.nn.dropout(
                                y_transform, keep_prob=self.keep_prob)
                    y_transform = fully_connected(y_transform, n_out)

                # Propagate the original input in Residual and Highway nets.
                if (layer_id < len(self.arch)-1 and self.ntype != 'plain'):
                    if self.ntype == 'highway':
                        with tf.variable_scope('highway'):
                            gate = fully_connected(y_pred, n_out,
                                activation_fn=tf.nn.sigmoid, name='gate')
                            y_pred = y_transform * gate + y_pred * (1 - gate)
                    elif self.ntype == 'residual':
                        with tf.variable_scope('residual'):
                            if layer_id < len(self.arch) - 1:
                                y_transform = tf.nn.elu(y_transform)
                            y_transform = fully_connected(y_transform, n_out)
                            y_pred += y_transform
                else:
                    y_pred = y_transform

        return y_pred

    def define_loss(self):
        return tf.losses.mean_squared_error(self.y_tf, self.y_pred)

    def define_training(self):
        return tf.train.AdamOptimizer(
                    self.lr_tf).minimize(self.loss_tf)

    def define_scalers(self):
        return StandardScaler(), StandardScaler()

    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        y_pred = sess.run(self.y_pred,
                {self.x_tf: x,
                 self.keep_prob: 1.,
                 self.is_training_tf: False})
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, sess, epochs=1000, batch_size=32,
            lr=1e-3, nn_verbose=False,
            dropout_keep_prob=1., writer=None, summary=None, **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            sess: Tensorflow session.
            epochs (int): How many batches to train for.
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            nn_verbose (bool): Display training progress messages (or not).
            dropout_keep_prob (float): Probability of keeping a unit on.
            writer: A writer object for Tensorboard bookkeeping.
            summary: Summary object for the writer.

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
        batch_size = min(batch_size, n_samples-n_val)

        for epoch_id in range(epochs):
            batch_ids = np.random.choice(n_samples-n_val,
                batch_size, replace=False)
            if summary:
                tr_loss, s = sess.run(
                    [self.loss_tf, summary],
                    {self.x_tf: x_tr[batch_ids],
                     self.y_tf: y_tr[batch_ids],
                     self.is_training_tf: False,
                     self.keep_prob: 1.})
            else:
                tr_loss = sess.run(
                    self.loss_tf,
                    {self.x_tf: x_tr[batch_ids],
                     self.y_tf: y_tr[batch_ids],
                     self.is_training_tf: False,
                     self.keep_prob: 1.})
            sess.run(self.train_op_tf,
                          {self.x_tf: x_tr[batch_ids],
                           self.y_tf: y_tr[batch_ids],
                           self.is_training_tf: True,
                           self.keep_prob: dropout_keep_prob,
                           self.lr_tf: lr})
            val_loss = sess.run(self.loss_tf,
                                     {self.x_tf: x_val,
                                      self.y_tf: y_val,
                                      self.is_training_tf: False,
                                      self.keep_prob: 1.})

            if writer:
                # Add summaries for Tensorboard.
                writer.add_summary(s, epoch_id)

            # Store losses.
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
                last_improved = epoch_id
                best_val = val_loss
                model_path = self.saver.save(
                    sess, self.tmpfile.name)

            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()

        self.saver.restore(sess, model_path)
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
    nn = NN(x_dim=x.shape[1], y_dim=y.shape[1], arch=[128]*10, ntype='highway')
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/{}'.format('nn'))
        writer.add_graph(sess.graph)

        # Fit the net.
        nn.fit(x, y, sess=sess, nn_verbose=True, lr=1e-2)

        # Predict.
        y_pred = nn.predict(x, sess=sess)
    plt.plot(x, y, x, y_pred)
    plt.savefig('res.png')
