""" Neural network routines. """
import sys
import time
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf

from neural_networks import scalers


def compute_nograd(nans, n_out):
    """ Compute a binary mask matrix of shape (x.shape[1], n_out). The matrix
    has '0' in each row that corresponds to x equal nan. """
    W = np.ones((nans.shape[1], n_out))
    W[nans.sum(axis=0) > 0] = 0.
    return W


def fully_connected(
    invar, n_out, activation_fn=tf.identity, W_nograd=None,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    biases_initializer=tf.constant_initializer(.1), name='fc'):
    """ A fully-connected layer. Use this instead of tf.contrib.layers
    to retain fine control over summaries. """
    with tf.variable_scope(name):
        n_in = invar.get_shape().as_list()[1]
        W = tf.get_variable('W', [n_in, n_out], tf.float32,
                            weights_initializer)
        tf.summary.histogram('W', W)
        b = tf.get_variable('b', [1, 1], tf.float32,
                            biases_initializer)
        tf.summary.scalar('b', tf.reduce_mean(b))
        if W_nograd is not None:
            W = tf.multiply(W, W_nograd)
        out = tf.add(tf.matmul(invar, W), b, name='preactivation')
        out = activation_fn(out, name='postactivation')
        tf.summary.histogram('postactivation', out)
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
        zeronans: If True, nans in inputs/outputs will be simply
            set to zero (after data normalization). If False, they
            will be ignored using a W_nograd matrix -- see fit().
    """
    def __init__(self, x_dim, y_dim, arch=[1024],
        ntype='plain', zeronans=True, **kwargs):
        # Bookkeeping.
        self.y_dim = y_dim
        self.arch = arch + [y_dim]
        self.ntype = ntype
        self.zeronans = zeronans
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='inputs')
        self.y_tf = tf.placeholder(
            tf.float32, [None, y_dim], name='outputs')
        self.lr_tf = tf.placeholder(tf.float32, name='learningrate')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        if not self.zeronans:
            self.W_nograd_tf = tf.placeholder(
                tf.float32, [x_dim, arch[0]], name='W_nograd')

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
        self.saver = tf.train.Saver(max_to_keep=2)

        # Define the data scaler.
        self.scaler_x, self.scaler_y = self.define_scalers()

    def define_nn(self):
        """ Define a Neural Network.

        The architecture of the network is deifned by the Ws, list of weight
        matrices. The last matrix must be of shape (?, y_dim). If the number
        of layers is lower than 3, use the sigmoid nonlinearity.
        Otherwise, use the relu.
        """
        y_pred = self.x_tf
        W_nograd = None if self.zeronans else self.W_nograd_tf

        if self.ntype not in ['residual', 'highway', 'plain']:
            raise ValueError('Network type not available.')

        # Check that (for Highway and Residual nets) the layers have equal size.
        if self.ntype != 'plain':
            for layer_id in range(len(self.arch)-1):
                if self.arch[layer_id] != self.arch[0]:
                    raise ValueError('Highway and Residual nets require all' 
                        'layers to be the same size.')

            # Reshape the input accordingly.
            with tf.variable_scope('layerreshape'):
                y_pred = fully_connected(
                    y_pred, self.arch[0], W_nograd=W_nograd)
                y_pred = tf.nn.dropout(y_pred, keep_prob=self.keep_prob)

        for layer_id in range(len(self.arch)):
            n_out = self.arch[layer_id]
            with tf.variable_scope('layer{}'.format(layer_id)):
                y_transform = y_pred

                # Transform the input.
                with tf.variable_scope('transform'):
                    if 0 < layer_id < len(self.arch) - 1:
                        y_transform = tf.nn.elu(y_transform)
                        y_transform = tf.nn.dropout(
                            y_transform, keep_prob=self.keep_prob)
                    if layer_id == 0 and self.ntype == 'plain':
                        y_transform = fully_connected(
                            y_transform, n_out, W_nograd=W_nograd)
                    else:
                        y_transform = fully_connected(y_transform, n_out)

                # Propagate the original input in Residual and Highway nets.
                if layer_id < len(self.arch)-1 and self.ntype != 'plain':

                    if self.ntype == 'highway':
                        with tf.variable_scope('highway'):
                            gate = fully_connected(
                                y_pred, n_out,
                                activation_fn=tf.nn.sigmoid,
                                name='gate')
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
        return scalers.StandardScaler(), scalers.StandardScaler()

    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        nans = np.isnan(x)
        x = self.scaler_x.transform(x)
        x[np.isnan(x)] = 0
        feed_dict = {self.x_tf: x, self.keep_prob: 1.}
        if not self.zeronans:
            W_nograd = compute_nograd(nans, self.arch[0])
            feed_dict[self.W_nograd_tf] = W_nograd
        y_pred = sess.run(self.y_pred, feed_dict)
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, sess, epochs=1000, batch_size=32,
            lr=1e-3, valsize=.1, nn_verbose=False,
            dropout_keep_prob=1., writer=None, summary=None, **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            sess: Tensorflow session.
            epochs (int): How many batches to train for.
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            valsize (float): Proportion of data
                (between 0 and 1) for validation.
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
        if not 0 <= valsize < 1:
            raise ValueError('Invalid validation dataset size')
        n_val = int(n_samples * float(valsize))
        if valsize > 0:
            n_val = max(1, n_val)
        nans = np.isnan(x)
        ids_perm = np.random.permutation(n_samples)

        # Copy the data into new arrays before normalization.
        # TODO: This wastes memory: ideally user would normalize
        # the data. Maybe add a 'normalize' option instead?
        x = np.array(x)
        y = np.array(y)

        # Choose training and validation data randomly.
        x[ids_perm[n_val:]] = self.scaler_x.fit_transform(x[ids_perm[n_val:]])
        y[ids_perm[n_val:]] = self.scaler_y.fit_transform(y[ids_perm[n_val:]])
        if n_val > 0:
            x[ids_perm[:n_val]] = self.scaler_x.transform(x[ids_perm[:n_val]])
            y[ids_perm[:n_val]] = self.scaler_y.transform(y[ids_perm[:n_val]])
        if self.zeronans:
            x[np.isnan(x)] = 0
            y[np.isnan(y)] = 0

        # Train the neural net.
        tr_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        best_val = np.inf
        start_time = time.time()
        batch_size = min(batch_size, n_samples-n_val)

        for epoch_id in range(epochs):
            batch_ids = ids_perm[n_val +
                np.random.choice(n_samples-n_val,  batch_size, replace=False)]
            feed_dict = {self.x_tf: x[batch_ids], self.y_tf: y[batch_ids],
                self.keep_prob: dropout_keep_prob, self.lr_tf: lr}
            if not self.zeronans:
                W_nograd = compute_nograd(nans[batch_ids], self.arch[0])
                feed_dict[self.W_nograd_tf] = W_nograd
            if summary is None:
                _, tr_loss = sess.run(
                    [self.train_op_tf, self.loss_tf],
                    feed_dict)
            else:
                _, tr_loss, s = sess.run(
                    [self.train_op_tf, self.loss_tf, summary],
                    feed_dict)
            if n_val > 0:
                feed_dict[self.x_tf] = x[ids_perm[:n_val]]
                feed_dict[self.y_tf] = y[ids_perm[:n_val]]
                val_loss = sess.run(self.loss_tf, feed_dict)
            else:
                val_loss = tr_loss

            if writer is not None:
                # Add summaries for Tensorboard.
                writer.add_summary(s, epoch_id)

            # Store losses.
            tr_losses[epoch_id] = tr_loss
            val_losses[epoch_id] = val_loss
            if val_loss < best_val:
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
        # Restore the net with best validation loss.
        self.saver.restore(sess, model_path)
        if nn_verbose:
            print(('Trainig done in {} epochs, {}s.'
                   ' Validation loss {:.4g}.').format(
                   epoch_id, tr_time, best_val))
        return tr_losses, val_losses

if __name__ == "__main__":
    """ Check that everything works as expected. Should take several 
    seconds on a CPU machine. """
    import numpy as np
    from matplotlib import pyplot as plt
    n_samples = 1000
    n_val = int(n_samples / 10)
    # Make a dataset: 1D inputs, 1D outputs.
    x = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    y = np.sin(5 * x ** 2) + x + np.random.randn(*x.shape) * .1

    # Set some inputs to NaN, to show that we can deal with missing data.
    perm_ids = np.random.permutation(n_samples)
    x[perm_ids[:int(n_samples/10)]] = np.nan

    # Create the neural net.
    nn = NN(x_dim=x.shape[1], y_dim=y.shape[1], arch=[32]*10, ntype='plain')

    # Create a tensorflow session and run the net inside.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/{}'.format('plain'))
        writer.add_graph(sess.graph)

        # Fit the net.
        nn.fit(x[perm_ids[:-n_val]], y[perm_ids[:-n_val]],
               sess=sess, nn_verbose=True,
               writer=writer, summary=summary)

        # Predict.
        y_pred = nn.predict(x[perm_ids[-n_val:]], sess=sess)

    # Visualize the predicions.
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x[perm_ids[-n_val:]], y[perm_ids[-n_val:]], '.', label='data')
    plt.plot(x[perm_ids[-n_val:]], y_pred, '.', label='prediction')
    plt.legend(loc=0)
    plt.show()
