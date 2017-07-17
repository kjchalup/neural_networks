""" An implementation of the Fully Convolutional
Neural Network (with residual connections).

Krzysztof Chalupka, 2017.
"""
import sys
import time
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf

from neural_networks import scalers


class FCNN(object):
    """ A Fully Convolutional Neural Net with residual connections.

    The interface mimics that of sklearn: first, initialize the FCNN.
    Fit with FCNN.fit(), predict with FCNN.predict().

    Args:
        x_shape (int, int, int): Input shape (h, w, d).
            NOTE: Right now, output shape is
            restricted to be the same as the input shape. You can easily
            modify this behavior if needed by adjusting the architecture.
        n_layers (int): Number of fully-convolutional residual layers.
        n_filters: An integer array of shape (n_layers,). Number of filters
            in each layer.
        x_tf (tf.placeholder): If given, use as graph input.

    TODO: This first version has no residual connections!
    """
    def __init__(self, x_shape, n_layers=3, n_filters=np.array([4] * 3),
        x_tf=None, **kwargs):
        self.x_shape = x_shape
        self.n_layers = n_layers
        self.n_filters = n_filters
        if n_layers != len(n_filters):
            raise ValueError('Number of layers must equal len(n_filters)')

        # Set up the placeholders.
        if x_tf is None:
            self.x_tf = tf.placeholder(
                tf.float32, [None, x_shape[0], x_shape[1], x_shape[2]], name='input')
        else:
            self.x_tf = x_tf
        self.y_tf = tf.placeholder(
            tf.float32, [None, x_shape[0], x_shape[1], x_shape[2]], name='output')
        self.lr_tf = tf.placeholder(tf.float32, name='learningrate')

        # Inference.
        self.y_pred = self.define_fcnn(**kwargs)

        # Loss.
        with tf.name_scope('loss'):
            self.loss_tf = self.define_loss()

        # Training.
        self.train_op_tf = self.define_training()

        # Define the saver object for model persistence.
        self.tmpfile = NamedTemporaryFile()
        self.saver = tf.train.Saver(max_to_keep=2)

        # Define the data scaler.
        self.scaler_x, self.scaler_y = self.define_scalers()

    def define_fcnn(self, **kwargs):
        """ Define the FCNN. """
        y_pred = self.x_tf

        # Create the hidden layers.
        for layer_id in range(self.n_layers):
            with tf.name_scope('layer{}'.format(layer_id)):
                y_pred = tf.layers.conv2d(
                    y_pred, filters=self.n_filters[layer_id], kernel_size=3,
                    padding='same', activation=tf.nn.elu)
                tf.summary.histogram('feature_map', y_pred)
    
        # The final layer polls depth channels with 1x1 convolutions.
        with tf.name_scope('1x1conv'):
            y_pred = tf.layers.conv2d(
                y_pred, filters=self.x_shape[2], kernel_size=1, padding='same',
                activation=None)
            tf.summary.histogram('feature_map', y_pred)

        return y_pred

    def define_loss(self):
        loss = tf.losses.mean_squared_error(self.y_tf, self.y_pred)
        return loss

    def define_training(self):
        return tf.train.AdamOptimizer(self.lr_tf).minimize(self.loss_tf)

    def define_scalers(self):
        return scalers.StandardScaler(), scalers.StandardScaler()

    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, h, w, c): Input data.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        nans = np.isnan(x)
        x = self.scaler_x.transform(x)
        x[np.isnan(x)] = 0
        feed_dict = {self.x_tf: x}
        y_pred = sess.run(self.y_pred, feed_dict)
        return self.scaler_y.inverse_transform(y_pred)

    def fit(self, x, y, sess, epochs=1000, batch_size=32,
            lr=1e-3, valsize=.1, nn_verbose=True,
            writer=None, summary=None, **kwargs):
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
            writer: A writer object for Tensorboard bookkeeping.
            summary: Summary object for the writer.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).
        """
        tr_summary = tf.summary.scalar("training_loss", self.loss_tf)
        val_summary = tf.summary.scalar("validation_loss", self.loss_tf)

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
        # the data. Add a 'normalize' option instead?
        x = np.array(x)
        y = np.array(y)

        # Choose training and validation data randomly.
        x[ids_perm[n_val:]] = self.scaler_x.fit_transform(x[ids_perm[n_val:]])
        y[ids_perm[n_val:]] = self.scaler_y.fit_transform(y[ids_perm[n_val:]])
        if n_val > 0:
            x[ids_perm[:n_val]] = self.scaler_x.transform(x[ids_perm[:n_val]])
            y[ids_perm[:n_val]] = self.scaler_y.transform(y[ids_perm[:n_val]])
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
                np.random.choice(n_samples-n_val, batch_size, replace=False)]
            feed_dict = {self.x_tf: x[batch_ids],
                         self.y_tf: y[batch_ids],
                         self.lr_tf: lr}

            if summary is None:
                _, tr_loss = sess.run(
                    [self.train_op_tf, self.loss_tf],
                    feed_dict)
            else:
                _, tr_loss, s, trs = sess.run(
                    [self.train_op_tf, self.loss_tf, summary, tr_summary],
                    feed_dict)

            if n_val > 0:
                feed_dict[self.x_tf] = x[ids_perm[:n_val]]
                feed_dict[self.y_tf] = y[ids_perm[:n_val]]
                val_loss, vals = sess.run([self.loss_tf, val_summary],
                                          feed_dict)
            else:
                val_loss = tr_loss

            if writer is not None:
                # Add summaries for Tensorboard.
                writer.add_summary(s, epoch_id)
                writer.add_summary(trs, epoch_id)
                writer.add_summary(vals, epoch_id)

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
    """ Check that the network works as expected. Denoise MNIST. 
    Takes about a minute on a Titan X GPU.
    """
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ims_tr = mnist.train.images.reshape(-1, 28, 28, 1)
    ims_ts = mnist.test.images.reshape(-1, 28, 28, 1)

    # Create a dataset of MNIST with random Gaussian noise as inputs
    # and the original digits as outputs.
    X_tr = ims_tr + np.random.randn(*ims_tr.shape) * .1
    Y_tr = ims_tr
    X_ts = ims_ts + np.random.randn(*ims_ts.shape) * .1
    Y_ts = ims_ts

    # Define the graph.
    fcnn = FCNN(x_shape = ims_tr.shape[1:])

    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/fcnn')
        writer.add_graph(sess.graph)

        # Fit the net.
        import pdb; pdb.set_trace()
        fcnn.fit(X_tr, Y_tr, sess, writer=writer, summary=summary)

        # Predict.
        Y_pred = fcnn.predict(X_ts, sess).reshape(-1, 28, 28)
    
    # Show the results.
    plt.figure(figsize=(16, 6))
    for im_id_id, im_id in enumerate(np.random.choice(X_ts.shape[0], 8)):
        plt.subplot2grid((3, 8), (0, im_id_id))
        plt.title('Noisy')
        plt.axis('off')
        plt.imshow(X_ts[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')

        plt.subplot2grid((3, 8), (1, im_id_id))
        plt.title('Denoised')
        plt.axis('off')
        plt.imshow(Y_pred[im_id], cmap='Greys', interpolation='nearest')

        plt.subplot2grid((3, 8), (2, im_id_id))
        plt.title('Original')
        plt.axis('off')
        plt.imshow(Y_ts[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')

    plt.show()
