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

def bn_relu_conv(in_tf, is_training_tf, n_filters=16,
    kernel_size=3, stride=(1, 1), nonlin=tf.nn.elu, residual=False, reuse=False):
    """ A convolutional resnet building block.
    
    Pushes in_tf through batch_norm, relu, and convolution.
    If residual = True, make two bn_relu_conv layers and add
    a skip connection. All convolutional filters are 3x3.

    Args:
        in_tf: Input tensor.
        is_training_tf: bool tensor, indicates whether we're in the training
            phase or not (used by batch_norm and dropout).
        n_filters (int): Number of convolution filters.
        kernel_size (int): Size of the kernel.
        stride (tuple(int, int)): Kernel strides.
        residual (bool): Whether to make the layer residual.
        reuse (bool): Whether to reuse Tensorflow variables.

    Returns:
        out_tf: Output tensor.
    """

    # Apply batch normalization.
    out_tf = tf.layers.batch_normalization(
        in_tf, center=True, scale=True, training=is_training_tf)

    # Apply the nonlinearity.
    out_tf = nonlin(out_tf)

    # Apply convolutions.
    out_tf = tf.layers.conv2d(
        out_tf, filters=n_filters, kernel_size=kernel_size, strides=stride,
        padding='same', activation=None, reuse=reuse, name='conv1')

    if residual:
        out_tf = tf.layers.batch_normalization(
            out_tf, center=True, scale=True, training=is_training_tf)
        out_tf = nonlin(out_tf)
        out_tf = tf.layers.conv2d(
            out_tf, filters=n_filters, kernel_size=kernel_size, strides=stride,
            padding='same', activation=None, reuse=reuse, name='conv2')
        out_tf += in_tf
    tf.summary.histogram('ftr_map', out_tf)

    return out_tf


class FCNN(object):
    """ A Fully Convolutional Neural Net with residual connections.

    The interface mimics that of sklearn: first, initialize the FCNN.
    Fit with FCNN.fit(), predict with FCNN.predict().

    Args:
        x_shape (int, int, int): Input shape (h, w, d).
            NOTE: Right now, output h and w is
            restricted to be the same as the input shape. You can easily
            modify this behavior if needed by adjusting the architecture.
        y_channels (int): Number of output channels.
        n_layers (int): Number of fully-convolutional residual layers.
        n_filters: An integer array of shape (n_layers,). Number of filters
            in each layer.
        x_tf (tf.placeholder): If given, use as graph input.
        reuse (bool): Whether to reuse the net weights.
        bn (bool): Whether to use batch normalization.
        res (bool): Whether to add residual connections.
        save_fname (str): Checkpoint location. If None, use a temp file.

    TODO: This first version has no residual connections!
    """
    def __init__(self, x_shape, y_channels, n_layers=4, n_filters=None,
        x_tf=None, reuse=False, bn=True, res=True, save_fname=None, **kwargs):
        if n_filters is None:
            n_filters = np.array([64] * n_layers)
        if n_layers != len(n_filters):
            raise ValueError('Number of layers must equal len(n_filters)')
        self.x_shape = x_shape
        self.y_channels = y_channels
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.reuse = reuse
        self.bn = bn
        self.res = res
        self.save_fname = save_fname

        # Set up the placeholders.
        if x_tf is None:
            self.x_tf = tf.placeholder(tf.float32,
                [None, x_shape[0], x_shape[1], x_shape[2]], name='input')
        else:
            self.x_tf = x_tf
        self.y_tf = tf.placeholder(tf.float32,
            [None, x_shape[0], x_shape[1], y_channels], name='output')
        self.lr_tf = tf.placeholder(tf.float32, name='learningrate')
        self.is_training_tf = tf.placeholder(tf.bool, name='train_flag')

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

    def define_fcnn(self, **kwargs):
        """ Define the FCNN. """
        y_pred = self.x_tf

        # Local response normalization.
        y_pred = tf.nn.lrn(y_pred, name='lrn')

        # Use 1x1 convolutions to reshape the channel direction
        # for use with residual connections.
        if self.res:
            with tf.variable_scope('inflatten'):
                y_pred = tf.layers.conv2d(
                    y_pred, filters=1, kernel_size=1,
                    reuse=self.reuse)

        # Create the hidden layers.
        for layer_id in range(self.n_layers):
            with tf.variable_scope('layer{}'.format(layer_id)):
                y_pred = bn_relu_conv(y_pred, self.is_training_tf,
                    self.n_filters[layer_id], self.res, self.reuse)
                
        # The final layer polls depth channels with 1x1 convolutions.
        with tf.variable_scope('outflatten'):
            y_pred = tf.layers.conv2d(
                y_pred, filters=self.y_channels, kernel_size=1, padding='same',
                activation=None, reuse=self.reuse)

        return y_pred

    def define_loss(self):
        loss = tf.losses.mean_squared_error(self.y_tf, self.y_pred)
        return loss

    def define_training(self):
        # Make sure to include batch_norm dependencies during training.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(self.lr_tf).minimize(self.loss_tf)
        return train_op

    def predict(self, x, sess):
        """ Compute the output for given data.

        Args:
            x (n_samples, h, w, c): Input data.
            sess: Tensorflow session.

        Returns:
            y (n_samples, y_dim): Predicted outputs.
        """
        feed_dict = {self.x_tf: x, self.is_training_tf: False}
        y_pred = sess.run(self.y_pred, feed_dict)
        return y_pred

    def fit(self, sess, fetch_data, epochs=1000, batch_size=128,
            lr=1e-3, nn_verbose=True,
            writer=None, summary=None, **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            sess: Tensorflow session.
            fetch_data: A method that takes an int argument batch_size 
                and a string argument data_type. If data_type=='val', returns
                a batch of validation data (or None if there's no validation
                data available). If data_type=='train', returns a batch of
                training data.
            epochs (int): How many batches to train for.
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            nn_verbose (bool): Display training progress messages (or not).
            writer: A writer object for Tensorboard bookkeeping.
            summary: Summary object for the writer.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).
        """
        tr_summary = tf.summary.scalar("training_loss", self.loss_tf)
        val_summary = tf.summary.scalar("validation_loss", self.loss_tf)

        # Train the neural net.
        tr_losses = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        best_val = np.inf
        start_time = time.time()

        for epoch_id in range(epochs):
            x, y = fetch_data(batch_size, 'train')
            feed_dict = {self.x_tf: x,
                         self.y_tf: y,
                         self.is_training_tf: True,
                         self.lr_tf: lr}

            if summary is None:
                _, tr_loss = sess.run(
                    [self.train_op_tf, self.loss_tf],
                    feed_dict)
            else:
                _, tr_loss, s, trs = sess.run(
                    [self.train_op_tf, self.loss_tf, summary, tr_summary],
                    feed_dict)
            
            val_data = fetch_data(batch_size, 'val')
            if val_data is not None:
                x, y = val_data
                feed_dict[self.is_training_tf] = True
                feed_dict[self.x_tf] = x
                feed_dict[self.y_tf] = y
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
                if self.save_fname is not None:
                    model_path = self.saver.save(
                        sess, self.save_fname)
                else:
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
    import matplotlib
    from sklearn.preprocessing import StandardScaler
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    ims_tr = mnist.train.images.reshape(-1, 28, 28, 1)
    ims_ts = mnist.test.images.reshape(-1, 28, 28, 1)

    # Create a dataset of MNIST with random binary noise as inputs
    # and the original digits as outputs.
    X_tr = ims_tr + np.random.randn(*ims_tr.shape) * .1
    Y_tr = ims_tr
    X_ts = ims_ts + np.random.randn(*ims_ts.shape) * .1
    Y_ts = ims_ts
    perm_ids = np.random.permutation(X_tr.shape[0])

    def fetch_data(batch_size, data_type):
        n_train = int(perm_ids.size * .9)
        if data_type == 'train':
            ids = np.random.choice(perm_ids[:n_train], batch_size, replace=True)
        else:
            ids = np.random.choice(perm_ids[n_train:], batch_size, replace=True)
        return X_tr[ids], Y_tr[ids]

    # Define the graph.
    fcnn = FCNN(x_shape=X_tr.shape[1:],
        y_channels=Y_tr.shape[-1], bn=True, res=True)

    # Create a Tensorflow session and train the net.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/fcnn')
        writer.add_graph(sess.graph)

        # Fit the net.
        fcnn.fit(sess, fetch_data, epochs=100, writer=writer, summary=summary)

        # Predict.
        Y_pred = fcnn.predict(X_ts[:1000], sess).reshape(-1, 28, 28)
    
    # Show the results.
    plt.figure(figsize=(16, 6))
    for im_id in range(8):
        plt.subplot2grid((3, 8), (0, im_id))
        plt.title('Noisy')
        plt.axis('off')
        plt.imshow(X_ts[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')

        plt.subplot2grid((3, 8), (1, im_id))
        plt.title('Denoised')
        plt.axis('off')
        plt.imshow(Y_pred[im_id], cmap='Greys', interpolation='nearest')

        plt.subplot2grid((3, 8), (2, im_id))
        plt.title('Original')
        plt.axis('off')
        plt.imshow(Y_ts[im_id].reshape(28, 28),
            cmap='Greys', interpolation='nearest')

    plt.savefig('fcnn_results.png')
