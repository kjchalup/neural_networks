""" Mixture Density Network (MDN) routines. """
import sys
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf


def sample_gmm(coeffs, sigmas, means):
    """ Sample from GMM mixtures.

    Args:
        coeffs (n_samples, n_comp): Mixture component coefficients.
        sigmas (n_samples, n_comp): Component stds.
        means (n_samples, n_comp): Component means.

    Returns:
        samples (n_samples, 1): Samples.
    """
    n_samples, n_comp = coeffs.shape
    samples = np.zeros((n_samples, 1))
    for sample_id in range(n_samples):
        comp = np.random.choice(n_comp, p=coeffs[sample_id])
        samples[sample_id] = (np.random.randn() * sigmas[sample_id, comp]
            + means[sample_id, comp])
    return samples


def get_mdn_params(mdn_out):
    """ Convert mdn outputs into GMM parameters.

    Args:
        mdn_out (n_samples, n_comp * 3): MDN outputs.

    Returns:
        coeffs (n_samples, n_comp): GMM component coefficients.
        sigmas (n_samples, n_comp): GMM standard deviations.
        means (n_samples, n_comp): GMM means.
        n_comp (int): Number of GMM components.
    """
    n_comp = int(mdn_out.shape[1])
    if n_comp % 3 != 0:
        raise ValueError('mdn_out.shape[1] must be divisible by 3.')
    n_comp //= 3
    n_comp_tf = tf.cast(n_comp, tf.int32)
    coeffs = tf.nn.softmax(mdn_out[:, :n_comp_tf])
    sigmas = 1e-3 + tf.exp(mdn_out[:, n_comp_tf:n_comp_tf*2])
    means = mdn_out[:, n_comp_tf*2:n_comp_tf*3]
    return coeffs, sigmas, means, n_comp


def mdn_loglik(y, mdn_out):
    """ Compute the likelihoods of y given MDN parameters.

    Args:
        y (n_samples, 1): Data tensor, must be 1D.
        mdn_out (n_samples, n_comp * 3): MDN parameters.

    Return:
        logliks (n_samples, 1): Log-likelihood of the data.
    """
    coeffs, sigmas, means, n_comp = get_mdn_params(mdn_out)
    loglik = 0
    for comp_id in range(n_comp):
        coeff = coeffs[:, comp_id:comp_id+1]
        sigma2 = tf.pow(sigmas[:, comp_id:comp_id+1], 2)
        mean = means[:, comp_id:comp_id+1]
        unnormalized = tf.div(tf.pow(y - mean, 2), 2 * sigma2)
        normalizer = tf.sqrt(2 * sigma2 * np.pi)
        loglik += coeff * tf.exp(-unnormalized) / normalizer
    return tf.log(loglik)


def define_nn(x_tf, y_dim, Ws, bs, keep_prob):
    """ Define a Neural Network.

    The architecture of the network is deifned by the Ws, list of weight
    matrices. The last matrix must be of shape (?, y_dim). If the number
    of layers is lower than 3, use the sigmoid nonlinearity.
    Otherwise, use the relu.

    Args:
        x_tf (n_samples, x_dim): Input data.
        y_dim: Output data dimensionality.
        Ws: List of weight tensors.
        bs: List of bias tensors.
        keep_prob: Dropout probability of keeping a unit on.

    Returns:
        out: Predicted y.

    Raises:
        ValueError: When the last weight tensor's output is not compatible
            with the input shape.
    """

    nonlin = tf.nn.sigmoid
    out = tf.add(tf.matmul(x_tf, Ws[0]), bs[0])
    for layer_id, (W, b) in enumerate(zip(Ws[1:], bs[1:])):
        with tf.name_scope('hidden{}'.format(layer_id)):
            out = nonlin(out)
            out = tf.nn.dropout(out, keep_prob=keep_prob)
            out = tf.add(tf.matmul(out, W), b)
    return out


class MDN(object):
    """ A Mixture Density Network object.

    The interface mimics that of sklearn: first,
    initialize the MDN.

    Args:
        x_dim: Input data dimensionality.
        y_dim: Output data dimensionality. NOTE: For now,
            only 1-dimensional outputs are supported.
        n_comp: Number of GMM components.
        arch: A list of integers, each corresponding to the number
            of units in the NN's hidden layer.
    """

    def __init__(self, x_dim, y_dim, n_comp, arch=[1024], **kwargs):
        if y_dim != 1:
            raise NotImplementedError('Only MDNs with 1D outputs are' 
                                      'currently supported.')
        self.arch = arch
        self.n_comp = n_comp
        self.x_tf = tf.placeholder(
            tf.float32, [None, x_dim], name='input_data')
        self.y_tf = tf.placeholder(
            tf.float32, [None, y_dim], name='output_data')
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')
        self.y_dim = y_dim

        # Initialize the weights.
        if len(arch) > 0:
            self.Ws = [tf.truncated_normal([
                x_dim, self.arch[0]], stddev=.1 / x_dim, dtype=tf.float32)]
            for layer_id in range(len(arch)-1):
                self.Ws.append(tf.truncated_normal(
                    [arch[layer_id], arch[layer_id+1]],
                    stddev=1. / arch[layer_id], dtype=tf.float32))
            self.Ws.append(tf.truncated_normal(
                [arch[-1], 3 * n_comp], stddev=.1 / arch[-1], dtype=tf.float32))
        else:
            self.Ws = [tf.truncated_normal([x_dim, 3 * n_comp],
                stddev=.1 / x_dim, dtype=tf.float32)]
        self.Ws = [tf.Variable(W_init) for W_init in self.Ws]

        # Initialize the biases.
        self.bs = [tf.Variable(tf.zeros((1, 1)), dtype=tf.float32)
                   for num_units in arch]
        self.bs.append(tf.Variable(tf.zeros((1, 1)), dtype=tf.float32))

        # Initialize dropout keep_prob.
        self.keep_prob = tf.placeholder(tf.float32)

        # Define the MDN outputs as a function of input data.
        self.y_pred = define_nn(
                self.x_tf, y_dim, self.Ws, self.bs, self.keep_prob)

        # Define the loss function: MSE.
        self.loss_tf = -tf.reduce_mean(mdn_loglik(self.y_tf, self.y_pred))

        # Define the optimizer.
        self.train_op_tf = tf.train.AdamOptimizer(
            self.lr_tf).minimize(self.loss_tf)

        # Define the data scaler.
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Define the saver object for model persistence.
        self.saver = tf.train.Saver(max_to_keep=1)

        # Define the Tensorflow session, and its initializer op.
        self.sess = tf.Session()
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

    def loglik(self, x, y):
        """ Compute the log-likelihood of the data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.
            y (n_samples, 1): Output data.

        Returns:
            loglik (n_samples, 1): Log-likelihood of the data.
        """
        try:
            x = self.scaler_x.transform(x)
            y = self.scaler_y.transform(y)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        loglik = self.sess.run(
            mdn_loglik(self.y_tf, self.y_pred),
            {self.x_tf: x,
             self.y_tf: y,
             self.keep_prob: 1.})
        return loglik

    def sample(self, x):
        """ Sample from the MDN.

        Args:
        x (n_samples, x_dim): Input points.

        Returns:
        smpls (n_samples, 1): Samples.
            Sample from P(Y | X=x_pt) for each x_pt in x.
        """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        n_grid = y_grid.shape[0]
        samples = np.zeros((x.shape[0], 1))
        coeffs, sigmas, means, n_comp = get_mdn_params(self.y_pred)
        coeffs, sigmas, means = self.sess.run([coeffs, sigmas, means],
                {self.x_tf: x, self.keep_prob: 1.})
        return self.scaler_y.inverse_transform(
                sample_gmm(coeffs, sigmas, means))

    def predict(self, x):
        """ Compute the output for given data.

        Args:
            x (n_samples, x_dim): Input data.
                x_dim must agree with that passed to __init__.

        Returns:
            y (n_samples, y_dim): Predicted expected outputs.
        """
        try:
            x = self.scaler_x.transform(x)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        coeffs, sigmas, means, n_comp = get_mdn_params(self.y_pred)
        coeffs, sigmas, means = self.sess.run([coeffs, sigmas, means],
                {self.x_tf: x, self.keep_prob: 1.})
        return self.scaler_y.inverse_transform(
                np.sum(coeffs * means, axis=1, keepdims=True))

    def fit(self, x, y, max_epochs=1000, min_epochs=10, batch_size=32,
            lr=1e-3, max_time=np.inf, nn_verbose=True,
            max_nonimprovs=30, dropout_keep_prob=1., **kwargs):
        """ Train the MDN to maximize the data likelihood.

        Args:
            x (n_samples, x_dim): Input data.
            y (n_samples, y_dim): Output data.
            max_epochs (int): Max number of training epochs. Each epoch goes
                through the whole dataset (possibly leaving out
                mod(x.shape[0], batch_size) data).
            min_epochs (int): Do at least this many epochs (even if max_time
                already passed).
            batch_size (int): Training batch size.
            lr (float): Learning rate.
            max_time (float): Maximum training time, in seconds. Training will
                stop if max_time is up OR num_epochs is reached.
            nn_verbose (bool): Display training progress messages (or not).
            max_nonimprovs (int): Number of epochs allowed without improving
                the validation score before quitting.
            dropout_keep_prob (float): Probability of keeping a unit on.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).
        """
        if max_epochs < min_epochs:
            print('Setting max_epochs to {}'.format(min_epochs))
            max_epochs = min_epochs
        # Split data into a training and validation set.
        n_samples = x.shape[0]
        n_val = int(n_samples / 10)
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
        tr_losses = np.zeros(max_epochs)
        val_losses = np.zeros(max_epochs)
        best_val = np.inf
        start_time = time.time()
        for epoch_id in range(max_epochs):
            batch_ids = np.random.choice(n_samples-n_val,
                    batch_size, replace=False)
            self.sess.run(self.train_op_tf,
                          {self.x_tf: x_tr[batch_ids],
                           self.y_tf: y_tr[batch_ids],
                           self.keep_prob: dropout_keep_prob,
                           self.lr_tf: lr})
            tr_loss = self.sess.run(
                self.loss_tf, {self.x_tf: x_tr[batch_ids],
                               self.y_tf: y_tr[batch_ids],
                               self.keep_prob: 1.})
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
                    self.sess, './tmp')

            tr_time = time.time() - start_time
            if nn_verbose:
                sys.stdout.write(('\rTraining epoch {}, time {}s. Tr loss '
                                  '{:.4g}, val loss {:.4g}, best val {:.4g}.'
                              ).format(epoch_id, int(tr_time),
                                       tr_loss, val_loss, best_val))
                sys.stdout.flush()
            # Finish training if:
            # 1) min_epochs are done, and
            # 2a) either we're out of time, or
            # 2b) there was no validation score
            #     improvement for max_nonimprovs epochs.
            if (epoch_id >= min_epochs
                and (time.time() - start_time > max_time
                or epoch_id - last_improved > max_nonimprovs)):
                break

        self.saver.restore(self.sess, model_path)
        if nn_verbose:
            print(('Trainig done in {} epochs, {}s.'
               'Validation loss {:.4g}.').format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses

if __name__ == "__main__":
    """ Check that everything works as expected. Should take about 5min
    on a cpu machine. """
    import numpy as np
    from matplotlib import pyplot as plt

    # Make test data: a noisy mix of functions sampled with different probs.
    x = np.linspace(-1, 1, 10000).reshape(-1, 1)
    y1 = x**2 + np.random.randn(*x.shape) * .3
    y2 = -x/2-1 + np.random.randn(*x.shape) * .1
    y3 = -x-2 + np.random.randn(*x.shape) * .1
    y4 = x-3 + np.random.randn(*x.shape) * .1
    y = np.hstack([y1, y2, y3, y4])
    ids = np.random.choice(4, y.shape[0], p=[.7, .1, .1, .1])
    y = x + y[np.arange(y.shape[0]), ids].reshape(-1, 1)

    # Fit a mixture-density network. You should see the negative log-likelihood
    # decrease to about zero.
    mdn = MDN(x_dim=x.shape[1], y_dim=y.shape[1], n_comp=10, arch=[32, 32])
    mdn.fit(x, y, nn_verbose=True, lr=1e-2,
            min_epochs=10000, max_epochs=10000, batch_size=128)

    # Predict the expected value of P(Y | X).
    y_pred = mdn.predict(x)

    # Compute the likelihood and sample from P(Y | X).
    x_grid = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_grid = np.linspace(y.min(), y.max(), 100)
    logliks = mdn.loglik(np.tile(x_grid, [100, 1]),
            np.repeat(y_grid, 100).reshape(-1, 1))
    logliks = np.exp(logliks)
    samples = mdn.sample(x=x)

    # Compute the most-likely Y given each X.
    most_lik = y_grid[np.argmax(logliks.reshape(100, 100), axis=0)]

    # Plot the answers.
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.title('Learning expected y | x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'k.', alpha=.2)
    plt.plot(x, y_pred)

    plt.subplot(1, 3, 2)
    plt.title('Estimated p(y | x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.imshow(logliks.reshape(100, 100), interpolation='nearest',
            extent=[x.min(), x.max(), samples.min(), samples.max()],
            aspect='auto', cmap='coolwarm', origin='low')

    plt.subplot(1, 3, 3)
    plt.title('Learning most likely y | x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'k.', x_grid, most_lik)

    plt.show()
