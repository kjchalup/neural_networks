""" Multi-task networks. """
import sys
import time
from typing import List
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import tensorflow as tf

from neural_networks import nn


class HalfScaler(object):
    def __init__(self, ignore_dim):
        self.scaler = StandardScaler()
        self.ignore_dim = ignore_dim

    def fit(self, x):
        self.scaler.fit(x[:, self.ignore_dim:])

    def fit_transform(self, x):
        self.scaler.fit(x[:, self.ignore_dim:])
        xtr = self.scaler.transform(x[:, self.ignore_dim:])
        return np.hstack([x[:, :self.ignore_dim], xtr])

    def transform(self, x):
        xtr = self.scaler.transform(x[:, self.ignore_dim:])
        return np.hstack([x[:, :self.ignore_dim], xtr])

    def inverse_transform(self, y):
        ytr = self.scaler.inverse_transform(y[:, self.ignore_dim:])
        return np.hstack([y[:, self.ignore_dim], ytr])


class MTN(nn.NN):
    def __init__(self, x_dim, y_dim, 
            arch=[128, 128], ntype='plain', name='mtn', **kwargs):
        """ A multi-task network. 

        The output is a concatenation of the outputs for all n_task tasks.
        Let the tasks have output dimensionalities y1, ..., yn. The input
        then consists of:
        1) A task-flag section: a bit vector of length sum(yi), containing
            zeros everywhere except for coordinates corresponding to the task
            of the current input (where the bits are set to 1).
        2) The true input, which must have the same dimensionality
            for all tasks.
        These two input parts should be concatenated.
        """
        super().__init__(x_dim, y_dim, arch, ntype, **kwargs)

    def define_loss(self):
        x_dim = self.x_tf.get_shape().as_list()[1]
        y_dim = self.y_tf.get_shape().as_list()[1]

        return tf.losses.mean_squared_error(
            self.y_tf, self.y_pred * self.x_tf[:, :y_dim])

    def define_scalers(self):
        xscale = HalfScaler(ignore_dim=self.y_dim)
        yscale = StandardScaler()
        return xscale, yscale

def make_data(n_samples, noise_std, projection_mat, transform_mat):
    # Sample points on a circle and project up.
    x_circ = np.random.uniform(low=0, high=2*np.pi, size=(n_samples, 1))
    X = np.hstack([np.cos(x_circ), np.sin(x_circ)])
    X = np.matmul(X, projection_mat)
    X += np.random.randn(*X.shape)
    X = np.sin(np.log(np.abs(X))**2)
    X = np.matmul(X, transform_mat)
    X *= np.sin(X)**2
    # Outputs are a randomized, noisy function on the circle.
    Y = x_circ + np.random.randn(n_samples, 1)
    a = np.random.uniform(-1, 1)
    b = np.random.uniform(low=-2, high=2)
    c = np.random.uniform(low=-2, high=2)
    Y = a * Y**2 + b * Y + c + np.random.randn(*Y.shape) * noise_std
    yield (X, Y)

if __name__=="__main__":
    """ Check that everything works as expected. Should take several
    seconds on a CPU machine. """
    import numpy as np
    from neural_networks import nn, ae
    import matplotlib.pyplot as plt

    n_tasks = 100
    dim = 1000
    samples = 50
    epochs = 1000
    noise_std = .1
    kwargs = {
              'arch': [128],
              'ntype': 'plain',
              'lr': 1e-3,
              'batch_size': 32,
              'dropout_keep_prob': 1.
             }
    # Make data: X is a list of datasets, each with the same coordinates
    # but potentially 1) different n_samples and 2) different output tasks in Y.
    np.random.seed(1)
    projection_mat = np.random.uniform(low=-1, high=1, size=(2, dim))
    transform_mat = np.random.uniform(low=-1, high=1, size=(dim, dim))
    X, Y = zip(*[next(make_data(samples, noise_std, projection_mat, transform_mat))
        for _ in range(n_tasks)])
    X_multi = np.zeros((samples * n_tasks, n_tasks + dim))
    Y_multi = np.zeros((samples * n_tasks, n_tasks))
    for task_id, x in enumerate(X):
        X_multi[task_id*samples : (task_id+1)*samples, task_id:task_id+1] = 1.
        X_multi[task_id*samples : (task_id+1)*samples:, n_tasks:] = x
        Y_multi[task_id*samples : (task_id+1)*samples,
            task_id:task_id+1] = Y[task_id]

    # Train and evaluate a multi-task network.
    #mtnet = MTN(x_dim=X_multi.shape[1], y_dim=n_tasks, **kwargs)
    #mtnet.fit(X_multi, Y_multi, nn_verbose=True,
    #        epochs=epochs*min(10, n_tasks), **kwargs)
    errs_mtn = []
    #for task_id in range(n_tasks):
    #    mtnpred = mtnet.predict(X_multi[task_id*samples:(task_id+1)*samples])
    #    mtnpred = mtnpred[:, task_id:task_id+1]
    #    errs_mtn.append(np.mean((mtnpred - Y[task_id])**2))
    #mtnet.close()

    # Train a single nn for each task, for comparison.
    # Auto-encode Xs first.
    ae = ae.Autoencoder(x_dim=dim, arch=[2])
    ae.fit(np.vstack(X))
    Xae = np.split(ae.encode(np.vstack(X)), n_tasks, axis=0)
    ae.close()
    errs_nn = []
    errs_ae = []
    for task_id in range(n_tasks):
        print('Task {}: Training a vanilla NN'.format(task_id))
        nnet = nn.NN(x_dim=X[0].shape[1], y_dim=1, **kwargs)
        nnet.fit(X[task_id], Y[task_id], nn_verbose=True, epochs=epochs, **kwargs)
        nnpred = nnet.predict(X[task_id])
        errs_ae.append(np.mean((nnpred - Y[task_id])**2))
        nnet.close()

        print('Task {}: Training an ae-NN'.format(task_id))
        nnet = nn.NN(x_dim=Xae[0].shape[1], y_dim=1, **kwargs)
        nnet.fit(Xae[task_id], Y[task_id], nn_verbose=True, epochs=epochs, **kwargs)
        nnpred = nnet.predict(Xae[task_id])
        errs_nn.append(np.mean((nnpred - Y[task_id])**2))
        nnet.close()

    fig = plt.figure(figsize=(5, 5))
    plt.xlabel('Task ID')
    plt.ylabel('Error')
    plt.title('MTN error to NN error ratio')
    print(errs_mtn)
    print(errs_nn)
    plt.bar(np.arange(n_tasks), np.log(errs_ae) - np.log(errs_nn), width=.8)
    plt.axhline(y=0)
    plt.savefig('res1.png')

    fig = plt.figure(figsize=(5, 5))
    plt.xlabel('Task ID')
    plt.ylabel('Error')
    plt.title('MTN error to NN error ratio')
    print(errs_mtn)
    print(errs_nn)
    plt.bar(np.arange(n_tasks), np.log(errs_ae) - np.log(errs_nn), width=.8)
    plt.axhline(y=0)
    plt.savefig('res1.png')
