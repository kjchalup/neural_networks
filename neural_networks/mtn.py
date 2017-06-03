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

def make_data(n_samples=100, dim=100, noise_std=1):
    projection_mat = np.random.uniform(low=-1, high=1, size=(2, dim))
    while True:
        # Sample points on a circle and project up.
        x_circ = np.random.uniform(low=0, high=2*np.pi, size=(n_samples, 1))
        X = np.hstack([np.cos(x_circ), np.sin(x_circ)])
        X = np.matmul(X, projection_mat)
        X += np.random.randn(*X.shape) * noise_std
        X = np.sin(np.log(np.abs(X))**2)
        # Outputs are a randomized, noisy function on the circle.
        Y = x_circ + np.random.randn(n_samples, 1) * .1
        a = np.random.choice(np.arange(-4, 4))
        b = np.random.uniform(low=-10, high=10)
        c = np.random.uniform(low=-10, high=10)
        Y = Y ** a * np.sin(b * np.abs(Y)) + c * Y
        Y += np.random.randn(*Y.shape) * noise_std
        yield (X, Y)

if __name__=="__main__":
    """ Check that everything works as expected. Should take several
    seconds on a CPU machine. """
    import numpy as np
    from neural_networks import nn
    import matplotlib.pyplot as plt

    n_tasks = 10
    dim = 10
    samples = 1000
    epochs = 100
    noise_std = .01
    kwargs = {
              'arch': [128] * 10,
              'ntype': 'resnet',
              'lr': 1e-2,
              'batch_size': 128
             }
    # Make data: X is a list of datasets, each with the same coordinates
    # but potentially 1) different n_samples and 2) different output tasks in Y.
    np.random.seed(1)
    X, Y = zip(*[next(make_data(samples, dim, noise_std)) for _ in range(n_tasks)])
    X_task = [np.zeros((samples, n_tasks)) for _ in range(n_tasks)]
    X_multi = np.zeros((samples * n_tasks, n_tasks + dim))
    Y_multi = np.zeros((samples * n_tasks, n_tasks))
    for task_id, x in enumerate(X):
        X_multi[task_id*samples : (task_id+1)*samples, task_id:task_id+1] = 1.
        X_multi[task_id*samples : (task_id+1)*samples:, n_tasks:] = x
        Y_multi[task_id*samples : (task_id+1)*samples,
            task_id:task_id+1] = Y[task_id]

    # Train and evaluate a multi-task network.
    mtnet = MTN(x_dim=X_multi.shape[1], y_dim=n_tasks, **kwargs)
    mtnet.fit(X_multi, Y_multi, nn_verbose=True,
            epochs=epochs*min(10, n_tasks), **kwargs)
    errs_mtn = []
    for task_id in range(n_tasks):
        mtnpred = mtnet.predict(X_multi[task_id*samples:(task_id+1)*samples])
        mtnpred = mtnpred[:, task_id:task_id+1]
        errs_mtn.append(np.mean((mtnpred - Y[task_id])**2))
    mtnet.close()

    # Train a single nn for each task, for comparison.
    errs_nn = []
    for task_id in range(n_tasks):
        nnet = nn.NN(x_dim=X[0].shape[1], y_dim=1, **kwargs)
        nnet.fit(X[task_id], Y[task_id], nn_verbose=True, epochs=epochs, **kwargs)
        nnpred = nnet.predict(X[task_id])
        errs_nn.append(np.mean((nnpred - Y[task_id])**2))
        nnet.close()

    fig = plt.figure(figsize=(5, 5))
    plt.xlabel('Task ID')
    plt.ylabel('Error')
    plt.title('MTN error to NN error ratio')
    print(errs_mtn)
    print(errs_nn)
    plt.bar(np.arange(n_tasks), np.log(errs_mtn) - np.log(errs_nn), width=.8)
    plt.axhline(y=0)
    plt.savefig('res.png')
