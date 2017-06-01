""" Multi-task network routines. """
import sys
import time
from tempfile import NamedTemporaryFile

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np
import tensorflow as tf
from typing import List

def rescale(data: np.ndarray,
        scalers: List[StandardScaler],
        tasks: np.ndarray,
        direction: int) -> np.ndarray:
    """ Apply scaler.transform to each datapoint. 

    Args:
        data (n_samples, data_dim): Data to be transformed.
        scalers [StandardScaler]: A list of scalers -- must be 
            pre-fitted.
        tasks (n_samples,): Task ids. Each corresponds to one
            of the scalers.
        direction (int): If 0, scaler.transform. If 1, scaler.inverse_transform.

    Returns:
        rescaled (n_samples, data_dim): Rescaled data.
    """
    rescaled = np.zeros_like(data)
    for data_id, task in enumerate(tasks):
        if direction == 0:
            scaler = scalers[task].transform 
        else:
            scaler = scalers[task].inverse_transform
        out = scaler(np.atleast_2d(data[data_id]))
        rescaled[data_id][:] = out.flatten()
    return rescaled


def define_mtn(x: tf.placeholder,
               y_dims: List[int],
               W_shared: List[tf.placeholder],
               b_shared: List[tf.placeholder],
               W_tasks: List[List[tf.Variable]],
               b_tasks: List[List[tf.Variable]],
               keep_prob: tf.placeholder,
               phase: tf.placeholder, 
               batch_norm: bool) -> List[tf.Tensor]:
    """ Define a Multi-Task Network in Tensorflow.

    Args:
        x: Input variable.
        y_dims: List of output dimensionalities (one per each task).
        W_shared: Hidden layers following x.
        b_shared: Biases following x.
        W_tasks: Task-specific hidden layers.
        b_tasks: Task-specific biases.
        keep_prob: Probability of keeping a unit on (for dropout).
        phase: Indicates whether we're training or not (for batch norm).
        batch_norm: If True, use batch normalization.

    Returns:
        y_pred: List of output tensors, one per task.
    """
    out_dims = [Ws[-1].get_shape().as_list()[1] for Ws in W_tasks]
    if out_dims != y_dims:
        raise ValueError('MTN output dimension is not '
                'compatible with output shape.')
    if len(W_shared) + len(W_tasks[0]) < 3:
        nonlin = tf.nn.sigmoid
    else:
        nonlin = tf.nn.relu

    # Create the shared layers.
    with tf.name_scope('sharedhidden0'):
        out = tf.add(tf.matmul(x, W_shared[0]), b_shared[0])
    for layer_id, (W, b) in enumerate(zip(W_shared[1:], b_shared[1:])):
        with tf.name_scope('sharedhidden{}'.format(layer_id+1)):
            if batch_norm:
                out = tf.contrib.layers.batch_norm(out, 
                        center=True, scale=True, is_training=phase)
            out = nonlin(out)
            out = tf.nn.dropout(out, keep_prob=keep_prob)
            out = tf.add(tf.matmul(out, W), b)

    # Create the task-specific layers.
    y_preds = []
    for task_id in range(len(y_dims)):
        with tf.name_scope('task{}hidden0'.format(task_id)):
            if batch_norm:
                out_s = tf.contrib.layers.batch_norm(out, 
                        center=True, scale=True, is_training=phase)
                out_s = nonlin(out_s)
            else:
                out_s = nonlin(out)
            out_s = tf.nn.dropout(out_s, keep_prob=keep_prob)
            out_s = tf.add(tf.matmul(out_s, W_tasks[task_id][0]), b_tasks[task_id][0])
        for layer_id, (W, b) in enumerate(
                zip(W_tasks[task_id][1:], b_tasks[task_id][1:])):
            with tf.name_scope('task{}hidden{}'.format(task_id, layer_id+1)):
                if batch_norm:
                    out_s = tf.contrib.layers.batch_norm(out_s, 
                            center=True, scale=True, is_training=phase)
                out_s = nonlin(out_s)
                out_s = tf.nn.dropout(out_s, keep_prob=keep_prob)
                out_s = tf.add(tf.matmul(out_s, W), b)
        y_preds.append(out_s)

    return y_preds


class MTN(object):
    def __init__(
            self,
            x_dim: int,
            y_dims: List[int],
            shared_arch: List[int],
            task_arch: List[int],
            batch_norm: bool):
        """ A Multi-Task Network. 

        Creates a neural net that can be trained to perform `n_tasks`
        that share the input space and hidden layers, but have different
        output tasks. Thus, the last layer consists of sum(y_dims) units,
        each group of y_dims[i] units connected to the penultimate layer
        with a weight matrix. 
        
        Upon training, on each step the task_id
        must be provided, and the gradient propagates only through the
        units relevant to the task.

        Args:
            x_dim: Input dimensionality.
            y_dims: Output task dimensionalities.
            shared_arch: Weight numbers in shared hidden layers.
            task_arch: Weight numbers in each of the task-specific layers.
            batch_norm: If True, use batch normalization.
        """
        self.n_tasks = len(y_dims)
        self.x_dim = x_dim
        self.y_dims = y_dims
        self.n_tasks = len(y_dims)

        # Define Tensorflow placeholders.
        self.lr_tf = tf.placeholder(tf.float32, name='learning_rate')
        self.x_tf = tf.placeholder(tf.float32, [None, x_dim], name='X')
        self.ys_tf = [tf.placeholder(tf.float32, [None, y_dims[task_id]],
            name='Y_{}'.format(task_id)) for task_id in range(self.n_tasks)]
        self.keep_prob_tf = tf.placeholder(tf.float32)
        self.phase_tf = tf.placeholder(tf.bool)

        # Initialize shared weights:
        # 1) The first shared hidden layer.
        self.W_shared = [tf.truncated_normal([
            x_dim, shared_arch[0]], stddev=1./x_dim, dtype=tf.float32)]
        # 2) Remaining shared hidden layers.
        for layer_id in range(len(shared_arch)-1):
            self.W_shared.append(tf.truncated_normal(
                [shared_arch[layer_id], shared_arch[layer_id+1]],
                stddev=1. / shared_arch[layer_id], dtype=tf.float32))
        self.W_shared = [tf.Variable(W_init) for W_init in self.W_shared]
        # 3) Shared biases.
        self.b_shared = [tf.Variable(tf.zeros((1, 1)), dtype=tf.float32)
                for _ in shared_arch]

        # Initialize task-specific weights.
        # 1) First task-specific layer (one for each task).
        self.W_tasks = [[tf.truncated_normal([
            shared_arch[-1], task_arch[0]],
            stddev=1./x_dim, dtype=tf.float32)]
            for _ in range(self.n_tasks)]
        # 2) Remaining task-specific layers (one copy of each per task).
        for layer_id in range(len(task_arch)-1):
            for task_id in range(self.n_tasks):
                self.W_tasks[task_id].append(tf.truncated_normal(
                    [task_arch[layer_id], task_arch[layer_id+1]],
                    stddev=1. / task_arch[layer_id], dtype=tf.float32))
        # 3) Finally, make the output layer weights.
        for task_id in range(self.n_tasks):
            self.W_tasks[task_id].append(tf.truncated_normal(
                [task_arch[-1], self.y_dims[task_id]],
                stddev=1. / task_arch[-1], dtype=tf.float32))
        for task_id in range(self.n_tasks):
            self.W_tasks[task_id] = [tf.Variable(W_init)
                    for W_init in self.W_tasks[task_id]]
        # Task-specific biases.
        self.b_tasks = [[tf.Variable(tf.zeros((1, 1)), dtype=tf.float32)
                for _ in self.W_tasks[task_id]] for task_id in range(self.n_tasks)]
        
        # Define the neural networks.
        self.y_preds = define_mtn(
            self.x_tf, self.y_dims, self.W_shared, self.b_shared,
            self.W_tasks, self.b_tasks, self.keep_prob_tf, self.phase_tf, batch_norm)

        # Define the mutli-task loss and training ops.
        self.losses_tf = [tf.losses.mean_squared_error(self.ys_tf[task_id],
            self.y_preds[task_id]) for task_id in range(self.n_tasks)]
        self.train_ops_tf = [tf.train.AdamOptimizer(self.lr_tf).minimize(
            self.losses_tf[task_id]) for task_id in range(self.n_tasks)]

        # Define the data scaler.
        self.scalers_x = [StandardScaler() for _ in range(self.n_tasks)]
        self.scalers_y = [StandardScaler() for _ in range(self.n_tasks)]

        # Define the saver object for model persistence.
        self.tmpfile = NamedTemporaryFile()
        self.saver = tf.train.Saver(max_to_keep=1)

        # Define the Tensorflow session, and its initializer op.
        self.sess = tf.Session()
        #writer = tf.summary.FileWriter('logs', self.sess.graph)
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

    def predict(self,
            x: np.ndarray,
            task_ids: np.ndarray) -> np.ndarray:
        """ Predict the expected value of y given x for specified tasks.

        Args:
            x (n_samples, x_dim): Input data.
            task_ids (n_samples,): Task id for each x.

        Returns:
            y (n_samples, y_dim[i]): Predicted output.
        """
        try:
            x = rescale(x, self.scalers_x, task_ids, direction=0)
        except NotFittedError:
            print('Warning: scalers are not fitted.')
        y_pred = self.sess.run(self.y_preds,
                {self.x_tf: x, self.keep_prob_tf: 1., self.phase_tf: False})
        # Extract outputs for each task.
        y_pred = [y_pred[task_id][data_id] for (data_id, task_id) in
                enumerate(task_ids)]
        y_pred = rescale(y_pred, self.scalers_y, task_ids, direction=1)
        return y_pred

    def fit(self,
            x: List[(np.ndarray)],
            y: List[(np.ndarray)],
            mtn_verbose: bool = False,
            lr: float = 1e-3,
            max_epochs: int = 1000,
            min_epochs: int = 10,
            batch_size: int = 32,
            max_nonimprovs: int = 30,
            dropout_keep_prob: float = 1.) -> (np.ndarray, np.ndarray):
        """ Train a Multi-Task Network.

        Args:
            x [(n_samples, x_dim)]: Input data.
            y [(n_samples, y_dim)]: Output data.
            mtn_verbose: If True, print out progress.
            lr: Learning rate.
            max_epochs: Max number of training epochs. Each epoch goes
                through the whole dataset (possibly leaving out
                mod(x.shape[0], batch_size) data).
            min_epochs: Do at least this many epochs.
            batch_size: Training batch size.
            max_nonimprovs: Number of epochs allowed without improving
                the validation score before quitting.
            dropout_keep_prob: Probability of keeping a unit on in Dropout.

        Returns:
            tr_losses (num_epochs,): Training errors (zero-padded).
            val_losses (num_epochs,): Validation errors (zero-padded).

        Each element of the input/output lists corresponds to one
        task. Thus, len(x) = len(y) must equal self.n_tasks. The 
        number of datapoints in each task can differ.
        """
        if len(x) != len(y) or len(x) != self.n_tasks:
            raise ValueError('Number of tasks not consistent with data.')
        # Split data into a training and validation set.
        n_samples = [x_task.shape[0] for x_task in x]
        n_val = [int(ns * .1) for ns in n_samples]
        ids_perm = [np.random.permutation(ns) for ns in n_samples]
        self.valid_ids = [ids_perm[task_id][:n_val[task_id]] for
                task_id in range(self.n_tasks)]

        # Split data into validation and training.
        x_tr = [x[task_id][ids_perm[task_id][n_val[task_id]:]]
                for task_id in range(self.n_tasks)]
        y_tr = [y[task_id][ids_perm[task_id][n_val[task_id]:]]
                for task_id in range(self.n_tasks)]
        x_val = [x[task_id][self.valid_ids[task_id]]
                for task_id in range(self.n_tasks)]
        y_val = [y[task_id][self.valid_ids[task_id]]
                for task_id in range(self.n_tasks)]

        # Fit scalers on the training data and rescale training data.
        x_tr = [self.scalers_x[task_id].fit_transform(x_tr[task_id])
                for task_id in range(self.n_tasks)]
        y_tr = [self.scalers_y[task_id].fit_transform(y_tr[task_id])
                for task_id in range(self.n_tasks)]

        # Rescale the validation data.
        x_val = [self.scalers_x[task_id].transform(x_val[task_id])
                for task_id in range(self.n_tasks)]
        y_val = [self.scalers_y[task_id].transform(y_val[task_id])
                for task_id in range(self.n_tasks)]

        # Train the neural net.
        tr_losses = np.zeros((max_epochs, self.n_tasks))
        val_losses = np.zeros((max_epochs, self.n_tasks))
        best_val = np.inf
        start_time = time.time()
        for epoch_id in range(max_epochs):
            # On each epoch, do a batch of each task.
            ps = val_losses[epoch_id-1]
            for task_id in np.random.permutation(np.arange(n_tasks)):
                for _ in range(10):
                    batch_ids = np.random.choice(n_samples[task_id]
                            -n_val[task_id], batch_size, replace=False)
                    tr_loss = self.sess.run(
                        self.losses_tf[task_id],
                        {self.x_tf: x_tr[task_id][batch_ids],
                         self.ys_tf[task_id]: y_tr[task_id][batch_ids],
                         self.keep_prob_tf: 1.,
                         self.phase_tf: False})
                    self.sess.run(self.train_ops_tf[task_id],
                                  {self.x_tf: x_tr[task_id][batch_ids],
                                   self.ys_tf[task_id]: y_tr[task_id][batch_ids],
                                   self.keep_prob_tf: dropout_keep_prob,
                                   self.phase_tf: True,
                                   self.lr_tf: lr})
                    val_loss = self.sess.run(self.losses_tf[task_id],
                                             {self.x_tf: x_val[task_id],
                                              self.ys_tf[task_id]: y_val[task_id],
                                              self.keep_prob_tf: 1.,
                                              self.phase_tf: False})
                    tr_losses[epoch_id][task_id] = tr_loss
                    val_losses[epoch_id][task_id] = val_loss

            # Average losses of the tasks.
            if np.mean(val_losses[epoch_id]) < best_val:
                last_improved = epoch_id
                best_val = np.mean(val_losses[epoch_id])
                model_path = self.saver.save(
                    self.sess, self.tmpfile.name)

            tr_time = time.time() - start_time
            if mtn_verbose:
                sys.stdout.write(('\nTraining epoch {}, time {}s.'
                    '\nTr loss {:.4g} [{}].'
                    '\nVal loss {:.4g} [{}], best val {:.4g}.'
                    ).format(epoch_id, int(tr_time),
                    tr_losses[epoch_id].mean(), tr_losses[epoch_id],
                    val_losses[epoch_id].mean(), val_losses[epoch_id], best_val))
                sys.stdout.flush()
            # Finish training if:
            #   1) min_epochs are done, and
            #   2a) either we're out of time, or
            #   2b) there was no validation score
            #       improvement for max_nonimprovs epochs.
            if (epoch_id >= min_epochs or epoch_id
                    - last_improved > max_nonimprovs):
                    break

        self.saver.restore(self.sess, model_path)
        if mtn_verbose:
            print('Trainig done in {} epochs, {}s. Total validation loss {:.4g}.'.format(
                epoch_id, tr_time, best_val))
        return tr_losses, val_losses

def make_data(n_samples=100, dim=100):
    projection_mat = np.random.uniform(low=-1, high=1, size=(2, dim))
    while True:
        # Sample points on a circle and project up.
        x_circ = np.random.uniform(low=0, high=2*np.pi, size=(n_samples, 1))
        X = np.hstack([np.cos(x_circ), np.sin(x_circ)])
        X = np.matmul(X, projection_mat)
        X = np.sin(np.log(np.abs(X))**2)
        # Outputs are a randomized, noisy function on the circle.
        Y = x_circ + np.random.randn(n_samples, 1) * .1
        a = np.random.choice(np.arange(-4, 4))
        b = np.random.uniform(low=-10, high=10)
        c = np.random.uniform(low=-10, high=10)
        Y = Y ** a * np.sin(b * np.abs(Y)) + c * Y
        yield (X, Y)


if __name__=="__main__":
    """ Check that everything works as expected. Should take several
    seconds on a CPU machine. """
    import numpy as np
    from neural_networks import nn
    import matplotlib.pyplot as plt
    
    n_tasks = 100
    dim = 10
    samples = 50
    n_epochs = 100
    # Make data: X is a list of datasets, each with the same coordinates
    # but potentially 1) different n_samples and 2) different output tasks in Y.
    X, Y = zip(*[next(make_data(samples, dim)) for _ in range(n_tasks)])

    # Train a multi-task network.
    mtnet = MTN(x_dim=X[0].shape[1], y_dims=[1] * n_tasks,
            shared_arch=[32]*5, task_arch=[32]*5, batch_norm=False)
    mtnet.fit(X, Y, mtn_verbose=True, lr=1e-3, dropout_keep_prob=1.,
            min_epochs=n_epochs * n_tasks, max_epochs=n_epochs * n_tasks)# max_nonimprovs=n_epochs)

    # Train a single nn for each task, for comparison.
    nnets = [nn.NN(x_dim=X[0].shape[1], y_dim=1, arch=[128]*2) for _ in range(n_tasks)]
    for task_id in range(n_tasks):
        nnets[task_id].fit(X[task_id], Y[task_id], nn_verbose=True, lr=1e-4,
                min_epochs=n_epochs, max_epochs=n_epochs)# max_nonimprovs=n_epochs)

    # Compare the results.
    errs_mtn = []
    errs_nn = []
    for task_id in range(n_tasks):
        mtnpred = mtnet.predict(X[task_id],
                task_ids=np.array([task_id] * X[task_id].shape[0]))
        nnpred = nnets[task_id].predict(X[task_id])
        errs_mtn.append(np.mean((mtnpred - Y[task_id])**2))
        errs_nn.append(np.mean((nnpred - Y[task_id])**2))

    fig = plt.figure(figsize=(5, 5))
    plt.xlabel('Task ID')
    plt.ylabel('Error')
    plt.title('MTN error to NN error ratio')
    #plt.bar(np.arange(n_tasks), np.array(errs_mtn))
    plt.bar(np.arange(n_tasks), np.array(errs_mtn) / np.array(errs_nn), width=.8)
    plt.axhline(y=1)
    plt.show()
