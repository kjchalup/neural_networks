""" Multi-task networks. """
import numpy as np
import tensorflow as tf

from neural_networks import nn
from neural_networks import scalers


class MTN(nn.NN):
    def __init__(self, x_dim, y_dim,
            arch=[128, 128], ntype='plain', **kwargs):
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
        super().__init__(x_dim, y_dim, arch=arch,
                ntype=ntype, **kwargs)

    def define_loss(self):
        x_dim = self.x_tf.get_shape().as_list()[1]
        y_dim = self.y_tf.get_shape().as_list()[1]

        return tf.losses.mean_squared_error(
            self.y_tf, self.y_pred * self.x_tf[:, :y_dim])

    def define_scalers(self):
        xscale = scalers.HalfScaler(ignore_dim=self.y_dim)
        yscale = scalers.StandardScaler()
        return xscale, yscale

def make_data(x, y, n_data, noise_std, degree=3):
    """ Create a 'dataset' outputs: a random polynomial
    over noisy MNIST labels.

    Args:
        x (n_samples, 28**2): MNIST images.
        y (n_samples, 1): MNIST labels.
        n_data (int): Extract a subset of this size.
        noise_std (float): Standard deviation of Gaussian noise
            to be added to the labels.
        degree (int): Degree of the polynomial whose random
            coefficients will define this dataset.

    Returns:
        x, y: The dataset.   
    """
    y_orig = np.array(y)
    while True:
        n_samples = x.shape[0]
        data_ids = np.random.choice(n_samples, n_data, replace=False)
        coeffs = np.random.rand(degree) * 2
        y = np.sum(np.array([coeffs[i] * y_orig**i for i in range(degree)]),
                   axis=0).reshape(-1, 1)
        y += np.random.rand(*y.shape) * noise_std
        yield (x[data_ids], y[data_ids])

def concatenate_tasks(X, Y, task_start, task_end, samples, n_test):
    """ Given a list of X and Y data, extract sublists and concatenate
    into one dataset.

    Args:
        X (List((n_samples, x_dim))): Input data.
        Y (List((n_samples, y_dim))): Output data.
        task_start (int): First dataset to extract.
        task_end (int): Last dataset to extract.
        samples (int): Number of training samples to
            extract from each dataset.
        n_test (int): Number of test samples to extract from each data.

    Returns:
        X_multi, Y_multi: Training data, concatenated datasets between
            task_start and task_end.
        X_test, Y_test: As above, but test data.
    """

    n_tasks = task_end - task_start
    X_multi = np.zeros((samples * n_tasks, n_tasks + dim))
    Y_multi = np.zeros((samples * n_tasks, n_tasks))
    X_test = np.zeros((n_test * n_tasks, n_tasks + dim))
    Y_test = np.zeros((n_test * n_tasks, n_tasks))
    for task_id_id, task_id in enumerate(range(task_start, task_end)):
        X_multi[task_id_id*samples : (task_id_id+1)*samples,
                task_id_id:task_id_id+1] = 1.
        data = np.array(X[task_id])
        X_multi[task_id_id*samples : (task_id_id+1)*samples,
                n_tasks:] = data[:samples]
        Y_multi[task_id_id*samples : (task_id_id+1)*samples,
                task_id_id:task_id_id+1] = Y[task_id_id][:samples]

        X_test[task_id_id*n_test : (task_id_id+1)*n_test,
               task_id_id:task_id_id+1] = 1.
        data = np.array(X[task_id])
        X_test[task_id_id*n_test : (task_id_id+1)*n_test,
               n_tasks:] = data[samples:]
        Y_test[task_id_id*n_test : (task_id_id+1)*n_test,
               task_id_id:task_id+1] = Y[task_id_id][samples:]

    return X_multi, Y_multi, X_test, Y_test


if __name__=="__main__":
    """ Check that everything works as expected. """
    print('===============================================================')
    print('Evaluating MTN. Takes about 30min on a Titan X machine.')
    print('===============================================================')
    import numpy as np
    from tensorflow.examples.tutorials.mnist import input_data
    import matplotlib.pyplot as plt

    # Load MNIST data.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    mY = mnist.train.labels
    mX = mnist.train.images.reshape(mY.shape[0], -1)
    dim = mX.shape[1]

    # Fix task and nn parameters.
    n_task_list = [1, 2, 4, 8, 16, 32]
    max_tasks = max(n_task_list)
    samples = 100
    noise_std = .1
    n_test = 10000
    kwargs = {
              'arch': [32]*30, #[32] * 30,
              'ntype': 'highway',
              'batch_size': 32,
              'lr': 1e-4,
              'valsize': .3,
              'epochs': 10000
             }
    kwargs = {}

    # Make data: X is a list of datasets, each with the same coordinates
    # but potentially 1) different sample sizes and
    # 2) different output tasks in Y.
    np.random.seed(1)
    data = make_data(mX, mY, samples + n_test, noise_std)
    X, Y = zip(*[next(data) for _ in range(max_tasks)])
    errs_mtn = np.zeros((len(n_task_list), max_tasks))

    for n_tasks_id, n_tasks in enumerate(n_task_list):
        print('=' * 70)
        print('Starting {}-split training'.format(n_tasks))
        print('=' * 70)
        # Run multi-task prediction on a group of n_tasks tasks.
        for task_start in range(0, max_tasks, n_tasks):
            print('task_start = {}'.format(task_start))
            X_multi, Y_multi, X_test, Y_test = concatenate_tasks(
                X, Y, task_start, task_start + n_tasks, samples,
                n_test)

            # Create the Tensorflow graph.
            mtnet = MTN(x_dim=X_multi.shape[1], y_dim=n_tasks, **kwargs)
            with tf.Session() as sess:
                # Define the Tensorflow session, and its initializer op.
                sess.run(tf.global_variables_initializer())

                # Fit the net.
                mtnet.fit(X_multi, Y_multi, sess=sess, nn_verbose=True,
                          **kwargs)

                # Run the prediction on the task-group
                for task_id in range(n_tasks):
                    mtnpred = mtnet.predict(
                        X_test[task_id*n_test:(task_id+1)*n_test], sess=sess)
                    mtnpred = mtnpred[:, task_id:task_id+1]
                    errs_mtn[n_tasks_id, task_start+task_id] = (
                        np.sqrt(np.mean((
                            mtnpred - Y_test[task_id*n_test:(task_id+1)*n_test,
                            task_id:task_id+1])**2)))

            # Reset the neural net graph.
            tf.reset_default_graph()
        print('Done\n.')

    vmax = np.max(errs_mtn)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    barw = .8 / len(n_task_list)
    for n_tasks_id in range(len(n_task_list)):
        plt.title('MTN performance as number of tasks grows'.format(
            n_task_list[n_tasks_id]))
        plt.xlabel('task ID')
        plt.ylabel('error')
        plt.bar(np.arange(max_tasks) + n_tasks_id * barw,
                width=barw,
                height=errs_mtn[n_tasks_id],
                label='{}'.format(n_task_list[n_tasks_id]))
        plt.ylim([0, vmax])
    plt.legend(loc=0)

    plt.subplot(2, 1, 2)
    plt.xlabel('n_tasks')
    plt.ylabel('average error')
    plt.plot(n_task_list, errs_mtn.mean(axis=1), 'o')

    plt.savefig('res.png')
