.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

* Various flavors of neural networks implemented in `TensorFlow`_.*

Motivation
----------
Core `TensorFlow`_ is a rather low-level framework for building and executing
computational graphs, not strictly for neural networks. There are now numerous
higher-level frameworks built on top of `TensorFlow`_ that implement neural
networks (e.g. `Keras`_, now officially supported by `TensorFlow`_).

This repository implements such a high-level framework for my own research
purposes. In research, I need low-level control over the computational graph,
but I also find I often re-use basic neural net architectures.

I can see two main reasons why anyone would use this repository:
    1) You are me, as I put exactly what I need in here.
    2) There are cutting-edge neural net types not implemented
        elsewhere, but found here (e.g. highway nets, GANs etc).

Usage
-----
I tried to mimic the `scikit-learn`_ interface. You fit a network
using nn.fit, and predict with nn.predict methods. In some cases
there are other useful methods, e.g. GANs (see below) have a gan.sample()
method.
See individual module documentation for more details.

Each module's usage is exemplified in its __main__ part.
For example, `nn.py`_ contains the following section:

.. code:: python 
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

This trains a neural net with 

Implemented Methods
-------------------
At the moment, the reposity contains the following methods:
  
  * `nn.py`_: Multi-layer perceptron (MLP) with Dropout (`arXiv:1207.0580`_).
  * `mdn.py`_: Mixture-Density Network (`MDN`_)

Requirements
------------
To use the nn methods:
    * `NumPy`_ >= 1.12
    * `TensorFlow`_ >= 1.0.0
    * `scikit-learn`_ >= 0.18.1
   
.. _numpy: http://www.numpy.org/
.. _scikit-learn: http://scikit-learn.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _nn.py: nn.py
.. _mdn.py: nn.py
.. _arXiv:1207.0580: https://arxiv.org/pdf/1207.0580.pdf)
.. _MDN: https://publications.aston.ac.uk/373/1/NCRG_94_004.pdf
