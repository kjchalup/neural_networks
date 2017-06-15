Neural nets in `TensorFlow`_.
##############################

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

`TensorFlow`_ is a low-level framework for building and executing
computational graphs. There are now numerous
higher-level frameworks built on top of `TensorFlow`_ that implement neural
networks (e.g. `Keras`_, now officially supported by `TensorFlow`_).

This repository implements one such high-level framework.
In research, I need low-level control over the computational graph,
but I also find I often re-use basic neural net architectures. This makes it
inconvenient to use pre-existing frameworks (lack of low-level control) --
but I don't want to re-implement basic nn components each time.

Why would anyone use this repository?

    * You are me, and use it for doing research.
    * You want a unified access to high-level nn implementations that are not easily found elsewhere (e.g. highway nets, GANs etc).

Usage
-----
I tried to mimic the `scikit-learn`_ interface. You fit a network
using nn.fit, and predict with nn.predict methods. In some cases
there are other useful methods, e.g. GANs (see below) have a gan.sample()
method.
See individual module documentation for more details.

Each module's usage is exemplified in its __main__ part.
For example, `nn.py`_ contains the following section:

.. code-block:: python

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
    nn = NN(x_dim=x.shape[1], y_dim=y.shape[1])

    # Create a tensorflow session and run the net inside.
    with tf.Session() as sess:
        # Define the Tensorflow session, and its initializer op.
        sess.run(tf.global_variables_initializer())

        # Use a writer object for Tensorboard visualization.
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/')
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

This trains a neural net with the default settings. The output should look something like this:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/example_output.png
        :alt: Example NN training output.
        :align: center
        
The code above also saves training info into the logs/ directory. You can
then use `Tensorboard`_ to visualize the network graph and training stats.
In this case, the default network has ten layers with 32 units each, as shown
in the graph:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/example_graph.png
        :alt: Example NN graph.
        :align: center
        
Finally, the training and validation loss progress looked like this:

    .. image:: https://github.com/kjchalup/neural_networks/blob/master/example_training.png
        :alt: Example NN training.
        :align: center

Implemented Methods
-------------------
At the moment, the reposity contains the following methods:
  
  * `nn.py`_: Multi-layer perceptron (MLP) with Dropout (`arXiv:1207.0580`_).
  * `nn.py`_: Residual Network (`arXiv:1512.03385`_).
  * `nn.py`_: Highway Network (`arXiv:1505.00387`_).
  * `gan.py`_: Least-Squares Generative Adversarial Network (`arXiv:1611.04076v2`_, in my experience the best GAN, though doesn't a convergence criterion like Wasserstein GANs).
  * `mtn.py`_: Multi-Task Networks (my own creation) -- learn from multiple datasets with related inputs but different output tasks.

Requirements
------------
Everything should work with Python 2 and 3.

    * `NumPy`_ >= 1.12
    * `TensorFlow`_ >= 1.0.0
   
.. _numpy: http://www.numpy.org/
.. _scikit-learn: http://scikit-learn.org/
.. _TensorFlow: https://www.tensorflow.org/
.. _nn.py: nn.py
.. _mtn.py: mtn.py
.. _gan.py: gan.py
.. _arXiv:1207.0580: https://arxiv.org/pdf/1207.0580.pdf)
.. _arXiv:1512.03385: https://arxiv.org/pdf/1512.03385.pdf
.. _arXiv:1505.00387: https://arxiv.org/pdf/1505.00387.pdf
.. _arXiv:1611.04076v2: https://arxiv.org/abs/1611.04076v2
