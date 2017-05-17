.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: http://www.gnu.org/licenses/gpl-3.0
    :alt: GPL 3.0 License

*Various neural networks implemented in `TensorFlow`_.*

Usage
-----
I tried to mimic the `scikit-learn`_ interface. You fit a network
using nn.fit, and predict with nn.predict methods. In some cases
there are other useful methods, e.g. the Mixture Density Network (`MDN`_)
has an nn.loglik and nn.sample method. See individual module
documentation for more details.

Each module has an example script in its __main__ part that shows
full the range of its uses. For example, `mdn.py`_ contains
the following section:

.. code:: python 

    import numpy as np
    from matplotlib import pyplot as plt 
        
    # Make test data: a noisy mix of functions sampled with different probs.
    x = np.linspace(-1, 1, 10000).reshape(-1, 1)
    y1 = x**2
    y2 = -x/2-1
    y3 = -x-2
    y4 = x-3 
    y = np.hstack([y1, y2, y3, y4])
    ids = np.random.choice(4, y.shape[0], p=[.7, .1, .1, .1])
    y = x + (y[np.arange(y.shape[0]), ids].reshape(-1, 1)
            + np.random.randn(*x.shape) * .1) 
        
    # Fit a mixture-density network. You should see the negative log-likelihood
    # decrease to about zero.
    mdn = MDN(x_dim=x.shape[1], y_dim=y.shape[1], n_comp=10, arch=[32, 32])
    mdn.fit(x, y, nn_verbose=True, lr=1e-2,
            min_epochs=10000, max_epochs=10000, batch_size=128)
        
    # Predict the expected value of Y | X.
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
