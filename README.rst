
Benchmark for deep inverse problems
===================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of the deepinv library solving imaging inverse problems of the form


.. math::

    y = Ax+e


where $A$ is a linear operator, $x$ is an image to recover, $y$ is the observed data and $e$ is the realization of
random noise; the goal is to recover $x$ from $y$.

This benchmark is based on the deepinv library, which is a library of solvers and linear operators $A$
for inverse problems in imaging.

Currently, the following solvers are implemented in this benchmark:
- DPIR (Deep Plug-and-Play Priors)
- Wavelet
- Wavelet dictionary

The following inverse problems are also implemented in this benchmark:
- Image deconvolution
- MRI reconstruction
- Non-cartesian MRI (work in progress)


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/matthieutrs/benchmark_deepinv
   $ benchopt run benchmark_deepinv


.. note::

    The interface between this benchmark and deepinv is still under development. In order to use it, you need to install the deepinv library from the `mri_nc` branch.



The methods currently implemented can be solved using the following command:

.. code-block::

	$ benchopt run -d fastMRI -f wavelets -f DPIR -f wavelet_dict


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/matthieutrs/benchmark_deepinv/workflows/Tests/badge.svg
   :target: https://github.com/matthieutrs/benchmark_deepinv/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
