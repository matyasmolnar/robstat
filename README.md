# robstat

Robust statistical location estimates for multivariate data.

This package contains a range of functions that provides robust location estimates of directional and high-dimensional data. Applications for the latter include the median of complex quantities, such as radio interferometric visibilities.

[`NumPy`](https://github.com/numpy/numpy) computations and [`SciPy`](https://github.com/scipy/scipy) minimizations are accelerated with the [`JAX`](https://github.com/google/jax) machine learning library.

[R](https://www.r-project.org/) and certain R functions are used in this package, and are run in Python using the [`rpy2`](https://github.com/rpy2/rpy2) R-Python bridge.

## Installation

Preferred installation method is `pip install .` in top-level directory. Alternatively, one can use `python setup.py install`.

### Dependencies

* R to be installed
*  `rpy2` required and to be installed with `pip`

## Included functions

* [L1-median](https://en.wikipedia.org/wiki/Geometric_median) - also known as the geometric median
* Tukey median - as well as other medians that also employ the notion of location depth, e.g. Oja, Spatial (Tukey, 1975)
* Mardia median - for directional data (Mardia, 1972)
