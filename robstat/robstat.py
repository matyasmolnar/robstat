"""Module with all sorts out robust statistics useful for data science"""


import io
import functools
from contextlib import redirect_stdout

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# attempt to import the JAX library for acceleration
# below allows a one-line switch to JAX
try:
    pJAX = True
    from jax.config import config
    config.update('jax_enable_x64', True)
    import jax
    from jax import numpy as jnp
    from jax import jit as JJ
    from jax.scipy.optimize import minimize as jminimize
except ImportError:
    pJAX = False
    jnp = np
    JJ = lambda x: x
    jminimize = minimize

# Tukey median package from R
try:
    pR = True
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    TukeyRegion = importr('TukeyRegion')
    depth = importr('depth')
    pR = True
except ImportError:
    pR = False

from . import utils


def circ_mean_dev(angles, angle_est, weights=None):
    """
    Circular mean deviation (Fisher, 1993, p. 35-36)

    Args:
        angles (ndarray): Angles in radians.
        angle_est (float, int): Angle estimate.

    Returns:
        Circular mean deviation (float)
    """
    if weights is None:
        return jnp.pi - 1 / angles.size * jnp.abs(jnp.pi - jnp.abs(angles - \
            angle_est)).sum()
    else:
        return jnp.pi - 1 / weights.sum() * (weights * jnp.abs(jnp.pi - \
            jnp.abs(angles - angle_est))).sum()


def circ_med_dev(angles, angle_est):
    """
    Circular median deviation

    Args:
        angles (ndarray): Angles in radians.
        angle_est (float, int): Angle estimate.

    Returns:
        Circular median deviation (float)
    """
    return jnp.med(jnp.pi - jnp.abs(jnp.pi - jnp.abs(angles - angle_est)))


def mardia_median(angles, weights=None, init_guess=None):
    """
    Minimize the circular median deviation to obtain the "Mardia Median"
    (1972, p. 28,31). Note that this angle is not necessarily unique.

    Args:
        angles (ndarray): Angles in radians.
        weights (ndarray): array of weights associated with the values in data.
        init_guess (float): initial guess for the Mardia median.

    Returns:
        Mardia median (float).
    """
    if init_guess is None:
        init_guess = jnp.array([stats.circmean(angles)])
    else:
        if not isinstance(init_guess, (np.ndarray, jnp.ndarray)):
            init_guess = np.array([init_guess])

    # reorder arguments for partial fill
    _cmd = lambda a, w, i : circ_mean_dev(a, i, weights=w)
    ff = JJ(functools.partial(_cmd, angles, weights))

    res = jminimize(ff, init_guess, method='bfgs', options={'maxiter':3000}, \
                    tol=1e-8)
    if pJAX:
        res = res._asdict()
    return res['x'].item()


def geometric_median(data, weights=None, init_guess=None):
    """
    Geometric median. Also known as the L1-median.

    JAX acceleration not yet available for scipy.spatial.distance.

    TODO: rewrite cdist in euclidian space for this module.

    Args:
        data (ndarray): n-dimensional data.
        weights (ndarray): array of weights associated with the values in data.
        init_guess (ndarray): initial guess for the geometric median.

    Returns:
        Geometric median (ndarray).
    """
    if np.iscomplexobj(data):
        Cdata = True
        data = utils.decomposeCArray(data)
    else:
        Cdata = False

    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    if init_guess is None:
        init_guess = np.zeros(data.shape[1])

    def agg_dist(weights, x):
        """
        Aggregate distance objective function for minimizer.

        Note that parameters initial guess must be a single element ndarray.
        """
        if weights is None:
            return cdist(x * np.ones_like(data), data, metric='euclidean')[0, :].sum()
        else:
            return (weights*cdist(x * np.ones_like(data), data, metric='euclidean')[0, :]).sum()

    ff = functools.partial(agg_dist, weights)

    resx = minimize(ff, init_guess, method='bfgs', options={'maxiter':3000}, \
                    tol=1e-8)['x']
    if Cdata:
        resx = resx[0] + resx[1]*1j

    return resx


def tukey_median(data, weights=None):
    """
    Tukey median calculated using the TukeyRegion R package.

    Requires R to be installed, as well as the rpy2 Python-R bridge.

    Full documentation:
    https://cran.r-project.org/web/packages/TukeyRegion/TukeyRegion.pdf

    Args:
        data (ndarray): n-dimensional data.
        weights (ndarray): array of integer weights associated with the values in data.

    Returns:
        Tukey median (ndarray).
    """
    if TukeyRegion is None:
        print('TukeyRegion R package not installed.')
        return None
    else:
        rpy2.robjects.numpy2ri.activate()
        stdout = io.StringIO()

        # repeat entries by weights
        if weights is not None:
            assert weights.dtype == int, "Weights must be integers"
            data = np.repeat(data, weights, axis=0)

        with redirect_stdout(stdout): # suppress output
            TR_res = TukeyRegion.TukeyMedian(data)

        return dict(TR_res.items())


def mv_median(data, method, weights=None, approx=False, eps=1e-8):
    """
    Multivariate median using the 'depth' package in R.

    Requires R to be installed, as well as the rpy2 Python-R bridge.

    Full documentation:
    https://cran.r-project.org/web/packages/depth/depth.pdf

    Args:
        data (ndarray): n-dimensional data.
        method (str): determines the depth function used (e.g. 'Tukey',
        'Oja', 'Spatial')
        weights (ndarray): array of integer weights associated with the values in data.
        approx (bool): should an approximate Tukey median be computed? Useful in
        dimension 2 only when sample size is large.
        eps (float): error tolerance to control the calculation.

    Returns:
        Multivariate median (ndarray).
    """
    if not pR:
        print('depth R package not installed.')
        return None
    else:
        rpy2.robjects.numpy2ri.activate()
        stdout = io.StringIO()

        # repeat entries by weights
        if weights is not None:
            assert weights.dtype == int, "Weights must be integers"
            data = np.repeat(data, weights, axis=0)

        with redirect_stdout(stdout): # suppress output
            TR_res = depth.med(data, method=method)

        return dict(TR_res.items())


def install_R_packages():
    """
    Installs the TukeyRegion and depth R package.

    https://cran.r-project.org/web/packages/TukeyRegion/
    https://cran.r-project.org/web/packages/depth/
    """
    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('TukeyRegion')
    utils.install_packages('depth')
