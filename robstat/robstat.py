"""Module with all sorts out robust statistics useful for data science"""


import io
import functools
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# # attempt to import the JAX library for acceleration
# # below allows a one-line switch to JAX
# try:
#     pJAX = True
#     from jax.config import config
#     config.update('jax_enable_x64', True)
#     import jax
#     from jax import numpy as jnp
#     from jax import jit as JJ
#     from jax.scipy.optimize import minimize as jminimize
# except ImportError:
#     pJAX = False
#     jnp = np
#     JJ = lambda x: x
#     jminimize = minimize

# JAX slows down minimization for geometric_median
# investigate and use normal scipy for time being
pJAX = False
jnp = np
JJ = lambda x: x
jminimize = minimize

# Tukey median package from R
try:
    pR = True
    from rpy2.robjects import numpy2ri, pandas2ri
    numpy2ri.activate()
    pandas2ri.activate()
    from rpy2.robjects.packages import importr
    TukeyRegion = importr('TukeyRegion')
    depth = importr('depth')
    MVN = importr('MVN')
    pR = True
except ImportError:
    pR = False

from . import utils


def omit_nans(data, weights):
    """
    Remove nans from array

    Args:
        data (ndarray): input data.
        weights (ndarray): array of weights associated with the values in data.

    Returns:
        Filtered data and weights arrays (ndarray)
    """
    if data.ndim <= 2:
        if data.ndim == 1:
            nan_idx = ~np.isnan(data)
            data = data[nan_idx]
        if data.ndim == 2:
            nan_idx = ~np.isnan(data).any(axis=1)
            data = data[nan_idx, :]
        if weights is not None:
            weights = weights[nan_idx]
    else:
        raise ValueError('data must be either a 1D or 2D ndarray.')
    return data, weights


def circ_mean_dev(angles, angle_est, weights=None):
    """
    Circular mean deviation (Fisher, 1993, p. 35-36)

    Args:
        angles (ndarray): angles in radians.
        angle_est (float, int): angle estimate.

    Returns:
        Circular mean deviation (float)
    """
    if np.isnan(angles).any():
        angles, weights = omit_nans(angles, weights)

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
        angles (ndarray): angles in radians.
        angle_est (float, int): angle estimate.

    Returns:
        Circular median deviation (float)
    """
    if np.isnan(angles).any():
        angles = angles[~np.isnan(angles)]
    return jnp.med(jnp.pi - jnp.abs(jnp.pi - jnp.abs(angles - angle_est)))


def mardia_median(angles, weights=None, init_guess=None):
    """
    Minimize the circular median deviation to obtain the "Mardia Median"
    (1972, p. 28,31). Note that this angle is not necessarily unique.

    Args:
        angles (ndarray): angles in radians.
        weights (ndarray): array of weights associated with the values in data.
        init_guess (float): initial guess for the Mardia median.

    Returns:
        Mardia median (float).
    """
    if np.isnan(angles).any():
        angles, weights = omit_nans(angles, weights)

    if init_guess is None:
        init_guess = jnp.array([stats.circmean(angles)])
    else:
        if not isinstance(init_guess, (np.ndarray, jnp.ndarray)):
            init_guess = np.array([init_guess])

    # wrap init_guess angle between -pi and pi
    if np.abs(init_guess) >= np.pi:
        init_guess = (init_guess + np.pi) % (2 * np.pi) - np.pi

    # reorder arguments for partial fill
    _cmd = lambda a, w, i: circ_mean_dev(a, i, weights=w)
    ff = JJ(functools.partial(_cmd, angles, weights))

    res = jminimize(ff, init_guess, method='bfgs', options={'maxiter':3000}, \
                    tol=1e-8)
    if pJAX:
        res = res._asdict()
    return res['x'].item()


def c_mardia_median(Carr, weights=None, init_guess=None):
    """
    Median estimate of a complex number, with the angle calculated using the Mardia
    median and the absolute value calculated separately using median.

    Illustrative function=, not to be used for data analysis.

    Args:
        Carr (ndarray): complex array.
        weights (ndarray): array of weights associated with the values in Carr.
        init_guess (float): initial guess for the Mardia median (complex tuple).
        Only the angle part will be used.

    Returns:
        Median estimate (complex tuple).
    """
    if init_guess is not None:
        init_guess = np.angle(init_guess) # only use phase
    med_vis_amp = np.nanmedian(np.abs(Carr))
    mmed_vis_phase = mardia_median(np.angle(Carr), weights=weights, \
                                   init_guess=init_guess)
    return med_vis_amp * np.exp(1j * mmed_vis_phase)


def cdist_jax(arr_A, arr_B):
    """
    Euclidean distance written with numpy functions - can be accelerated in JAX,
    while scipy.spatial.distance.cdist cannot be (as of yet).

    Args:
        arr_A (ndarray): data array A.
        arr_B (ndarray): data array B.
        weights (ndarray): array of weights associated with the values in data.
        init_guess (ndarray): initial guess for the geometric median.

    Returns:
        Euclidean distances between the input arrays (ndarray).
    """
    return jnp.sqrt(jnp.sum(jnp.square(arr_A - arr_B), axis=1))


def geometric_median(data, weights=None, init_guess=None, \
                     method='minimize', options={}, verbose=False):
    """
    Geometric median. Also known as the L1-median.

    JAX acceleration not yet available for scipy.spatial.distance.

    Args:
        data (ndarray): n-dimensional data. Array must be coordinates of ndim = 2,
        with shape [# data points, n-coordinates].
        weights (ndarray): array of weights associated with the values in data.
        init_guess (ndarray): initial guess for the geometric median. Can specify
        "median" or "mean" to choose those as starting points.
        method (str): method to use for computation - {"minimize", "weiszfeld"}
        options (dict): options for method used. possible kwargs:
            tol (float): tolerance used for minimization.
            maxiter (int): maximum number of iterations.
            min_method (str): minimization method.
            keep_res (bool): keep result from unsuccesful minimization instead of
            replacing with nan.
        verbose (bool): status updates of geometric median computation.

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
    elif init_guess == 'median':
        init_guess = np.nanmedian(data, axis=0)
    elif init_guess == 'mean':
        init_guess = np.nanmean(data, axis=0)
    # init_guess could be nan if it is reused in a loop
    elif np.isnan(init_guess):
        init_guess = np.zeros(data.shape[1])

    if np.iscomplexobj(init_guess):
        init_guess = np.array([init_guess.real, init_guess.imag])

    # remove rows with nan coordinates
    if np.isnan(data).any():
        data, weights = omit_nans(data, weights)

    if not pJAX:
        eucl_dist = functools.partial(lambda m, XA, XB: cdist(XA, XB, metric=m), \
                                      'euclidean')
    else:
        eucl_dist = cdist_jax

    def agg_dist(weights, x):
        """
        Aggregate distance objective function for minimizer.

        Note that parameters initial guess must be a single element ndarray.
        """
        ed = np.squeeze(eucl_dist(x[np.newaxis, :], data))
        if weights is not None:
            ed *= weights
        return ed

    method_err = 'specify valid method: must be either "minimize", '\
                 '"weiszfeld" or "modified_weiszfeld".'

    if method == 'minimize':
        # Solve as a convex optimization problem
        default_options = {'maxiter': 3000, 'tol': 1e-3, 'keep_res':False, \
                           'min_method': 'bfgs'}
        default_options.update(options)
        opts = default_options

        ff = lambda x: JJ(functools.partial(agg_dist, weights))(x).sum()
        res = jminimize(ff, np.array(init_guess, ndmin=1), method=opts['min_method'], \
                        options={'maxiter':opts['maxiter']}, tol=opts['tol'])

        if pJAX:
            res = res._asdict()
            success = res['success'].item()
            utils.echo(res['status'], verbose=verbose)
        else:
            success = res['success']
            utils.echo(res['message'], verbose=verbose)

        if not success and not opts['keep_res']:
            utils.echo('Minimization unsuccesful - nan returned', \
                       verbose=verbose)
            return np.nan
        else:
            resx = res['x']

    elif 'weiszfeld' in method:
        default_options = {'maxiter': 1000, 'tol': 1e-7}
        default_options.update(options)
        opts = default_options

        if weights is None:
            weights = np.ones(data.shape[0])

        def ff(x, w):
            distances = JJ(functools.partial(agg_dist, None))(x)
            # catch divide by zero that causes algorithm to get stuck
            # displace problematic distance slightly
            distances = np.where(distances == 0, 1, distances)
            return w / distances

        iters = 0
        guess = init_guess

        if method == 'weiszfeld':
            # Weiszfeld's algorithm of iteratively re-weighted least squares
            while iters < opts['maxiter']:
                w_div_d = ff(guess, weights) # weights devided by distances
                guess_next = (data.T*w_div_d).sum(axis=1) / (w_div_d).sum()
                guess_movement = np.sqrt(((guess - guess_next)**2).sum())
                guess = guess_next
                if guess_movement <= opts['tol']:
                    break
                iters += 1

            resx = guess

        elif method == 'modified_weiszfeld':
            # Modified Weiszfeld's algorithm of iteratively re-weighted least squares
            # https://www.pnas.org/content/97/4/1423
            while iters < opts['maxiter']:
                eq_idx = np.equal(data, guess).all(axis=1)
                if eq_idx.any():
                    _weights = np.where(eq_idx, 0, weights)
                    w_div_d = ff(guess, _weights)
                    T = (data.T*w_div_d).sum(axis=1) / (w_div_d).sum()
                    R = (T - guess) * (w_div_d).sum()
                    r = np.sqrt(np.sum(np.square(R)))
                    r = np.where(r == 0, r, 1) # replace zeros with ones s.t. "0/0 = 0" below
                    mod_weight = weights[eq_idx.nonzero()[0]]
                    w_div_r = mod_weight / r
                    guess_next = (1 - w_div_r)*T + min(1, w_div_r)*guess
                else:
                    w_div_d = ff(guess, weights) # weights devided by distances
                    guess_next = (data.T*w_div_d).sum(axis=1) / (w_div_d).sum()
                guess_movement = np.sqrt(((guess - guess_next)**2).sum())
                guess = guess_next

                if guess_movement <= opts['tol']:
                    break

                iters += 1

            resx = guess

        else:
            raise ValueError(method_err)

    else:
        raise ValueError(method_err)

    if Cdata:
        resx = resx[0] + 1j*resx[1]

    return resx


def tukey_median(data, weights=None, verbose=False):
    """
    Tukey median calculated using the TukeyRegion R package.

    Requires R to be installed, as well as the rpy2 Python-R bridge.

    Full documentation:
    https://cran.r-project.org/web/packages/TukeyRegion/TukeyRegion.pdf

    Args:
        data (ndarray): n-dimensional data.
        weights (ndarray): array of integer weights associated with the values in data.
        verbose (bool): status updates of tukey median computation.

    Returns:
        Tukey median (ndarray).
    """
    if not pR:
        print('rpy2 Python-R bridge not installed and/or TukeyRegion R package not installed.')
        return None
    else:
        stdout = io.StringIO()

        # remove rows with nan coordinates
        if np.isnan(data).any():
            data, weights = omit_nans(data, weights)

        null_res = {'depth': np.nan, 'innerPointFound': False, 'barycenter': np.nan}

        if data.size == 0:
            utils.echo('No non-nan input data; Tukey median cannot be returned.', \
                       verbose=verbose)
            return null_res

        # separate to matrix if data is is complex
        if np.iscomplexobj(data):
            Cdata = True
            data = utils.decomposeCArray(data)
        else:
            Cdata = False

        if data.shape[0] <= data.shape[1]:
            utils.echo('Input data should be a matrix with at least d = 2 columns '\
                       'and at least d + 1 rows', verbose=verbose)
            return null_res

        # repeat entries by weights
        if weights is not None:
            assert weights.dtype == int, 'Weights must be integers'
            data = np.repeat(data, weights, axis=0)

        with redirect_stdout(stdout): # suppress output
            TR_res = TukeyRegion.TukeyMedian(data)

        res = dict(TR_res.items())

        if not res['innerPointFound'][0]:
            res['barycenter'] = np.nan
            utils.echo('Inner point of the region has not been found; nan returned '\
                       'for the barycenter.', verbose=verbose)
        else:
            if Cdata:
                bcenter = res['barycenter']
                res['barycenter'] = bcenter[0] + bcenter[1]*1j

        return res


def mv_median(data, method, weights=None, approx=False, eps=1e-8):
    """
    Multivariate median using the depth package in R.

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
        print('rpy2 Python-R bridge not installed and/or depth R package not installed.')
        return None
    else:
        stdout = io.StringIO()

        # remove rows with nan coordinates
        if np.isnan(data).any():
            data, weights = omit_nans(data, weights)

        # repeat entries by weights
        if weights is not None:
            assert weights.dtype == int, 'Weights must be integers'
            data = np.repeat(data, weights, axis=0)

        with redirect_stdout(stdout): # suppress output
            TR_res = depth.med(data, method=method)

        return dict(TR_res.items())


def mv_normality(data, method='hz', verbose=False):
    """
    Assess multivariate normality using the mvn function from the MVN R package.

    Requires R to be installed, as well as the rpy2 Python-R bridge.

    See multivariate_normality from pingouin package for a Python implementation.
    https://pingouin-stats.org/index.html

    Full documentation:
    https://cran.r-project.org/web/packages/MVN/MVN.pdf

    Args:
        data (ndarray): n-dimensional data.
        method (str): MVN test method ('hz', 'royston', 'dh', 'energy').
        verbose (bool): status updates of MVN computation.

    Returns:
        Multivariate normality results (dict).
    """
    if not pR:
        print('rpy2 Python-R bridge not installed and/or MVN R package not installed.')
        return None
    else:
        # remove rows with nan coordinates
        if np.isnan(data).any():
            data, _ = omit_nans(data, None)

        stat_dict = {'hz': 'HZ', 'royston': 'H', 'dh': 'E', 'energy': 'Statistic'}

        null_res = {'Test': np.nan, stat_dict[method]: np.nan, 'p value': np.nan, \
                    'MVN': np.nan}

        if data.size == 0:
            utils.echo('No non-nan input data; normality test result cannot be returned.', \
                       verbose=verbose)
            return null_res

        if data.shape[0] <= 7:
            utils.echo('Sample size must be greater than 7; normality test result '\
                       'cannot be returned.', verbose=verbose)
            return null_res

        # separate to matrix if data is is complex
        if np.iscomplexobj(data):
            data = utils.decomposeCArray(data)

        MVN_res = MVN.mvn(data=data, mvnTest=method).rx2('multivariateNormality')
        return dict(zip(MVN_res.columns, MVN_res.values[0]))


def mv_outlier(data, method='quan', chi2_quantile=0.975, verbose=False):
    """
    Robust multivariate outlier detection using robust Mahalanobis distances,
    using the mvOutlier function from the MVN R package.

    Two approaches are available, one based on Mahalanobis distances ('quan')
    and adjusted Mahalanobis distance ('adj').

    Requires R to be installed, as well as the rpy2 Python-R bridge.

    Full documentation:
    https://cran.r-project.org/web/packages/MVN/MVN.pdf

    Args:
        data (ndarray): n-dimensional data.
        method (str): outlier detection approach('quan', 'adj').
        chi2_quantile (float): chi^2 quantile threshold to declare outliers.
        verbose (bool): status updates of outlier computation.

    Returns:
        Multivariate outlier detection results (dict).
    """
    if not pR:
        print('rpy2 Python-R bridge not installed and/or MVN R package not installed.')
        return None
    else:
        # separate to matrix if data is is complex
        if np.iscomplexobj(data):
            data = utils.decomposeCArray(data)

        df = pd.DataFrame(data)
        df.dropna(axis=0, how='any', inplace=True)

        df = MVN.mvOutlier(df, method=method, qqplot=False).rx2('outlier')

        # reformat cols
        df.index = pd.to_numeric(df.index)
        df.rename(columns={'Mahalanobis Distance': 'RS Mahalanobis Distance'}, \
                  inplace=True)
        for col in ['Observation', 'RS Mahalanobis Distance']:
            df[col] = pd.to_numeric(df[col])
        df['Outlier'] = df['Outlier'].map({'FALSE': False, 'TRUE': True})

        if chi2_quantile != 0.975:
            # MVN.mvOutlier takes the classical 0.975 threshold; need to re-declare
            # outliers if this threshold is changed
            chi2_thresh = stats.chi2.ppf(chi2_quantile, data.shape[1])
            df['Outlier'] = df['RS Mahalanobis Distance'] > chi2_thresh

        if np.isnan(data).any():
            # adding back in empty nan rows
            og_size = data.shape[0]
            to_in = np.delete(np.arange(og_size), df['Observation'].values)

            nan_arr = np.empty_like(to_in)*np.nan
            false_arr = np.zeros_like(nan_arr).astype(bool)

            df2 = pd.DataFrame(list(zip(to_in, nan_arr, false_arr)), index=to_in, \
                               columns=df.columns)

            df = pd.concat((df, df2))

        df.sort_index(inplace=True)

        return df


def install_R_packages():
    """
    Installs the TukeyRegion, depth and MVN R packages.

    https://cran.r-project.org/web/packages/TukeyRegion/
    https://cran.r-project.org/web/packages/depth/
    https://cran.r-project.org/web/packages/MVN/
    """
    import rpy2.robjects.packages as rpackages
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)
    utils.install_packages('TukeyRegion')
    utils.install_packages('depth')
    utils.install_packages('MVN')
