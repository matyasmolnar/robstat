"""Standard statistical techniques"""


import warnings

import numpy as np
from astropy.stats import sigma_clip, mad_std

from . import utils


def mad_clip(data, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False):
    """
    Robust median absolute deviation clipping.

    This function will directly replace flagged and clipped data in array with
    a np.nan.

    Inspired by sigma_clip in HERA hera_cal sigma clipping routine for LST-binning:
    https://github.com/HERA-Team/hera_cal/blob/master/hera_cal/lstbin.py

    Also see scipy.stats.sigmaclip and astropy.stats.sigma_clip.

    Args:
        data (ndarray): input data.
        flags (ndarray): existing boolean flags for data array. True if flagged.
        sigma (float): sigma threshold to cut above.
        axis (int): axis of array to sigma clip.
        min_N (int): minimum length of array to sigma clip, below which no sigma
        clipping is performed.
        verbose (bool): print number of MAD-clipped data points.

    Returns:
        clipped_data (ndarray): clipped data with nans applied
        clip_flags (ndarray): clipped flags
    """
    # ensure array is an array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    sc_data = np.copy(data)

    assert not np.iscomplexobj(data) # MAD clipping erroneous if complex

    # ensure array passes min_N criteria:
    if isinstance(axis, int):
        data_N = data.shape[axis]
    else:
        data_N = np.prod([ax for i, ax in enumerate(data.shape) if i not in axis])
    if data_N < min_N:
        utils.echo('No sigma clipping performed as length of array for specified axis ' \
                   '< min_N.', verbose=verbose)
        if flags is None:
            flags = np.zeros_like(data, dtype=bool)
        return data, flags

    # create empty clip_flags array
    clip_flags = np.zeros_like(data, dtype=bool)

    # inherit flags if fed and apply flags to data
    if flags is not None:
        clip_flags += flags
        sc_data[flags] *= np.nan
    else:
        nan_idxs = np.isnan(sc_data)
        if nan_idxs.any():
            clip_flags += nan_idxs

    flg_count = clip_flags.sum()

    # get robust location
    location = np.nanmedian(data, axis=axis)
    ex_dims = np.ones(data.ndim, dtype=int)
    if not isinstance(axis, int):
        for ax in axis:
            ex_dims[ax] = data.shape[ax]
    else:
        ex_dims[axis] = data.shape[axis]
    tile_loc = np.tile(np.expand_dims(location, axis=axis), ex_dims)

    # get MAD! * 1.482579 correction factor
    scale = np.nanmedian(np.abs(data - tile_loc), axis=axis) * 1.482579
    tile_scale = np.tile(np.expand_dims(scale, axis=axis), ex_dims)

    # get clipped data
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    clip = np.abs(data - tile_loc) / tile_scale > sigma
    np.seterr(**old_settings) # reset to default

    # set clipped data to nan and set clipped flags to True
    sc_data[clip] *= np.nan
    clip_flags[clip] = True
    utils.echo('{} data points MAD-clipped.'.format(clip_flags.sum()-flg_count), \
               verbose=verbose)

    return sc_data, clip_flags


def rsc_avg(data, stat, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False):
    """
    Location of data after robust sigma clipping.

    Mimics HERA LST-binning data averaging.

    Args:
        data (ndarray): input data.
        stat (str): statistic to use for averaging ('mean', 'median').
        flags (ndarray): existing boolean flags for data array. True if flagged.
        sigma (ndarray): sigma threshold to cut above.
        axis (int): axis of array to perform mean over.
        min_N (int): minimum length of array to sigma clip, below which no sigma
        clipping is performed.
        verbose (bool): print number of MAD-clipped data points.

    Returns:
        robust sigma clipped mean (ndarray).
    """
    if stat in ('mean', 'median'):
        np_stat = getattr(np, 'nan{}'.format(stat))
    else:
        raise ValueError('stat must be either "mean" or "median".')

    iscomplex = np.iscomplexobj(data)

    if np.isnan(data).all():
        utils.echo('All-nan slice encountered - returning nan.', verbose=verbose)
        if data.ndim == 1:
            if iscomplex:
                return np.nan + 1j*np.nan
            else:
                return np.nan
        else:
            rtn_arr_shape = np.delete(data.shape, axis)
            return np.empty(rtn_arr_shape, dtype=data.dtype) * np.nan

    if iscomplex:
        if flags is not None:
            flg_count = flags.sum()
        else:
            flg_count = 0
        real_d, real_f = mad_clip(data.real, flags=flags, sigma=sigma, axis=axis, \
                                  min_N=min_N)
        # ensure nans are in imag part too
        data_im = data.imag
        if np.isnan(data.real).any():
            data_im[np.isnan(data.real)] = np.nan
        imag_d, imag_f = mad_clip(data_im, flags=flags, sigma=sigma, axis=axis, \
                                  min_N=min_N)
        utils.echo('{} data points MAD-clipped.'.format(real_f.sum()+imag_f.sum()-flg_count), \
                   verbose=verbose)
        mc_data = real_d + 1j * imag_d
        mc_data[real_f + imag_f] *= np.nan # flag data if flagged in either Re or Imag component
        return np_stat(mc_data, axis=axis)

    else:
        d, f = mad_clip(data, flags=flags, sigma=sigma, axis=axis, min_N=min_N, \
                        verbose=verbose)
        return np_stat(d, axis=axis)


rsc_mean = lambda data, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False: \
    rsc_avg(data, 'mean', flags, sigma, axis, min_N, verbose)

rsc_median = lambda data, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False: \
    rsc_avg(data, 'median', flags, sigma, axis, min_N, verbose)


def rsc_avg_astropy(data, stat, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False):
    """
    Location of data after robust sigma clipping.

    Astropy implementation.

    Args:
        data (ndarray): input data.
        stat (str): statistic to use for averaging ('mean', 'median').
        flags (ndarray): existing boolean flags for data array. True if flagged.
        sigma (ndarray): sigma threshold to cut above.
        axis (int): axis of array to perform mean over.
        min_N (int): minimum length of array to sigma clip, below which no sigma
        clipping is performed.
        verbose (bool): print number of MAD-clipped data points.

    Returns:
        robust sigma clipped mean (ndarray).
    """
    if stat in ('mean', 'median'):
        np_stat = getattr(np, 'nan{}'.format(stat))
    else:
        raise ValueError('stat must be either "mean" or "median".')

    iscomplex = np.iscomplexobj(data)
    if iscomplex:
        # ensure imag entries are also nan'd
        if np.isnan(data).any():
            data[np.isnan(data)] *= np.nan

    if flags is not None:
        data[flags] *= np.nan

    if np.isnan(data).all():
        utils.echo('All-nan slice encountered - returning nan.', verbose=verbose)
        if data.ndim == 1:
            if iscomplex:
                return np.nan + 1j*np.nan
            else:
                return np.nan
        else:
            rtn_arr_shape = np.delete(data.shape, axis)
            return np.empty(rtn_arr_shape, dtype=data.dtype) * np.nan

    # pass ignore_nan=True argument to astropy mad_std
    def mad_std_(d, axis=None):
        return mad_std(d, axis=axis, func=None, ignore_nan=True)

    # ensure array passes min_N criteria:
    if isinstance(axis, int):
        data_N = data.shape[axis]
    else:
        data_N = np.prod([ax for i, ax in enumerate(data.shape) if i not in axis])
    if data_N < min_N:
        utils.echo('No sigma clipping performed as length of array for specified axis ' \
                   '< min_N.', verbose=verbose)
        return np_stat(data, axis=axis)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Input data contains invalid values')
        if iscomplex:
            sc_re = sigma_clip(data.real, cenfunc=np.nanmedian, stdfunc=mad_std_, \
                               sigma=sigma, maxiters=1, axis=axis)
            sc_im = sigma_clip(data.imag, cenfunc=np.nanmedian, stdfunc=mad_std_, \
                               sigma=sigma, maxiters=1, axis=axis)
            sc = sc_re + 1j*sc_im
            sc.set_fill_value(value=np.nan+1j*np.nan)

        else:
            sc = sigma_clip(data, cenfunc=np.nanmedian, stdfunc=mad_std_, \
                            sigma=sigma, maxiters=1, axis=axis)
            sc.set_fill_value(value=np.nan)

    return np_stat(sc.filled(), axis=axis)
