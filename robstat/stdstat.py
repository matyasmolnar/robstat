"""Standard statistical techniques"""


import numpy as np

# attempt to import the JAX library for acceleration
# below allows a one-line switch to JAX
try:
    pJAX = True
    from jax.config import config
    config.update('jax_enable_x64', True)
    import jax
    from jax import numpy as jnp
except ImportError:
    pJAX = False
    jnp = np


def sigma_clip(data, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False):
    """
    Robust sigma clipping, about the median.

    This function will directly replace flagged and clipped data in array with
    a np.nan.

    Inspired by sigma_clip in HERA hera_cal sigma clipping routine for LST-binning:
    https://github.com/HERA-Team/hera_cal/blob/master/hera_cal/lstbin.py

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

    assert data.dtype  != np.complex # MAD clipping erroneous

    # ensure array passes min_N criteria:
    if data.shape[axis] < min_N:
        print('No sigma clipping performed as length of array for specified axis ' \
              '< min_N.')
        if flags is None:
            flags = np.zeros_like(data, np.bool)
        return data, flags

    # create empty clip_flags array
    clip_flags = np.zeros_like(data, np.bool)

    # inherit flags if fed and apply flags to data
    if flags is not None:
        clip_flags += flags
        sc_data[flags] *= np.nan

    flg_count = clip_flags.sum()

    # get robust location
    location = np.nanmedian(data, axis=axis)

    # get MAD! * 1.482579 correction factor
    scale = np.nanmedian(np.abs(data - location), axis=axis) * 1.482579

    # get clipped data
    clip = np.abs(data - location) / scale > sigma

    # set clipped data to nan and set clipped flags to True
    sc_data[clip] *= np.nan
    clip_flags[clip] = True
    if verbose:
        print('{} data points MAD-clipped.'.format(clip_flags.sum()-flg_count))

    return sc_data, clip_flags


def rsc_mean(data, flags=None, sigma=4.0, axis=0, min_N=4, verbose=False):
    """
    Mean of data after robust sigma clipping.

    Mimics HERA LST-binning data averaging.

    Args:
        data (ndarray): input data.
        flags (ndarray): existing boolean flags for data array. True if flagged.
        sigma (ndarray): sigma threshold to cut above..
        axis (int): axis of array to perform mean over.
        min_N (int): minimum length of array to sigma clip, below which no sigma
        clipping is performed.
        verbose (bool): print number of MAD-clipped data points.

    Returns:
        robust sigma clipped mean (ndarray).
    """
    if np.iscomplexobj(data):
        if flags is not None:
            flg_count = flags.sum()
        else:
            flg_count = 0
        real_d, real_f = sigma_clip(data.real, flags=flags, sigma=sigma, axis=axis, \
                                    min_N=min_N)
        # put nans are in imag part too
        data_im = data.imag
        if np.isnan(data.real).any():
            data_im[np.isnan(data.real)] = np.nan
        imag_d, imag_f = sigma_clip(data_im, flags=flags, sigma=sigma, axis=axis, \
                                    min_N=min_N)
        if verbose:
            print('{} data points MAD-clipped.'.format(real_f.sum()+imag_f.sum()-flg_count))
        return np.nanmean(real_d, axis=axis) + 1j * np.nanmean(imag_d, axis=axis)

    else:
        d = sigma_clip(data, flags=flags, sigma=sigma, axis=axis, min_N=min_N, \
                       verbose=verbose)
        return np.nanmean(d, axis=axis)