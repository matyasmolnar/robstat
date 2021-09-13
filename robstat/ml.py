"""Machine learning and data manipulation techniques"""


import numpy as np
from scipy import interpolate

from . import utils


def nan_interp1d(data, kind='linear', verbose=False):
    """
    If nans are present in the 1D data array, these nan values will be replaced
    with interpolated values. If nans are present at the start or end of the array,
    those are removed (as to not extrapolate).

    Args:
        data (ndarray): 1D array with nans.
        kind (str): specifies the kind of interpolation e.g. {"linear", "quadratic",
        "cubic"}.
        verbose (bool): status updates of interpolation.

    Returns:
        1D array where nan entries have been interpolated.
    """
    if not np.isnan(data).any():
        utils.echo('No nan entries in array.', verbose=verbose)
        return data

    nn_data = data.copy()

    nan_slice = np.isnan(nn_data)
    nan_idxs = np.where(nan_slice)[0]

    fun_x = np.arange(nn_data.size)[~nan_slice]
    fun_y = nn_data[~nan_slice]

    interp_fun = interpolate.interp1d(fun_x, fun_y, kind=kind, \
                                      fill_value=np.nan, bounds_error=False)

    nn_data[nan_idxs] = interp_fun(nan_idxs)

    # remove first or last row if filled with nans
    if 0 in nan_idxs:
        nn_data = nn_data[1:]
    if nn_data.size-1 in nan_idxs:
        nn_data = nn_data[:-1]

    return nn_data


def nan_interp2d(data, kind='cubic', verbose=False):
    """
    If nans are present in the 2D data array, these nan values will be replaced
    with interpolated values. If nans are present at the start or end of the array,
    those are removed (as to not extrapolate).

    Args:
        data (ndarray): 2D array with nans.
        kind (str): specifies the kind of interpolation e.g. {"linear", "quadratic",
        "cubic"}.
        verbose (bool): status updates of interpolation.

    Returns:
        2D array where nan entries have been interpolated.
    """
    if not np.isnan(data).any():
        utils.echo('No nan entries in array.', verbose=verbose)
        return data

    # mask invalid values
    masked_arr = np.ma.masked_invalid(data)
    xx, yy = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    # get only the valid values
    x1 = xx[~masked_arr.mask]
    y1 = yy[~masked_arr.mask]
    masked_arr = masked_arr[~masked_arr.mask]

    nn_data = interpolate.griddata((x1, y1), masked_arr.ravel(),
                                   (xx, yy), method=kind)

    def extrem_nans(nan_data, stepsize=1):
        nan_idxs = np.where(nan_data)[0]
        first_idx = 0
        last_idx = nan_idxs.size - 1
        gc = np.split(nan_idxs, np.where(np.diff(nan_idxs) != stepsize)[0]+1)
        for i, grp in enumerate(gc):
            if (first_idx or last_idx) not in grp:
                gc.pop(i)
        return np.array(gc).flatten()

    # get rid of nans at extremities
    nans_t = np.isnan(nn_data).all(axis=0)
    nans_t_idxs = extrem_nans(nans_t)
    nn_data = np.delete(nn_data, nans_t_idxs, axis=0)

    nans_f = np.isnan(nn_data).all(axis=1)
    nans_f_idxs = extrem_nans(nans_f)
    nn_data = np.delete(nn_data, nans_f_idxs, axis=1)

    return nn_data
