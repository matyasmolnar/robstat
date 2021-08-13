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

    # get rid of nans at extremities
    nan_idxs = np.where(np.isnan(nn_data))
    if 0 in nan_idxs[0]:
        nn_data = nn_data[1:, :]
    if nn_data.shape[0]-1 in nan_idxs[0]:
        nn_data = nn_data[:-1, :]
    if 0 in nan_idxs[1]:
        nn_data = nn_data[:, 1:]
    if nn_data.shape[1]-1 in nan_idxs[1]:
        nn_data = nn_data[:, :-1]

    return nn_data
