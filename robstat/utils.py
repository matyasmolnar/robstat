"""Utility functions"""


import os
from pathlib import Path

import numpy as np


DATAPATH = os.path.join(Path(__file__).parent.absolute(), 'data')


def echo(message, type=0, verbose=True):
    """One line printing of messages when verbose

    Args:
        message (str): message to print.
        type (int): 0, 1 to specify wrapping if message
        verbose (bool): if to print message or not
    """
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-" * 40)


def decomposeCArray(arr):
    """
    Reformat 1D complex array into 2D real array

    The 1D complex array with elements [z_1, ..., z_i] is reformatted such
    that the new array has elements [[Re(z_1), Im(z_1)], ..., [Re(z_i), Im(z_i)]].

    Args:
        array (ndarray): 1D complex array

    Returns:
        2D real array (ndarray).
    """
    assert arr.ndim == 1
    assert arr.dtype  == np.complex
    dcmp_arr = np.vstack((arr.real, arr.imag)).transpose()
    # make corresponding imag cells to nan values also nan
    if np.isnan(dcmp_arr).any():
        dcmp_arr[:, 1][np.isnan(dcmp_arr)[:, 0]] = np.nan
    return dcmp_arr


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier


def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return np.floor(n * multiplier) / multiplier
