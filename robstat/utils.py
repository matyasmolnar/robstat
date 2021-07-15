"""Utility functions"""


import os
from pathlib import Path

import numpy as np


DATAPATH = os.path.join(Path(__file__).parent.absolute(), 'data')


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
    return np.vstack((arr.real, arr.imag)).transpose()
