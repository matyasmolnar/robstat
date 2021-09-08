"""HERA visibility manipulation"""


from collections import OrderedDict as odict

import numpy as np

from hera_cal.datacontainer import DataContainer
from hera_cal.io import HERAData
from hera_cal.utils import lst_rephase


def agg_tint_rephase(xd_data, redg, freqs, pol, lsts, antpos, no_bins_agg=2):
    """Rephase n consecutive time integrations to their mean such that they can be
    averaged in time.

    Note that the data must have already been rephased to the bin centres when
    aligning across JDs (with simpleredcal.XDgroup_data(..., rephase=True)).

    :param xd_data: Across JDs, aligned in LST, data array with dimensions
    (JDs, freqs, tints, baselines).
    :type xd_data: ndarray
    :param redg: Grouped baselines, as returned by groupBls
    :type redg: ndarray
    :param freqs: Frequencies of data [Hz]
    :type freqs: array-like
    :param pol: Polarization of data
    :type pol: str
    :param lsts: LSTs of data [radians]
    :type lsts: array-like
    :param antpos: Antenna positions from HERAData container
    :type antpos: dict
    :param no_bins_agg: Number of consecutive bins to aggregate in the rephasing
    :type no_bins_agg: int

    :return: Rephased xd data array with rephased data for specified number of
    consecutive bins
    :rtype: ndarray
    """
    lst_bin_centres = np.mean(lsts.reshape(-1, no_bins_agg), axis=1)

    xd_data_rph = np.empty_like(xd_data)

    for jd_idx in range(xd_data.shape[0]):
        # convert to DataContainer to feed into lst_rephase hera_cal function
        data_cont = odict()
        for bl_idx, bl in enumerate(redg[:, 1:]):
            data_cont[(bl[0], bl[1], pol)] = xd_data[jd_idx, ..., bl_idx].transpose()
        data_cont = DataContainer(data_cont)

        if jd_idx == 0:
            # only need to compute once since ant positions fixed
            bls = odict([(k, antpos[k[0]] - antpos[k[1]]) for k in data_cont.keys()])

        dlst = np.asarray(np.repeat(lst_bin_centres, no_bins_agg) - lsts)

        data_cont = lst_rephase(data_cont, bls, freqs, dlst, inplace=False, array=False)

        for bl_idx, bl in enumerate(redg[:, 1:]):
            xd_data_rph[jd_idx, ..., bl_idx] = data_cont[(bl[0], bl[1], pol)].transpose()

    return xd_data_rph
