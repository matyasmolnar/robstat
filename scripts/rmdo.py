import argparse
import multiprocess as multiprocessing
import os
import textwrap
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.covariance import MinCovDet

from robstat.utils import DATAPATH, decomposeCArray, flt_nan


def create_mp_array(arr):
    shared_arr = multiprocessing.RawArray(np.ctypeslib.as_ctypes_type(arr.dtype), int(np.prod(arr.shape)))
    new_arr = np.frombuffer(shared_arr, arr.dtype).reshape(arr.shape)  # shared_arr and new_arr the same memory
    new_arr[...] = arr
    return shared_arr, new_arr


def mp_init(shared_arr_, sharred_arr_shape_, sharred_arr_dtype_):
    global shared_arr, sharred_arr_shape, sharred_arr_dtype
    shared_arr = shared_arr_
    sharred_arr_shape = sharred_arr_shape_
    sharred_arr_dtype = sharred_arr_dtype_


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent(
    """High pass filtering of visibilities

    Takes a given HERA visibility array with dimensions
    [days, channels, times, baselines] and performs a high pass filter
    along the channels axis using the DAYENU - see
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.5195E/abstract
    """))
    parser.add_argument('vis_file', help='Visibility file to perform outlier detection on',
                        metavar='V', type=str)
    parser.add_argument('-i', '--in_dir', required=False, default=DATAPATH,
                        metavar='I', type=str, help='Directory in which input visibility file lives')
    parser.add_argument('-o', '--out_dir', required=False, default=None,
                        metavar='O', type=str, help='Directory in which to save filtered visibilities')
    parser.add_argument('-e', '--ext', required=False, default='rmd_clip_f',
                        metavar='E', type=str, help='HPF file extension')
    parser.add_argument('-s', '--sigma', required=False, default=5,
                        metavar='S', type=float, help='Sigma equivalent for outlier threshold')
    args = parser.parse_args()

    vis_file = args.vis_file
    sigma = args.sigma
    # for cleaner file naming
    if not isinstance(sigma, int):
        if sigma.is_integer():
            sigma = int(sigma)

    in_dir = args.in_dir
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(out_dir):
        path = Path(out_dir)
        path.mkdir(parents=True)

    vis_file_path = os.path.join(in_dir, vis_file)
    out_file = os.path.join(out_dir, vis_file.replace('.npz', '.{}_{}sig.npz'.format(args.ext, sigma)))

    if not os.path.exists(out_file):

        # load dataset
        global data
        vis_data = np.load(vis_file_path)
        data = vis_data['data']  # dimensions (days, freqs, times, bls)
        flags = np.isnan(data)

        # relationship between the quantiles of the chi-squared distribution and the quantiles of
        # the standard normal distribution (Fisher 1934)
        chi2_q = 0.5 * (sigma + np.sqrt(2*2 - 1))**2

        def mp_iter(s):
            d = data[:, s[0], s[1], s[2]]
            if not np.isnan(d).all():

                isfinite = np.isfinite(d).nonzero()[0]
                d = decomposeCArray(flt_nan(d))
                # choose random state 0 for reproducibility
                robust_cov = MinCovDet(random_state=0).fit(d)
                outliers = robust_cov.mahalanobis(d) > chi2_q

                rmd_clip_f = np.frombuffer(shared_arr, dtype).reshape(shape)
                rmd_clip_f[isfinite, s[0], s[1], s[2]] = outliers

        rmd_clip_f = np.ones_like(data, dtype=bool)
        # so that rmd_clip_f bool array with outlier information can be filled in parallel
        d_shared, rmd_clip_f = create_mp_array(rmd_clip_f)
        dtype = rmd_clip_f.dtype
        shape = rmd_clip_f.shape

        m_pool = multiprocessing.Pool(multiprocessing.cpu_count(), initializer=mp_init, \
                                      initargs=(d_shared, dtype, shape))
        _ = m_pool.map(mp_iter, np.ndindex(data.shape[1:]))
        m_pool.close()
        m_pool.join()

        rmd_clip_f = rmd_clip_f ^ flags

        np.savez(out_file, flags=rmd_clip_f)
        print('rMD outliers file saved to: {}'.format(out_file))

    else:
        print('rMD outliers file already exists at: {}'.format(out_file))


if __name__ == '__main__':
    main()
