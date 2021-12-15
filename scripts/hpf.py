import argparse
import multiprocess as multiprocessing
import os
import textwrap
from pathlib import Path

import numpy as np

import uvtools

from robstat.ml import extrem_nans
from robstat.utils import DATAPATH


def s_e_idxs(ex_nans, arr_size):
    if ex_nans.size == 0:
        start = 0
        end = arr_size
    else:
        # if only one edge has flags
        if (np.ediff1d(ex_nans) == 1).all():
            if 0 in ex_nans:
                start = ex_nans.max() + 1
                end = arr_size
            else:
                start = 0
                end = ex_nans.min()
        else:
            s_idxs, e_idxs = np.split(ex_nans, np.where(np.ediff1d(ex_nans) > 1)[0]+1)
            start = s_idxs.max() + 1
            end = e_idxs.min()
    return start, end


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent(
    """High pass filtering of visibilities

    Takes a given HERA visibility array with dimensions
    [days, channels, times, baselines] and performs a high pass filter
    along the channels axis using the DAYENU - see
    https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.5195E/abstract
    """))
    parser.add_argument('vis_file', help='Visibility file to filter', metavar='V', type=str)
    parser.add_argument('-i', '--in_dir', required=False, default=DATAPATH,
                        metavar='I', type=str, help='Directory in which input visibility file lives')
    parser.add_argument('-o', '--out_dir', required=False, default=None,
                        metavar='O', type=str, help='Directory in which to save filtered visibilities')
    parser.add_argument('-e', '--ext', required=False, default='hpf',
                        metavar='E', type=str, help='HPF file extension')
    parser.add_argument('-m', '--mode', required=False, default='dayenu_dpss_leastsq',
                        metavar='M', type=str, help='Filtering mode')
    parser.add_argument('-w', '--flt_width', required=False, default=1.5,
                        metavar='W', type=float, help='Filter half width, in ms')
    parser.add_argument('-p', '--multi_proc', required=False, action='store_true',
                        help='Whether to employ multiprocessing across day baseline slices')
    args = parser.parse_args()

    xd_vis_file = args.vis_file

    in_dir = args.in_dir
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = in_dir
    if not os.path.exists(out_dir):
        path = Path(out_dir)
        path.mkdir(parents=True)

    xd_vis_file_path = os.path.join(in_dir, xd_vis_file)
    hpf_vis_file = os.path.join(out_dir, xd_vis_file.replace('.npz', '_{}.npz'.format(args.ext)))

    mp = args.multi_proc

    if not os.path.exists(hpf_vis_file):

        # load dataset
        sample_xd_data = np.load(xd_vis_file_path)

        xd_data = sample_xd_data['data']  # dimensions (days, freqs, times, bls)
        xd_redg = sample_xd_data['redg']
        xd_rad_lsts = sample_xd_data['lsts']
        xd_pol = sample_xd_data['pol'].item()
        JDs = sample_xd_data['JDs']
        no_days = xd_data.shape[0]

        if 'lstb_no_avg' in xd_vis_file:
            xd_flags = np.isnan(xd_data)
            freqs = np.linspace(1e8, 2e8, 1025)[:-1]
            no_chans = freqs.size
        else:
            xd_flags = sample_xd_data['flags']
            freqs = sample_xd_data['freqs']
            chans = sample_xd_data['chans']
            no_chans = chans.size

        f_resolution = np.median(np.ediff1d(freqs))
        no_tints = xd_rad_lsts.size
        no_bls = xd_data.shape[3]


        # HPF with DAYENU
        filter_centers = [0.] # center of rectangular fourier regions to filter
        filter_half_widths = [args.flt_width*1e-6] # half-width of rectangular fourier regions to filter
        mode = args.mode

        skipped_slices = []

        def bl_iter(bl):
            hpf_data_d = np.empty((no_days, no_chans, no_tints), dtype=complex)
            for day in range(no_days):
                data = xd_data[day, ..., bl]
                flgs = xd_flags[day, ..., bl]

                d_res_d = np.empty_like(data)
                d_res_d.fill(np.nan + 1j*np.nan)

                if not flgs.all():
                    # flagged channels
                    ex_nans = extrem_nans(flgs.all(axis=1))
                    s, e = s_e_idxs(ex_nans, data.shape[0])

                    # flagged tints
                    isnan_tints = flgs.all(axis=0)
                    nnan_tints = np.logical_not(isnan_tints).nonzero()[0]

                    data_tr = data[s:e, nnan_tints].copy()
                    flgs_tr = flgs[s:e, nnan_tints]
                    data_tr[flgs_tr] = 0. # needed for fourier_filter to work properly
                    wgts = np.logical_not(flgs_tr).astype(float)
                    freqs_tr = freqs[s:e]

                    filter_kwargs = dict()
                    if mode != 'clean':
                        filter_kwargs['max_contiguous_edge_flags'] = no_chans

                    try:
                        _, d_res_tr, info = uvtools.dspec.fourier_filter(freqs_tr, data_tr, wgts, filter_centers,
                            filter_half_widths, mode, filter_dims=0, skip_wgt=0., zero_residual_flags=True, \
                            **filter_kwargs)

                        status_dict = info['status']['axis_0']
                        if 'skipped' in status_dict.values():
                            skipped_tints = list({k: v for k, v in status_dict.items() if v == 'skipped'}.keys())
                            print('Re-filtering tints {} for day/bl slice {}, {} as it was skipped when filtering '\
                                  'the 2D array'.format(nnan_tints[skipped_tints], day, bl))

                            for tint in skipped_tints:
                                # flagged channels
                                data_t = data_tr[:, tint]
                                flgs_t = flgs_tr[:, tint]

                                d_res_t = np.empty_like(data_t)
                                d_res_t.fill(np.nan + 1j*np.nan)

                                ex_nans = extrem_nans(flgs_t)
                                s_t, e_t = s_e_idxs(ex_nans, data_t.size)

                                data_t_tr = data_t[s_t:e_t]
                                flgs_t_tr = flgs_t[s_t:e_t]
                                wgts = np.logical_not(flgs_t_tr).astype(float)
                                freqs_t_tr = freqs_tr[s_t:e_t]

                                _, d_res_tr_t, info = uvtools.dspec.fourier_filter(freqs_t_tr, data_t_tr, wgts, filter_centers,
                                    filter_half_widths, mode, filter_dims=1, skip_wgt=0., zero_residual_flags=True, \
                                    **filter_kwargs)

                                d_res_t[s_t:e_t] = d_res_tr_t
                                d_res_tr[:, tint] = d_res_t

                                if info['info_deconv']['status']['axis_1'][0] == 'skipped' or info['status']['axis_1'][0] == 'skipped':
                                    skipped_slices.append((day, nnan_tints[tint], bl))
                                    d_res_tr[:, tint] *= np.nan
                                    print('Filter for tint {} for day/bl slice {}, {} failed'.\
                                          format(nnan_tints[tint], day, bl))

                    except ValueError:
                        # sometimes ValueError: On entry to DLASCL parameter number 4 had an illegal value is raised
                        # this occurs when the band edges need more trimming

                        print('Manually iterating through tints for day/bl slice {}, {}'.format(day, bl))

                        d_res_tr = np.empty_like(data_tr)
                        d_res_tr.fill(np.nan + 1j*np.nan)

                        for tint in range(data_tr.shape[1]):
                            data_t = data_tr[:, tint]
                            flgs_t = flgs_tr[:, tint]

                            # flagged channels
                            ex_nans = extrem_nans(flgs_t)
                            s_t, e_t = s_e_idxs(ex_nans, data_t.size)

                            data_t_tr = data_t[s_t:e_t]
                            flgs_t_tr = flgs_t[s_t:e_t]
                            wgts = np.logical_not(flgs_t_tr).astype(float)
                            freqs_t_tr = freqs_tr[s_t:e_t]

                            _, d_res_tr_t, info = uvtools.dspec.fourier_filter(freqs_t_tr, data_t_tr, wgts, filter_centers,
                                filter_half_widths, mode, filter_dims=1, skip_wgt=0., zero_residual_flags=True, \
                                **filter_kwargs)

                            if info['info_deconv']['status']['axis_1'][0] == 'skipped' or info['status']['axis_1'][0] == 'skipped':
                                skipped_slices.append((day, tint, bl))
                                d_res_tr_t *= np.nan
                                print('Filter for tint {} for day/bl slice {}, {} failed'.\
                                        format(tint, day, bl))
                                
                            d_res_tr[s_t:e_t, tint] = d_res_tr_t
                            

                    d_res_tr[flgs_tr] *= np.nan

                    d_res_d[s:e, nnan_tints] = d_res_tr

                # print('day/bl slice {}, {} done'.format(day, bl))
                # assert (np.isnan(d_res_d) == np.isnan(data)).all()

                hpf_data_d[day, ...] = d_res_d

            return hpf_data_d[..., np.newaxis]

        if mp:
            m_pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), no_bls))
            pool_res = m_pool.map(bl_iter, range(no_bls))
            m_pool.close()
            m_pool.join()
        else:
            pool_res = list(map(bl_iter, range(no_bls)))

        hpf_data = np.concatenate(pool_res, axis=3)


        # save results
        hpf_data[xd_flags] *= np.nan

        keys = list(sample_xd_data.keys())
        keys.remove('data')
        antpos_in = 'antpos' in keys
        if antpos_in:
            keys.remove('antpos')
        metadata = {k: sample_xd_data[k] for k in keys}
        if antpos_in:
            metadata['antpos'] = np.load(xd_vis_file_path, allow_pickle=True)['antpos'].item()

        np.savez(hpf_vis_file, data=hpf_data, skipped=skipped_slices,
                 filter_half_widths=filter_half_widths, mode=mode, **metadata)
        print('HPF visibility file saved to: {}'.format(hpf_vis_file))

    print('HPF visibility file already exists at: {}'.format(hpf_vis_file))


if __name__ == '__main__':
    main()
