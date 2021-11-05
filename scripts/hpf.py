import multiprocess as multiprocessing
import os

import numpy as np

import uvtools

from robstat.ml import extrem_nans
from robstat.utils import DATAPATH


def main():
    # xd_vis_file = 'xd_vis_rph.npz'
    xd_vis_file = 'lstb_no_avg/idr2_lstb_14m_ee_1.40949.npz'
    mp = True # turn on multiprocessing

    xd_vis_file_path = os.path.join(DATAPATH, xd_vis_file)
    hpf_vis_file = os.path.join(DATAPATH, xd_vis_file.replace('.npz', '_hpf.npz'))

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
        filter_half_widths = [1e-6] # half-width of rectangular fourier regions to filter
        mode = 'dayenu_dpss_leastsq'

        def bl_iter(bl):
            hpf_data_d = np.empty((no_days, no_chans, no_tints), dtype=complex)
            for day in range(no_days):
                data = xd_data[day, ..., bl]
                flgs = xd_flags[day, ..., bl]

                if flgs.all():
                    d_res_d = np.empty_like(data) * np.nan
                else:
                    ex_nans = extrem_nans(np.isnan(data).all(axis=1))
                    s_idxs, e_idxs = np.split(ex_nans, np.where(np.ediff1d(ex_nans) > 1)[0]+1)
                    s = s_idxs.max() + 1
                    e = e_idxs.min()                    
                    
                    data_tr = data[s:e, :].copy()
                    flgs_tr = flgs[s:e, :]
                    data_tr[flgs_tr] = 0. # needed for fourier_filter to work properly
                    wgts = np.logical_not(flgs_tr).astype(float)
                    freqs_tr = freqs[s:e]

                    _, d_res_tr, info = uvtools.dspec.fourier_filter(freqs_tr, data_tr, wgts, filter_centers,
                        filter_half_widths, mode, filter_dims=0, skip_wgt=0., zero_residual_flags=True, \
                        max_contiguous_edge_flags=data_tr.shape[0])

                    d_res_tr[flgs_tr] *= np.nan

                    d_res_d = np.empty_like(data)*np.nan
                    d_res_d[s:e, :] = d_res_tr

                hpf_data_d[day, ...] = d_res_d

            return  hpf_data_d[..., np.newaxis]

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

        np.savez(hpf_vis_file, data=hpf_data, **metadata)
        print('HPF visibility file saved to: {}'.format(hpf_vis_file))


if __name__ == '__main__':
    main()
