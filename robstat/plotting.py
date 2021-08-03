"""Plotting functions for Jupyter notebooks"""


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap


def row_heatmaps(arrs, apply_np_fn=None, clip_pctile=None, vmin=None, vmax=None, \
                 center=None, annot=False, fmt=None, xbase=5, ybase=10, titles=None, \
                 share_cbar=True, cbar_loc=None, figsize=(14, 6)):
    """
    Plot a row of heatmaps with shared colour bar.

    Args:
        arrs (list): list of ndarrays to plot.
        apply_np_fn (str): numpy function to apply to arrs.
        clip_pctile (float): top and bottom percentile of data to clip. No clipping
        if None.
        vmin, vmax (float): values to anchor the colormap. Supersedes clip_pctile.
        center (float): value at which to center the colormap when plotting divergant data.
        annot (bool): list of ndarrays to plot.
        fmt (str): string formatting code to use when adding annotations.
        xbase, ybase (int): set a tick on each integer multiple of the base.
        titles (list): list of strings to set as titles.
        share_cbar (bool): whether to have the same color bar for all plots, located
        at the right of the row
        cbar_loc (str): location of colorbar ("top", "bottom"). Only applicable
        if share_cbar is False.
        figsize (tuple): width, height in inches.
    """
    if isinstance(arrs, np.ndarray):
        arrs = [arrs]

    if share_cbar:
        width_ratios = np.append(np.ones(len(arrs)), [0.05*len(arrs)])
        ncols = len(arrs) + 1
    else:
        width_ratios = None
        ncols = len(arrs)

    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, \
                             gridspec_kw = {'width_ratios': width_ratios})

    if apply_np_fn is not None:
        arrs_np = [getattr(np, apply_np_fn)(arr) for arr in arrs]
        if apply_np_fn == 'imag':
            for i, arr in enumerate(arrs):
                if np.isnan(arr.real).any():
                    arrs_np[i][np.isnan(arr.real)] = np.nan
        arrs = arrs_np

    if vmin is None and vmax is None and share_cbar:
        all_values = np.concatenate(arrs).flatten()
        vmin = np.nanmin(all_values)
        vmax = np.nanmax(all_values)
        if clip_pctile is not None:
            vmin = np.nanpercentile(all_values, clip_pctile)
            vmax = np.nanpercentile(all_values, 100 - clip_pctile)
            all_values = np.clip(all_values, vmin, vmax)
        if center is not None:
            abs_values = np.abs(all_values)
            vmax = np.nanmax(abs_values)
            vmin = -vmax

    if annot and arrs[0].size > 50:
        annot = False

    if apply_np_fn == 'angle' or center == 0:
        cmap = 'bwr'
    else:
        cmap = sns.cm.rocket_r

    if cbar_loc is not None:
        cbar_kws = {'use_gridspec': False, 'location': cbar_loc}
    else:
        cbar_kws = None

    bool_arrs = [arr.dtype == bool for arr in arrs]
    if any(bool_arrs):
        bool_idxs = np.where(bool_arrs)[0]

        bool_colors = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        bool_cmap = LinearSegmentedColormap.from_list('Custom', bool_colors, \
                                                 len(bool_colors))
    else:
        bool_idxs = []

    yticklabels = True
    for i, arr in enumerate(arrs):
        if i in bool_idxs and not share_cbar:
            cmap_ = bool_cmap
        else:
            cmap_ = cmap
        ax = sns.heatmap(arr, cmap=cmap_, ax=axes[i], cbar=not share_cbar, \
            vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, \
            yticklabels=yticklabels, cbar_kws=cbar_kws)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if titles is not None:
            ax.set_title(titles[i])
        if i == 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            yticklabels = False
        if i in bool_idxs and not share_cbar:
            # Set the bool colorbar labels
            colorbar = ax.collections[0].colorbar
            colorbar.set_ticks([0.25,0.75])
            colorbar.set_ticklabels(['False', 'True'])

    if share_cbar:
        fig.colorbar(axes[0].collections[0], cax=axes[-1])

    plt.show()


def grid_heatmaps(arrs, apply_np_fn=None, clip_pctile=None, vmin=None, vmax=None, \
                  center=None, annot=False, fmt=None, xbase=5, ybase=10, titles=None, \
                  ylabels=None, share_cbar=True, figsize=(14, 6)):
    """
    Plot a row of heatmaps with shared colour bar.

    Args:
        arrs (list): list of list of ndarrays to plot.
        apply_np_fn (str): numpy function to apply to arrs.
        clip_pctile (float): top and bottom percentile of data to clip. No clipping
        if None.
        vmin, vmax (float): values to anchor the colormap. Supersedes clip_pctile.
        center (float): value at which to center the colormap when plotting divergant data.
        annot (bool): list of ndarrays to plot.
        fmt (str): string formatting code to use when adding annotations.
        xbase, ybase (int): set a tick on each integer multiple of the base.
        titles (list): list or list of lists of strings to set as titles.
        ylabels (list): ylabels for each row.
        share_cbar (bool): whether to have the same color bar for all plots across rows.
        figsize (tuple): width, height in inches.
    """
    if isinstance(arrs, np.ndarray):
        arrs = [[arrs]]

    if len(arrs) != 1:
        it = iter(arrs)
        next_len = len(next(it))
        if not all([len(l) == next_len for l in it]):
            raise ValueError('arrs must be a list of lists of the same length')
    else:
        assert all(isinstance(arr, list) for arr in arrs)

    if not all(isinstance(title, list) for title in titles):
        top_titles = True
    else:
        top_titles = False

    if share_cbar:
        width_ratios = np.append(np.ones(len(arrs)), [0.05*len(arrs)])
        ncols = len(arrs) + 1
    else:
        width_ratios = None
        ncols = len(arrs)

    fig, axes = plt.subplots(nrows=len(arrs[0]), ncols=ncols, figsize=figsize, \
                             gridspec_kw = {'width_ratios': width_ratios})

    if apply_np_fn is not None:
        arrs_np = [[getattr(np, apply_np_fn)(a) for a in arr] for arr in arrs]
        if apply_np_fn == 'imag':
            for i, arr in enumerate(arrs):
                for j, a in enumerate(arr):
                    if np.isnan(a.real).any():
                        arrs_np[i][j][np.isnan(a.real)] = np.nan
        arrs = arrs_np

    if vmin is None and vmax is None and share_cbar:
        all_values = np.concatenate(arrs).flatten()
        if clip_pctile is None and center is None:
            s_arrs = np.array(arrs)
            vmin_arr = np.nanmin(s_arrs, axis=(0, 2, 3))
            vmax_arr = np.nanmax(s_arrs, axis=(0, 2, 3))
            use_vmm_arr = True
        else:
            use_vmm_arr = False
            if clip_pctile is not None:
                vmin = np.nanpercentile(all_values, clip_pctile)
                vmax = np.nanpercentile(all_values, 100 - clip_pctile)
                all_values = np.clip(all_values, vmin, vmax)
            if center is not None:
                abs_values = np.abs(all_values)
                vmax = np.nanmax(abs_values)
                vmin = -vmax
    else:
        use_vmm_arr = False

    if annot and arrs[0][0].size > 50:
        annot = False

    if apply_np_fn == 'angle' or center == 0:
        cmap = 'bwr'
    else:
        cmap = sns.cm.rocket_r

    yticklabels = True
    for col, arr in enumerate(arrs):
        for row, a in enumerate(arr):
            if row == len(arr) - 1:
                xticklabels = True
            else:
                xticklabels = False
            if use_vmm_arr:
                vmin = vmin_arr[row]
                vmax = vmax_arr[row]
            ax = sns.heatmap(a, cmap=cmap, ax=axes[row][col], cbar=not share_cbar, \
                        vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, \
                        xticklabels=xticklabels, yticklabels=yticklabels)
            if row == len(arr) - 1:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            if titles is not None:
                if top_titles:
                    if row == 0:
                        ax.set_title(titles[col])
                else:
                    ax.set_title(titles[col][row])
            if col == 0:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                yticklabels = False
                if ylabels is not None:
                    ax.set_ylabel(ylabels[row])
                if share_cbar:
                    fig.colorbar(axes[row][0].collections[0], cax=axes[row][-1])

    plt.show()
