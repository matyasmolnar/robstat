"""Plotting functions for Jupyter notebooks"""


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


def row_heatmaps(arrs, apply_np_fn=None, clip_pctile=None, vmin=None, vmax=None, \
                 center=None, annot=False, fmt=None, xbase=5, ybase=10, titles=None, \
                 figsize=(14, 6)):
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
        figsize (tuple): width, height in inches.
    """
    if isinstance(arrs, np.ndarray):
        arrs = [arrs]

    width_ratios = np.append(np.ones(len(arrs)), [0.05*len(arrs)])

    fig, axes = plt.subplots(ncols=len(arrs)+1, figsize=figsize, \
                             gridspec_kw = {'width_ratios': width_ratios})

    if apply_np_fn is not None:
        arrs_np = [getattr(np, apply_np_fn)(arr) for arr in arrs]
        if apply_np_fn == 'imag':
            for i, arr in enumerate(arrs):
                if np.isnan(arr.real).any():
                    arrs_np[i][np.isnan(arr.real)] = np.nan
        arrs = arrs_np

    if vmin is None and vmax is None:
        all_values = np.concatenate(arrs).flatten()
        if clip_pctile is not None:
            vmin = np.nanpercentile(all_values, clip_pctile)
            vmax = np.nanpercentile(all_values, 100 - clip_pctile)
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

    yticklabels = True
    for i, arr in enumerate(arrs):
        ax = sns.heatmap(arr, cmap=cmap, ax=axes[i], cbar=False, \
                    vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, \
                    yticklabels=yticklabels)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if titles is not None:
            ax.set_title(titles[i])
        if i == 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            yticklabels = False

    fig.colorbar(axes[0].collections[0], cax=axes[-1])

    plt.show()


def grid_heatmaps(arrs, apply_np_fn=None, clip_pctile=None, vmin=None, vmax=None, \
                 center=None, annot=False, fmt=None, xbase=5, ybase=10, titles=None, \
                 ylabels=None, figsize=(14, 6)):
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
        titles (list): list of strings to set as titles.
        figsize (tuple): width, height in inches.
    """
    if isinstance(arrs, np.ndarray):
        arrs = [[arrs]]

    if len(arrs) != 1:
        it = iter(arrs)
        next_len = len(next(it))
        if not all([len(l) == next_len for l in it]):
            raise ValueError('arrs must be a list of lists of the same length')

    width_ratios = np.append(np.ones(len(arrs[0])), [0.05*len(arrs[0])])

    fig, axes = plt.subplots(nrows=len(arrs), ncols=len(arrs[0])+1, figsize=figsize, \
                             gridspec_kw = {'width_ratios': width_ratios})

    if apply_np_fn is not None:
        arrs_np = [[getattr(np, apply_np_fn)(a) for a in arr] for arr in arrs]
        if apply_np_fn == 'imag':
            for i, arr in enumerate(arrs):
                for j, a in enumerate(arr):
                    if np.isnan(a.real).any():
                        arrs_np[i][j][np.isnan(a.real)] = np.nan
        arrs = arrs_np

    if vmin is None and vmax is None:
        all_values = np.concatenate(arrs).flatten()
        if clip_pctile is not None:
            vmin = np.nanpercentile(all_values, clip_pctile)
            vmax = np.nanpercentile(all_values, 100 - clip_pctile)
        if center is not None:
            abs_values = np.abs(all_values)
            vmax = np.nanmax(abs_values)
            vmin = -vmax

    if annot and arrs[0][0].size > 50:
        annot = False

    if apply_np_fn == 'angle' or center == 0:
        cmap = 'bwr'
    else:
        cmap = sns.cm.rocket_r

    yticklabels = True
    for col, arr in enumerate(arrs):
        for row, a in enumerate(arr):
            ax = sns.heatmap(a, cmap=cmap, ax=axes[row][col], cbar=False, \
                        vmin=vmin, vmax=vmax, center=center, annot=annot, fmt=fmt, \
                        yticklabels=yticklabels)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xbase))
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            if titles is not None and row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                yticklabels = False
                if ylabels is not None:
                    ax.set_ylabel(ylabels[row])

                fig.colorbar(axes[row][0].collections[0], cax=axes[row][-1])

    plt.show()
