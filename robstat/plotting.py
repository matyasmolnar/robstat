"""Plotting functions for Jupyter notebooks"""


import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker


def row_heatmaps(arrs, apply_np_fn=None, clip_pctile=None, vmin=None, vmax=None, \
                 center=None, annot=False, fmt=None, xbase=5, ybase=10, \
                 figsize=(14, 6)):
    """
    Plot a row of heatmaps with shared colour bar.

    Args:
        arrs (ndarray): list of ndarrays to plot.
        apply_np_fn (str): numpy function to apply to arrs.
        clip_pctile (float): top and bottom percentile of data to clip. No clipping
        if None.
        vmin, vmax (float): values to anchor the colormap. Supersedes clip_pctile.
        center (float): value at which to center the colormap when plotting divergant data.
        annot (bool): list of ndarrays to plot.
        fmt (str): string formatting code to use when adding annotations.
        xbase, ybase (int): set a tick on each integer multiple of the base.
        figsize (tuple): width, height in inches.
    """
    if isinstance(arrs, np.ndarray):
        arrs = [arrs]

    width_ratios = np.append(np.ones(len(arrs)), [0.05*len(arrs)])

    fig, axes = plt.subplots(ncols=len(arrs)+1, figsize=figsize, \
                             gridspec_kw = {'width_ratios': width_ratios})

    if apply_np_fn is not None:
        arrs = [getattr(np, apply_np_fn)(arr) for arr in arrs]

    if vmin is None and vmax is None and clip_pctile is not None:
        all_values = np.concatenate(arrs).flatten()
        vmin = np.nanpercentile(all_values, clip_pctile)
        vmax = np.nanpercentile(all_values, 100 - clip_pctile)

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
        if i == 0:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(ybase))
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
            yticklabels = False

    fig.colorbar(axes[0].collections[0], cax=axes[-1])

    plt.show()
