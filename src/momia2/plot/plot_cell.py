from ..utils import *
from matplotlib import pyplot as plt

__all__ = ['plot_cells']

def plot_cells(patch,
            cell_ids=[],
            figsize=(5, 5),
            outline_prop={},
            midline_prop={},
            cell_plot=False,
            channel=-1):

    """
    Plots cells from a given patch with outlined boundaries and midlines (if available).

    :param patch: The patch containing cell data
    :type patch: Patch object
    :param cell_ids: List of cell ids to be plotted
    :type cell_ids: list, optional
    :param figsize: Size of the figure to be plotted
    :type figsize: tuple, optional
    :param outline_prop: Properties for the cell outline (color, lw - linewidth, ls - linestyle)
    :type outline_prop: dict, optional
    :param midline_prop: Properties for the cell midline (color, lw - linewidth, ls - linestyle)
    :type midline_prop: dict, optional
    :param cell_plot: If True, plot individual cell images, else plot all cells on patch
    :type cell_plot: bool, optional
    :param channel: Channel index for multi-channel images. If -1, use reference image
    :type channel: int, optional
    """

    outline_color = outline_prop['color'] if 'color' in outline_prop else 'orange'
    outline_lw = float(outline_prop['lw']) if 'lw' in outline_prop else 1
    outline_ls = outline_prop['ls'] if 'ls' in outline_prop else '-'
    midline_color = midline_prop['color'] if 'color' in midline_prop else 'orange'
    midline_lw = midline_prop['lw'] if 'lw' in midline_prop else 1
    midline_ls = midline_prop['ls'] if 'ls' in midline_prop else '-'

    if channel == -1:
        img = patch.get_ref_image()
    else:
        img = patch.get_channel_data(channel)

    if len(cell_ids) == 0:
        cell_ids = patch.regionprops.index[patch.regionprops['$include'] == 1]
    if cell_plot:
        for i in cell_ids:
            fig = plt.figure(figsize=figsize)
            if i in patch.regionprops.index:
                x1, y1, x2, y2, outline, midlines, raw_outline = patch.regionprops.loc[
                    i, ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2',
                        '$refined_outline', '$midlines', '$outline']].values
                plt.imshow(img[x1:x2, y1:y2], cmap='gist_gray')
                if len(outline) > 2:
                    plt.plot(outline[:, 1], outline[:, 0], color=outline_color, lw=outline_lw, ls=outline_ls)
                elif len(raw_outline) > 2:
                    plt.plot(raw_outline[:, 1], raw_outline[:, 0], color=outline_color, lw=outline_lw,
                             ls=outline_ls)
                if len(midlines) > 0:
                    for midline in midlines:
                        plt.plot(midline[:, 1], midline[:, 0],
                                 color=midline_color, lw=midline_lw, ls=midline_ls)
    else:
        fig = plt.figure(figsize=figsize)
        plt.imshow(img, cmap='gist_gray')
        for i in cell_ids:
            if i in patch.regionprops.index:
                x1, y1, x2, y2, outline, midlines, raw_outline = patch.regionprops.loc[
                    i, ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2',
                        '$refined_outline', '$midlines', '$outline']].values
                if len(outline) > 2:
                    plt.plot(outline[:, 1] + y1, outline[:, 0] + x1, color=outline_color, lw=outline_lw,
                             ls=outline_ls)
                elif len(raw_outline) > 2:
                    plt.plot(raw_outline[:, 1] + y1, raw_outline[:, 0] + x1, color=outline_color,
                             lw=outline_lw, ls=outline_ls)
                if len(midlines) > 0:
                    for midline in midlines:
                        plt.plot(midline[:, 1] + y1, midline[:, 0] + x1, color=midline_color, lw=midline_lw,
                                 ls=midline_ls)