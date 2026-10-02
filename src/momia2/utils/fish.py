from skimage import feature, measure, draw
from matplotlib import pyplot as plt, cm
from matplotlib.gridspec import GridSpec as gs
from ..utils import *

__all__ = ['find_puncta','puncta_heatmaps']

def find_puncta(img,mask,min_distance=1, threshold_rel=0.1):
    """

    :param img:
    :param mask:
    :param min_distance:
    :param threshold_rel:
    :return:
    """
    img_smoothed = filters.gaussian(img, sigma=0.5)
    img_smoothed = (img_smoothed - img_smoothed.min()) / (img_smoothed.max() - img_smoothed.min())
    LoG = ndi.gaussian_laplace(img_smoothed, sigma=1)
    extrema = feature.peak_local_max(-LoG * mask, min_distance=min_distance, threshold_rel=threshold_rel)
    output = []

    for ex in extrema:
        x, y = ex
        newx, newy, stable_maxima = subpixel_approximation_quadratic(x, y, img_smoothed)
        if stable_maxima:
            # z_image = local_puncta_zscore(img_smoothed, ex)
            # z_LoG = local_puncta_zscore(LoG, ex)
            output.append([newx, newy, img[x, y], LoG[x, y]])
    return np.array(output)


def subpixel_approximation_quadratic(x, y, energy_map, max_iteration=2):
    """

    :param x:
    :param y:
    :param energy_map:
    :param max_iteration:
    :return:
    """

    local_maxima_found = False
    for i in range(max_iteration):
        cube = energy_map[x - 1:x + 2, y - 1:y + 2]
        if cube[1, 1] == cube.max():
            local_maxima_found = True
            vx0, vx1, vx2 = cube[0, 1], cube[1, 1], cube[2, 1]
            vy0, vy1, vy2 = cube[1, 0], cube[1, 1], cube[1, 2]
            x += quadratic_maxima_approximation(vx0, vx1, vx2)
            y += quadratic_maxima_approximation(vy0, vy1, vy2)
            break
        else:
            dx, dy = np.array(np.unravel_index(cube.argmax(), cube.shape)) - 1
            x += dx
            y += dy
    return x, y, local_maxima_found


def quadratic_maxima_approximation(q1, q2, q3):
    """

    :param q1:
    :param q2:
    :param q3:
    :return:
    """
    if q2 <= max(q1, q3):
        raise ValueError('Mid point must be local maxima!')
    else:
        return (0.5 * (q1 - q3)) / (q1 - 2 * q2 + q3)


def local_puncta_zscore(image, coords, trim_low_bound=5):
    """

    :param image:
    :param coords:
    :param trim_low_bound:
    :return:
    """

    x, y = coords
    data = np.sort((image[x - 4:x + 5, y - 4:y + 5].copy()).flatten())[trim_low_bound:]

    threshold = filters.threshold_isodata(data, nbins=64)
    threshold = 0.5 * (threshold + data[-4:].mean())

    fg = data[data > threshold]
    bg = data[data <= threshold]

    if len(fg) <= 5:
        z = 0
    else:
        z = (np.mean(fg) - np.mean(bg)) / bg.std()
    return z


def standardize_punclist(punc):
    if len(punc) == 0:
        return np.array([None] * 6)
    else:
        return punc


def create_canvas(width=101, height=600):
    """

    :param width:
    :param height:
    :return:
    """
    canvas = np.zeros((width, height))
    r = int(width / 2)
    rr1, cc1 = draw.ellipse(r, int(r * 1.6), r, int((r) * 1.6))
    rr2, cc2 = draw.ellipse(r, int(height - 1.6 * r), r, int((r) * 1.6))
    rr3, cc3 = draw.rectangle(start=(1, int(r * 1.6)), end=(width - 2, height - int(r * 1.6)),
                              shape=canvas.shape)
    canvas[rr3, cc3] = 1
    canvas[rr1, cc1] = 1
    canvas[rr2, cc2] = 1
    l = len(np.nonzero(np.sum(canvas, axis=0))[0])
    counter = 0
    canvas = canvas.T
    return canvas


def create_mesh(canvas):
    """

    :param canvas:
    :return:
    """
    xt, yt = np.nonzero(canvas)
    l, m = np.sum(canvas, axis=0), np.sum(canvas, axis=1)
    norm_yt = np.zeros(xt.shape)
    norm_xt = (xt - xt.min()) / (l.max() - 1)
    count = 0
    for i in range(len(xt)):
        r = m.max() / m[xt[i]]
        norm_yt[i] = count * r / (m.max() - 1)
        if i != len(xt) - 1:
            if xt[i + 1] == xt[i]:
                count += 1
            else:
                count = 0
        else:
            count += 1
    return (xt, yt, norm_xt, norm_yt)


def project_image(xt, yt, norm_xt, norm_yt, canvas, data):
    """

    :param xt:
    :param yt:
    :param norm_xt:
    :param norm_yt:
    :param canvas:
    :param data:
    :return:
    """
    paint = np.zeros(canvas.shape)
    xid = norm_xt.copy()
    yid = norm_yt.copy()
    xid *= data.shape[0] - 1
    yid *= data.shape[1] - 1
    interpolated = bilinear_interpolate_numpy(data, xid, yid)
    paint[xt, yt] = interpolated
    return (paint)


def initiate_projection():
    """

    :return:
    """
    width = 75
    heights = [250, 350, 450, 550, 650]
    pad = np.zeros((50, width))
    gap = np.zeros((750, 10))
    padded = [gap]
    xt_list, yt_list, nxt_list, nyt_list = [], [], [], []
    for i in range(len(heights)):
        a = create_canvas(width=width, height=heights[i])
        half_pad = np.tile(pad, (len(heights) - i, 1))
        m_pad = np.concatenate([half_pad, a, half_pad], axis=0)
        m_pad = np.concatenate([m_pad, gap], axis=1)
        xt, yt, norm_xt, norm_yt = create_mesh(m_pad)
        xt_list.append(xt)
        yt_list.append(yt)
        nxt_list.append(norm_xt)
        nyt_list.append(norm_yt)
        padded.append(m_pad)
    contours = measure.find_contours(np.concatenate(padded, axis=1), level=0)
    optimized_outline = []
    for contour in contours:
        optimized_outline.append(spline_approximation(contour, n=2 * len(contour)))
    return padded, xt_list, yt_list, nxt_list, nyt_list, optimized_outline


def standardize_punclist(punc):
    """

    :param punc:
    :return:
    """
    if len(punc) == 0:
        return np.array([None] * 6)
    else:
        return punc


def filter_pundata(punc, threshold):
    if None in punc:
        return punc
    else:
        return standardize_punclist(punc[punc[:, 2] >= threshold])


def punc_string_parser(punc_data):
    """

    :param punc_data:
    :return:
    """
    punc = str(punc_data)
    new_list = []
    counter = 0
    substring = ''
    for x in str(punc):
        if x not in '1234567890e-+.':
            if counter != 0:
                new_list.append(substring)
                substring = ''
                counter = 0
        else:
            substring += x
            counter += 1
    new_list = np.array(new_list).reshape(int(len(new_list) / 6), 6).astype(float)
    return new_list


def adjust_relative_width(x, y, normx, normy):
    """

    :param x:
    :param y:
    :param normx:
    :param normy:
    :return:
    """
    p1 = np.sum((normx[np.newaxis, :] - y[:, np.newaxis]) < 0, axis=1)
    p0 = p1 - 1
    w = normy[p0] + (normy[p1] - normy[p0]) * (y - normx[p0]) / (normx[p1] - normx[p0])
    return x * w


def puncta_heatmaps(df, optimized_outline, padded, coords, coord2w_list,
                    channels=['TRITC', 'CY5'], targets=['1', '2'], cutoffs=[300, 100]):
    """

    :param df:
    :param optimized_outline:
    :param padded:
    :param coords:
    :param coord2w_list:
    :param channels:
    :param targets:
    :param cutoffs:
    :return:
    """
    from scipy.stats import gaussian_kde
    mRNA_loc = plt.figure(figsize=(4 * len(channels) + 0.2, 11.2))
    grids = gs(2, len(channels))
    for k, c in enumerate(channels):
        target = targets[k]
        cutoff = cutoffs[k]
        ax1 = mRNA_loc.add_subplot(grids[0, k])
        ax2 = mRNA_loc.add_subplot(grids[1, k])
        fg = np.concatenate(padded, axis=1)

        bg_scatter, bg_heatmap = np.zeros(fg.shape), np.zeros(fg.shape)
        ax1.imshow(bg_scatter, cmap='Greys', aspect='auto')
        ax1.set_xlim(0, bg_scatter.shape[1] - 1)
        for outline in optimized_outline:
            ax1.plot(outline.T[1], outline.T[0], c="black", alpha=1, lw=0.5)
            ax2.plot(outline.T[1], outline.T[0], c="white", alpha=1, lw=0.5)
        median_lengths = []
        for i, frac in enumerate([0, 0.2, 0.4, 0.6, 0.8]):
            l1 = df['length'].quantile(frac)
            l2 = df['length'].quantile(frac + 0.2)
            subset = df[(df['length'] < l2) & (df['length'] >= l1)]
            median_lengths.append(subset['length'].median())
            punc_list = []
            for punc in subset['{}_puncta_data'.format(c)]:
                punc = str(punc)
                if 'None' not in punc:
                    punc_list.append(punc_string_parser(punc))
            if len(punc_list) > 2:
                punc_list = np.vstack(punc_list)
                if len(punc_list) > 2:
                    punc_list = punc_list[punc_list[:, 2] >= cutoff]
            if len(punc_list) > 2:
                v, x, y = punc_list[:, np.array([2, 4, 5])].T
                norm_v = (v) / np.percentile(v, 85)
                norm_v[norm_v < 0] = 0
                norm_v[norm_v > 1] = 1
                normx, normy = coord2w_list[i]
                x0, y0, l, w = coords[4 - i]

                x = adjust_relative_width(x, y, normx, normy)
                # plot scatter
                ax1.scatter(x * w + x0, (y - 0.5) * l + y0, s=norm_v * 20, alpha=0.5, fc=cm.get_cmap('Blues')(norm_v))

                # plot heatmap
                data = np.array([x * w + x0, (y - 0.5) * l + y0]).T
                xgrid, ygrid, X, Y = construct_grids(x0, w, y0, l, 1)
                kernel = gaussian_kde(data.T, weights=v)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kernel(positions).T, X.shape)

                # normalize density
                Z /= Z.max()
                bg_heatmap[Y, X] = Z
        bg_heatmap[fg == 0] = 0
        ax2.imshow(bg_heatmap, cmap='magma', aspect='auto')
        ax1.set_xticks(np.array(coords)[:, 0])
        ax1.set_xticklabels(np.flip(np.round(median_lengths, 1)), fontsize=14)
        ax1.set_xlabel('median length [μm]', fontsize=14)
        ax2.set_xticks(np.array(coords)[:, 0])
        ax2.set_xticklabels(np.flip(np.round(median_lengths, 1)), fontsize=14)
        ax2.set_xlabel('median length [μm]', fontsize=14)
        ax1.set_title('{}-$\it{}$'.format(c, target), fontsize=14)
        ax1.set_yticks([])
        ax2.set_yticks([])
    return mRNA_loc


def construct_grids(x0, w, y0, l, gridsize=1):
    xmin = int(x0 - w * 0.5)
    xmax = int(x0 + w * 0.5)
    ymin = int(y0 - l * 0.5)
    ymax = int(y0 + l * 0.5)
    xgrid = np.arange(xmin, xmax, gridsize)
    ygrid = np.arange(ymin, ymax, gridsize)
    X, Y = np.meshgrid(xgrid, ygrid)
    return xgrid, ygrid, X, Y