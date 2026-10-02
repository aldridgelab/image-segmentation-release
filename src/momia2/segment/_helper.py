import numpy as np
import pandas as pd
import tifffile
import read_roi as rr
from skimage import filters, measure
from skimage.morphology import disk, remove_small_objects, binary_opening, binary_dilation, binary_erosion
from ..utils import linalg,generic

__all__ = ['roi2multilabel','roi2multimasks',
           'prediction2seed','prediction2foreground','np_method',
           'dist2labels_simp','dimension_scaler']

def roi2multilabel(roi_file, image_shape, dst, erosion_radius=2, subdivide_degree=2):
    from skimage import draw
    from skimage.measure import subdivide_polygon
    roi = rr.read_roi_zip(roi_file)
    canvas = np.zeros(image_shape)
    core = np.zeros(image_shape)
    for k, v in roi.items():
        xy = np.array([v['y'], v['x']]).T
        xy = subdivide_polygon(xy, degree=subdivide_degree, preserve_ends=True)
        mask = draw.polygon2mask(image_shape, xy)
        eroded = binary_erosion(mask, disk(erosion_radius))
        core[eroded == 1] = 1
    dilated = binary_dilation(core, disk(erosion_radius + 1))
    canvas[(dilated - core) > 0] = 125
    canvas[core > 0] = 254
    tifffile.imwrite(dst, canvas.astype(np.uint8), imagej=True)


def roi2multimasks(roi_file, image_file, dst, erosion_radius=2, subdivide_degree=2):
    from skimage import draw, measure
    from skimage.segmentation import watershed
    from skimage.measure import subdivide_polygon
    roi = rr.read_roi_zip(roi_file)
    image = tifffile.imread(image_file)
    image_shape = image.shape
    canvas = np.zeros(image_shape)
    core = np.zeros(image_shape)
    for k, v in roi.items():
        xy = np.array([v['y'], v['x']]).T
        xy = subdivide_polygon(xy, degree=subdivide_degree, preserve_ends=True)
        mask = draw.polygon2mask(image_shape, xy)
        eroded = binary_erosion(mask, disk(erosion_radius))
        core[eroded == 1] = 1
        canvas[mask == 1] = 1
    labeled = watershed(image, markers=measure.label(core),
                        mask=canvas, watershed_line=False, compactness=100)
    tifffile.imwrite(dst, labeled.astype(np.uint16), imagej=True)


def prediction2seed(pred_multilabel_mask,
                    seed_min=0.3,
                    edge_max=0.75,
                    min_seed_size=20,
                    opening_radius=1):
    seed = (pred_multilabel_mask[:, :, 2] > seed_min) * (pred_multilabel_mask[:, :, 1] < edge_max) * 1
    if isinstance(opening_radius, int) and opening_radius > 0:
        seed = binary_opening(seed, disk(opening_radius))
    seed = remove_small_objects(seed, min_seed_size)
    return measure.label(seed)


def dist2labels_simp(dist, mask,
                     dist_threshold=0.25,
                     mask_threshold=0.8,
                     opening=2,
                     min_particle_size=10,
                     watershedline=False):
    from skimage import measure, segmentation
    basin = 1 - dist
    seed = measure.label(dist > dist_threshold)
    seed = remove_small_objects(seed, min_size=min_particle_size)
    binary_mask = mask > mask_threshold
    if int(opening) > 0:
        binary_mask = binary_opening(binary_mask, disk(int(opening)))
    watersheded = segmentation.watershed(basin,
                                         markers=seed,
                                         mask=binary_mask,
                                         compactness=100,
                                         watershed_line=watershedline)
    return watersheded


def prediction2foreground(pred_multilabel_mask, channel=0,
                          threshold=0.4,
                          erosion_radius=1):
    fg = (pred_multilabel_mask[:, :, channel] < threshold) * 1
    # fg=morphology.binary_closing(fg,morphology.disk(1))
    if isinstance(erosion_radius, int) and erosion_radius > 0:
        fg = binary_erosion(fg, disk(erosion_radius))
    return fg * 1


def compute_dist(multi_mask, smooth_factor=10, max_v=7):
    from skimage import measure
    import edt
    from scipy.ndimage import median_filter

    canvas = np.zeros(multi_mask.shape)

    for i in np.unique(multi_mask):
        if i > 0:
            u_mask = (multi_mask == i) * 1
            xy = np.array(np.where(u_mask > 0)).T
            c = measure.find_contours(median_filter(multi_mask == i, 3), level=0.5)
            if len(c) == 1:
                smoothed = linalg.spline_approximation(c[0], smooth_factor=smooth_factor, n=len(c[0]))
                dis = np.min(linalg.distance_matrix(xy, smoothed), axis=1)
            else:
                dis = edt.edt(u_mask)[xy[:, 0], xy[:, 1]]
            canvas[xy[:, 0], xy[:, 1]] = dis
    return generic.normalize_image(canvas, min_perc=0, max_perc=100, min_v=0, max_v=7)


def np_method(data, method='mean', **kwargs):
    if method in ['default', 'mean', 'average', 'MEAN', 'Mean', 'Average']:
        return np.mean(data, **kwargs)
    elif method in ['median', 'Median', 'MEDIAN']:
        return np.median(data, **kwargs)
    elif method in ['Max', 'max', 'MAX']:
        return np.max(data, **kwargs)
    elif method in ['Min', 'min', 'MIN']:
        return np.min(data, **kwargs)
    else:
        print('method not found, use np.mean instead')
        return np.mean(data, **kwargs)


def dimension_scaler(image, axis=(0, 3)):
    """
    image used for segment is reformated as a four-dimensional tensor with each dimension representing:
    #0 - number of classes
    #1,2 - 2D image
    #3 - number of channels (e.g. this should be 3 for RGB images)

    if class labels are provided for a single channel image, the axis paramter should be (3).
    if a multi-channel image is provided with no class labesl, the axis paramter should be (0).
    """

    if len(image.shape) == 4:
        return image
    elif len(image.shape) == 3 and len(axis) == 1:
        return np.expand_dims(image, axis=axis)
    elif len(image.shape) == 2 and len(axis) == 2:
        return np.expand_dims(image, axis=axis)
    else:
        raise ValueError("Image shape and provided axis information don't match")