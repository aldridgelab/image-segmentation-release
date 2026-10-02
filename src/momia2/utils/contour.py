from .generic import *
from .linalg import *

"""
============================= Description =============================
contour/outline manipulation
"""

def find_contour_marching_squares(binary_mask,
                                  image = None,
                                  level=0.1,
                                  dilation=False,
                                  erosion=True,
                                  dark_background=False,
                                  sigma=1,
                                  approximate=False,
                                  tolerance=0.3,
                                  interp_distance=1,
                                  min_segment_count=3,
                                  ):
    """
    finds the outermost contour of a binary mask using the marching-squares algorithm.
    :param binary_mask: binary mask of the object
    :param image: optional bright-field image to use as reference
    :param level: threshold level for contour detection
    :param dilation: boolean flag to perform binary dilation on the mask
    :param erosion: boolean flag to perform binary erosion on the mask
    :param dark_background: boolean flag to indicate if the image has a dark background
    :param sigma: sigma value for Gaussian smoothing of the image
    :return: outermost contour of the object
    """
    from skimage import morphology, filters, measure

    # smoothen mask
    binary_mask = smooth_binary_mask(binary_mask.astype(int),sigma=1)

    # use local threshold to suppress shadow signals (usually from cell debris)
    dil_mask = morphology.binary_dilation(binary_mask,morphology.disk(2))
    masked_pixels = image[dil_mask == 1].flatten()
    if len(masked_pixels) == 0:
        # Return empty contour if no valid pixels
        return np.array([])
    th = filters.threshold_otsu(masked_pixels)
    binary_mask = dil_mask * (image < th)

    # erode or dilate
    if erosion:
        binary_mask = morphology.binary_erosion(binary_mask).astype(int)
    if dilation:
        binary_mask = morphology.binary_dilation(binary_mask).copy()

    # use the bright-field image as reference if available:
    if image is not None:
        if not identical_shapes([image, binary_mask]):
            raise ValueError('Input images are of different sizes!')
        if not dark_background:
            image = gaussian_smooth(invert_normalize(image),sigma=sigma)
        else:
            image = normalize_image(gaussian_smooth(image,sigma=sigma),min_perc=0,max_perc=0.995)
        intensity_mask = binary_mask*image
    else:
        intensity_mask = binary_mask
    contours = measure.find_contours(intensity_mask, level=level)
    if len(contours) == 0:
        return np.array([])
    contour = contours[0]
    if approximate:
        contour = simplify_polygon(contour,
                                   tolerance=tolerance,
                                   interp_distance=interp_distance,
                                   min_segment_count=min_segment_count)
        contour = spline_approximation(contour,n=len(contour),smooth_factor=1,closed=True)
    return contour


def estimate_outline_length(closed_coords):
    max_L = 0
    for i in range(len(closed_coords)):
        for j in range(i+1,len(closed_coords)):
            x1,y1 = closed_coords[i]
            x2,y2 = closed_coords[j]
            dist =  np.sqrt((x2-x1)**2+(y2-y1)**2)
            if dist > max_L:
                max_L = dist
    return max_L
