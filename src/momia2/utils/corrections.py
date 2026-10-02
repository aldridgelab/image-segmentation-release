import os
import warnings
from .generic import *
from skimage import filters, transform
from scipy import fftpack
import numpy as np


"""
=================================== Section 1. xy drift correction ===================================

"""
def drift_detection(img, reference, upsample_factor=10):
    """
    wrapped around phase_cross_correlation in skimage's registration module
    """
    from skimage.registration import phase_cross_correlation
    shift, error, _diff = phase_cross_correlation(reference,
                                                  img,
                                                  upsample_factor=upsample_factor)
    return shift

def drift_correction(target_img, drift, max_drift=1000):
    """
    Corrects xy drift between phase contrast image and fluorescent image(s).

    Parameters
    ----------
    target_img : ndarray
        Input image
    drift : tuple
        Subpixel xy drift
    max_drift : int, optional
        Maximum allowed drift, by default 1000

    Returns
    -------
    ndarray
        Drift corrected image
    """
    import scipy.ndimage as ndi

    # get target image dtype
    img_dtype = target_img.dtype

    # Fourier shift the image by the given subpixel xy drift
    offset_image = ndi.fourier_shift(np.fft.fftn(target_img), drift)
    offset_image = np.fft.ifftn(offset_image)
    offset_image = np.round(offset_image.real)
    offset_image[offset_image <= 0] = 1

    # Rescale the image to avoid overflow
    img_dtype = target_img.dtype
    if max(np.abs(drift)) <= max_drift:
        if img_dtype == np.uint8:
            offset_image[offset_image >= 255] = 255
            return offset_image.astype(np.uint8)
        elif img_dtype == np.uint16:
            offset_image[offset_image >= 65530] = 65530
            return offset_image.astype(np.uint16)
        elif img_dtype == int:
            offset_image[offset_image >= 2147483647] = 2147483647
        elif np.issubdtype(img_dtype, np.floating):
            return offset_image.astype(img_dtype)
        elif img_dtype == np.uint12:
            offset_image[offset_image >= 4095] = 4095
            return offset_image.astype(np.uint12)
        else:
            print("Unsupported dtype: {}".format(img_dtype))
            return target_img
    else:
        return target_img

def get_xydrift(ref_img, target_img):
    from skimage import registration
    """
    Calculate the shift between two images using phase cross-correlation.

    Parameters
    ----------
    ref_img : ndarray
        Reference image.
    target_img : ndarray
        Target image.

    Returns
    -------
    shift : tuple
        Tuple of x and y shift values.

    Examples
    --------
    >>> shift = get_xydrift(ref_img, target_img)
    """
    shift, error, _diff = registration.phase_cross_correlation(ref_img, target_img, upsample_factor=10)
    return shift


def correct_xydrift_timeseries(timeseries,
                               ref_channel=-1,
                               order='XYTC',
                               verbose=False,
                               drift_correction_threshold=5):
    """
    Corrects xy drift of microscopy images in a time series.

    Parameters
    ----------
    timeseries : ndarray
        Input time series of images
    ref_channel : int, optional
        Reference channel for drift correction, by default 1
    order : str, optional
        Dimension order of the input time series, by default 'XYTC'
    verbose : bool, optional
        Whether to print progress messages, by default False
    drift_correction_threshold : int, optional
        Maximum allowed drift, by default 5

    Returns
    -------
    ndarray
        Drift corrected time series of images
    """

    from tqdm import tqdm
    if np.sum([(order.find(x) >= 0) * 1 for x in 'XYT']) == 3:
        if 'C' in order and len(order) == 4:
            new_order = 'TCXY'
        elif 'C' not in order and len(order) == 3:
            new_order = 'TXY'
        else:
            raise ValueError('image dimension order specified to be "{}", but {} dimensional image found.'.format(order,
                                                                                                                  len(timeseries.shape)))
    else:
        raise ValueError(
            'Illegal order found: {}. Dimension order should only consist to following letters: "X","Y","C","T".'.format(
                                                                                                                    order))

    order_dict = {x: i for i, x in enumerate(order)}
    tm = np.transpose(timeseries, tuple([order_dict[x] for x in new_order]))
    if len(new_order) == 3:
        tm = np.expand_dims(tm, axis=1)

    last_drift = np.array([0, 0])
    nframes = tm.shape[0]
    nchannels = tm.shape[1]
    max_driftx = 0
    max_drifty = 0

    drift_corrected = np.zeros(tm.shape)
    drift_corrected[0, :, :, :] = tm[0, :, :, :]

    if ref_channel >= nframes:
        raise ValueError('reference frame id (ref_frame) should be smaller than the number of channels of the image')

    if verbose:
        print('Correct for planar drift')
    for t in tqdm(range(1, nframes)):
        last_drift = last_drift + get_xydrift(tm[t - 1, ref_channel],
                                              tm[t, ref_channel])
        max_driftx = max(abs(last_drift[0]), max_driftx)
        max_drifty = max(abs(last_drift[1]), max_drifty)
        for c in range(nchannels):
            drift_corrected[t, c] = drift_correction(tm[t, c], last_drift,
                                                     drift_correction_threshold)
    max_driftx = int(max_driftx)
    max_drifty = int(max_drifty)

    cropped = drift_corrected[:, :, max_driftx:(drift_corrected.shape[2] - max_driftx),
              max_drifty:(drift_corrected.shape[3] - max_drifty)]
    if verbose:
        print('Drift correction complete')
        print('Output image shape is {}'.format(cropped.shape))
    return cropped


"""
=================================== Section 2. background correction ===================================

"""

class rolling_ball:
    """
    A class to create a rolling ball structuring element for background subtraction.

    :param radius: int, optional (default=50)
        The radius of the rolling ball.

    :ivar width: int
        The width of the rolling ball.
    :ivar shrink_factor: int
        The factor by which the ball is shrunk.
    :ivar arc_trim_per: int
        The percentage of the arc to be trimmed.
    :ivar radius: float
        The radius of the rolling ball divided by the shrink factor.
    :ivar ball: numpy.ndarray
        The rolling ball structuring element.
    """

    def __init__(self, radius=50):
        self.width = 0
        if radius <= 10:
            self.shrink_factor = 1
            self.arc_trim_per = 24
        elif radius <= 20:
            self.shrink_factor = 2
            self.arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            self.arc_trim_per = 32
        else:
            self.shrink_factor = 8
            self.arc_trim_per = 40
        self.radius = radius / self.shrink_factor
        self.build()

    def build(self):
        """
        Builds the rolling ball structuring element.
        """
        x_trim = int(self.arc_trim_per * self.radius / 100)
        half_width = int(self.radius - x_trim)
        self.width = int(2 * half_width + 1)
        r_squre = np.ones((self.width, self.width)) * (self.radius ** 2)
        squared = np.square(np.linspace(0, self.width - 1, self.width) - half_width)
        x_val = np.tile(squared, (self.width, 1))
        y_val = x_val.T
        self.ball = np.sqrt(r_squre - x_val - y_val)


def rolling_ball_bg_subtraction(data, **kwargs):
    """
        Subtracts background from an image using rolling ball algorithm.

        Parameters
        ----------
        data : numpy.ndarray
            The input image.
        rolling_ball_radius : int, optional
            The radius of the rolling ball. Default is 40.
        **kwargs : dict
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        numpy.ndarray
            The background-subtracted image.
    """
    rolling_ball_radius = kwarg_or_default('rolling_ball_radius', kwargs, 40)
    output_data = data.copy()
    smoothed = filters.gaussian(data, sigma=1) * 65535
    ball = rolling_ball(radius=rolling_ball_radius)
    shrunk_img = shrink_img_local_min(smoothed, shrink_factor=ball.shrink_factor)
    bg = rolling_ball_bg(shrunk_img, ball.ball)
    bg_rescaled = transform.resize(bg, data.shape, anti_aliasing=True)
    output_data = output_data - bg_rescaled
    output_data[output_data <= 0] = 0
    return output_data.astype(np.uint16)


def shrink_img_local_min(image, shrink_factor=4):
    s = shrink_factor
    r, c = image.shape[0], image.shape[1]
    r_s, c_s = int(r / s), int(c / s)
    shrunk_img = np.ones((r_s, c_s))
    for x in range(r_s):
        for y in range(c_s):
            shrunk_img[x, y] = image[x * s:x * s + s, y * s:y * s + s].min()
    return shrunk_img


def rolling_ball_bg(image, ball):
    width = ball.shape[0]
    radius = int(width / 2)
    r, c = image.shape[0], image.shape[1]
    bg = np.ones((r, c)).astype(np.float32)
    # ignore edges to begin with
    for x in range(r):
        for y in range(c):
            x1, x2, y1, y2 = max(x - radius, 0), min(x + radius + 1, r), max(y - radius, 0), min(y + radius + 1, c)
            cube = image[x1:x2, y1:y2]
            cropped_ball = ball[radius - (x - x1):radius + (x2 - x), radius - (y - y1):radius + (y2 - y)]
            bg_cropped = bg[x1:x2, y1:y2]
            bg_mask = ((cube - cropped_ball).min()) + cropped_ball
            bg[x1:x2, y1:y2] = bg_cropped * (bg_cropped >= bg_mask) + bg_mask * (bg_cropped < bg_mask)
    return (bg)


def subtract_background(data, method='rolling_ball', **kwargs):
    """
    Function to subtract the background from an image.

    :param data: numpy.ndarray
            The input image.
    :param method: str, optional (default='rolling_ball')
            The method used for background subtraction.
            Currently, only 'rolling_ball' is supported.
    :param kwargs: dict
            Additional keyword arguments to be passed to the function.
    :return: numpy.ndarray
            The background-subtracted image.
    :raises ModuleNotFoundError: If the specified method is not found.
    """
    if method == 'rolling_ball':
        return rolling_ball_bg_subtraction(data, **kwargs)
    else:
        raise ModuleNotFoundError('Method {} not found!'.format(method))


"""
=================================== Section 3. Bounding box optimization ===================================
"""

def touching_edge(img_shape, optimized_bbox):
    """
    Check if the bounding box is touching the edge of the image.

    :param img_shape: The shape of the image.
    :param optimized_bbox: The bounding box coordinates.
    :return: True if the bounding box is touching the edge of the image, False otherwise.
    """

    (rows, columns) = img_shape
    (x1, y1, x2, y2) = optimized_bbox
    if min(x1, y1, rows - x2 - 1, columns - y2 - 1) <= 0:
        return True
    else:
        return False


EDGE_WIDTH_DEFAULT = int(os.getenv("TOUCHING_EDGE_WIDTH", "8"))
# Min/Max area for mask cleanup (matches filter defaults; overridden here for transparency)
MIN_PARTICLE_SIZE_DEFAULT = int(os.getenv("MIN_PARTICLE_SIZE", "50"))
MIN_HOLE_SIZE_DEFAULT = int(os.getenv("MIN_HOLE_SIZE", "50"))


def optimize_bbox(img_shape,
                  bbox,
                  edge_width=None):
    """
    function to optimize bounding box
    :param img_shape: shape of the image to be mapped onto
    :param bbox: inital bbox
    :param edge_width: max edge width
    :return:
    """
    (rows, columns) = img_shape
    (x1, y1, x2, y2) = bbox
    if edge_width is None:
        edge_width = EDGE_WIDTH_DEFAULT

    return max(0, x1 - edge_width), max(0, y1 - edge_width), min(rows - 1, x2 + edge_width), min(columns - 1,
                                                                                                 y2 + edge_width)


def optimize_bbox_batch(img_shape,
                        region_prop_table,
                        edge_width=None):
    """
    function to optimize bounding box
    :param img_shape: shape of the image to be mapped onto
    :param bbox: inital bbox
    :param edge_width: max edge width
    :return:
    """

    (rows, columns) = img_shape
    if edge_width is None:
        edge_width = EDGE_WIDTH_DEFAULT
    if 'bbox-0' in region_prop_table.columns:
        (x1, y1, x2, y2) = region_prop_table[['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']].values.T
    elif '$bbox-0' in region_prop_table.columns:
        (x1, y1, x2, y2) = region_prop_table[['$bbox-0', '$bbox-1', '$bbox-2', '$bbox-3']].values.T
    else:
        raise ValueError('Bbox columns should be "bbox-n" or "$bbox-n" where n is 0, 1, 2, or 3.')
    new_x1, new_y1, new_x2, new_y2 = x1 - edge_width, y1 - edge_width, x2 + edge_width, y2 + edge_width
    new_x1[new_x1 <= 0] = 0
    new_y1[new_y1 <= 0] = 0
    new_x2[new_x2 >= rows - 1] = rows - 1
    new_y2[new_y2 >= columns - 1] = columns - 1

    touching_edge = (((x1 <= 0.5 * edge_width) + (y1 <= 0.5 * edge_width) + \
                      (x2 >= rows - 1 - 0.5 * edge_width) + (y2 >= columns - 1 - 0.5 * edge_width)) > 0).astype(int)
    return np.array([new_x1, new_y1, new_x2, new_y2, touching_edge]).T


# Expose tweakable defaults for reference/documentation
TWEAKABLE_DEFAULTS = {
    "TOUCHING_EDGE_WIDTH": EDGE_WIDTH_DEFAULT,
    "MIN_PARTICLE_SIZE": MIN_PARTICLE_SIZE_DEFAULT,
    "MIN_HOLE_SIZE": MIN_HOLE_SIZE_DEFAULT,
}


"""
=================================== Section 4. Bandpass filters ===================================
"""


def fft(img, subtract_mean=True):
    """
    Perform the 2-dimensional Fast Fourier Transform (FFT) on an input image.
    :param img: ndarray, the input image.
    :param subtract_mean: bool, if True the mean of the image is subtracted before transforming, default is True.
    :return: ndarray, the FFT transformed image.
    """
    warnings.filterwarnings("ignore")
    if subtract_mean:
        img = img - np.mean(img)
    return fftpack.fftshift(fftpack.fft2(img))


def fft_reconstruction(fft_img, bandpass_filters):
    """
    Reconstruct an image after applying FFT bandpass filtering.

    :param fft_img: ndarray, the FFT transformed image.
    :param bandpass_filters: list, the low/high frequency bandpass filters.
    :return: ndarray, the bandpass filtered, restored phase contrast image.
    """

    warnings.filterwarnings("ignore")
    if len(bandpass_filters) > 0:
        for f in bandpass_filters:
            try:
                fft_img *= f
            except:
                raise ValueError("Illegal input filter found, shape doesn't match?")
    output = fftpack.ifft2(fftpack.ifftshift(fft_img)).real
    output -= output.min()
    return output


def make_centered_mesh(pixel_microns=0.065,
                       img_width=2048,
                       img_height=2048):
    """
    Create a centered meshgrid in the frequency domain.
    modified to make generic, 20210821
    :param pixel_microns: float, the unit length of a pixel.
    :param img_width: int, the width of the image in pixels.
    :param img_height: int, the height of the image in pixels.
    :return: ndarray, the centered mesh.
    """

    # create mesh
    u_max = round(1 / pixel_microns, 3) / 2
    v_max = round(1 / pixel_microns, 3) / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, img_width)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, img_height)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    centered_mesh = np.sqrt(u_mat ** 2 + v_mat ** 2)
    return centered_mesh


def bandpass_filter(centered_mesh, bandpass_width, highpass=True):
    """
    Create a discrete bandpass filter.

    :param centered_mesh: ndarray, the centered meshgrid.
    :param bandpass_width: float, the width of the bandpass.
    :param highpass: bool, if True a highpass filter is created, otherwise a lowpass filter is created.
    :return: ndarray, the bandpass filter.
    """
    shape = centered_mesh.shape
    if bandpass_width == 0:
        return np.ones((shape[1], shape[0]))
    elif highpass:
        return np.e ** (-(centered_mesh * bandpass_width) ** 2)
    else:
        return 1 - np.e ** (-(centered_mesh * bandpass_width) ** 2)


def dual_bandpass(img,
                  pixel_microns=0.065,
                  min_structure_scale=0.2,
                  max_structure_scale=20):
    """
    Apply a dual-bandpass filter to an image.

    :param img: ndarray, the input image.
    :param pixel_microns: float, the unit length of a pixel.
    :param min_structure_scale: float, the minimum structure size (in microns) for the highpass filter.
    :param max_structure_scale: float, the maximum structure size (in microns) for the lowpass filter.
    :return: ndarray, the image after applying the dual-bandpass filter.

    :raises ValueError: if the min_structure_scale is greater or equal to max_structure_scale.
    """

    if min_structure_scale >= max_structure_scale:
        raise ValueError('Minimal structure size (highpass) should be smaller \nthan maximal structure size (lowpass)')

    fft_img = fft(img)
    img_height, img_width = img.shape
    freq_domain = make_centered_mesh(pixel_microns,
                                     img_width=img_width,
                                     img_height=img_height)
    highpass_filter = bandpass_filter(freq_domain, min_structure_scale, True)
    lowpass_filter = bandpass_filter(freq_domain, max_structure_scale, False)

    # dual-bandpass filter
    filtered = fft_reconstruction(fft_img, (highpass_filter, lowpass_filter))
    return filtered.astype(img.dtype)


"""
=================================== Section 5. Future functions ===================================
"""
