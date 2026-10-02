import numpy as np
from skimage import morphology, filters, measure
from .linalg import *


"""
===================================. outlier detection .================================
"""
def zscore(data):
    """
    Calculates the Z-score (standard score, https://en.wikipedia.org/wiki/Standard_score) of input data
    :param data: list or numpy array, input n-D array
    :return: numpy array, calculated Z-score
    """
    data = np.array(data)
    return (data - data.mean()) / data.std()


def remove_outliers(df, feature='length', zcutoff=3):
    """
    :param df: pandas DataFrame
    :param feature: str, column name of the feature to remove outliers from
    :param zcutoff: float, the cutoff value for the Z-score.
                    Observations with Z-score greater than this value will be considered outliers.
    :return: pandas DataFrame, with outliers removed from the specified feature
    """
    if feature not in df:
        raise ValueError('Feature {} not found!'.format(feature))
    df_copied = df.copy().reset_index(drop=True)
    data = df[feature].values
    zscores = zscore(data)
    value_in_range = np.where(np.abs(zscores) < zcutoff)[0]
    df_filtered = df_copied.iloc[value_in_range].copy().sort_values(by=feature).reset_index(drop=True)
    return df_filtered


def remove_outliers_batch(df, features=None, zcutoffs=None):
    """
    Removes outliers from multiple features of a pandas DataFrame using the remove_outliers function.
    :param df: pandas DataFrame
    :param features: list of str, column names of the features to remove outliers from
    :param zcutoffs: list of float, the cutoff values for the Z-scores. Observations with Z-scores greater than these values will be considered outliers.
    :return: pandas DataFrame, with outliers removed from the specified features
    """

    if features is None:
        features = ['length']

    if zcutoffs is None:
        zcutoffs = [3]

    if len(features) != len(zcutoffs):
        raise ValueError("Feature and z-cutoff lists don't match")

    df_copy = df.copy()
    for f, z in zip(features, zcutoffs):
        df_copy = remove_outliers(df_copy, f, z)
    return df_copy

def find_outliers(df, feature='length', zcutoff=3):
    """
    Finds outliers in a pandas dataframe column using the Z-score method
    :param df: pandas dataframe, input dataframe
    :param feature: str, name of the column to find outliers in
    :param zcutoff: float, Z-score cutoff value
    :return: numpy array, binary array indicating whether each row is an outlier or not
    """
    if feature not in df:
        raise ValueError('Feature {} not found!'.format(feature))
    data = df[feature].values
    zscores = zscore(data)
    return (zscores > zcutoff).astype(int)


def find_outliers_batch(df, features=None, zcutoffs=None):
    """
    Finds outliers in multiple pandas dataframe columns using the Z-score method
    :param df: pandas dataframe, input dataframe
    :param features: list of str, names of the columns to find outliers in
    :param zcutoffs: list of float, Z-score cutoff values for each column
    :return: numpy array, binary array indicating whether each row is an outlier or not
    """
    if features is None:
        features = ['length']

    if zcutoffs is None:
        zcutoffs = [3]

    if len(features) != len(zcutoffs):
        raise ValueError("Feature and z-cutoff lists don't match")
    zscore_sum = np.zeros(len(df))
    for f, z in zip(features, zcutoffs):
        zscore_sum += find_outliers(df, f, z)
    return (zscore_sum>0).astype(int)


"""
===================================. CV helper .================================
"""

def gaussian_smooth(img,sigma=1,preserve_range=True):
    """
    wrapper of skimage.filters.gaussian, preserves range by default
    :param img: input image
    :param sigma: gaussian kernel sigma
    :return: gaussian smoothed image
    """
    return filters.gaussian(img,sigma=sigma,preserve_range=preserve_range)

def normalize_image(img,
                    mask=None,
                    min_perc=0.5,
                    max_perc=99.5,
                    min_v=0,
                    max_v=30000,
                    bg=0.5):
    """
    inherited from @kevinjohncutler with modifications
    perform either percentile based normalization or
    fixed range normalization or
    masked normalizatoin.

    For masked normalization:
    @ kevinjohncutler
    Normalize image by rescaling from 0 to 1 and then adjusting gamma to bring
    average background to specified value (0.5 by default).

    :params img: input two-dimensional image
    :params mask: input labels or foreground mask or anything but not none
    :params min_perc: lower bound of the percentile norm (0-100)
    :params max_perc: higher bound of the percentile norm (0-100)
    :params min_v: lower bound of the absolute value for normalization, overwrite min_perc,
    :params max_v: higher bound of the absolute value for normalization, overwrite max_perc,
    :params bg: background value in the range 0-1
    :return: gamma-normalized array with a minimum of 0 and maximum of 1
    """
    if mask is None:
        th1, th2 = np.percentile(img, min_perc), np.percentile(img, max_perc)
        if min_v is not None:
            th1 = min_v
        if max_v is not None:
            th2 = max_v
        img = (img - th1) / (th2 - th1)
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    else:
        img = (img - img.min()) / (img.max() - img.min())
        try:
            img = img ** (np.log(bg) / np.log(np.mean(img[morphology.binary_erosion(mask == 0)])))
        except:
            # just in case when mask is invalid
            mask = img < filters.threshold_isodata(img)
            img = img ** (np.log(bg) / np.log(np.mean(img[morphology.binary_erosion(mask == 0)])))

    return img


def smooth_binary_mask(mask, sigma=1.5):
    """
    Applies a Gaussian filter to a binary mask for smoothing.
    :param mask: 2D numpy array representing the binary mask.
    :param sigma: Float representing the standard deviation for Gaussian kernel. Default value is 1.5.
    :return: 2D numpy array of the smoothed binary mask where mask values > 0.5 are considered True (or 1) and those <= 0.5 are considered False (or 0).
    """
    smooth_mask = filters.gaussian(mask*1, sigma=sigma,preserve_range=True)
    return smooth_mask > 0.5


def invert_normalize(data, max_percentile=99):
    """
    invert then adjust the image drange to 0-1
    :param data: image data, usually masked
    :param max_percentile: percentile maximum for normalization
    :return: inverted, normalized data
    """
    max_val = np.percentile(data, max_percentile)
    inverted = max_val - data
    inverted[inverted <= 0] = 0
    normalized = inverted / inverted.max()
    return normalized


def percentile_normalize(data, perc1=5, perc2=95, fix_lower_bound=500, fix_higher_bound=None):
    q1, q2 = np.percentile(data, perc1), np.percentile(data, perc2)
    if fix_lower_bound is not None:
        q1 = min(q1, fix_lower_bound)
    if fix_higher_bound is not None:
        q2 = max(q2, fix_higher_bound)
    float_d = data.copy().astype(float)
    transformed = (float_d - q1) / (q2 - q1)
    transformed[transformed > 1] = 1
    transformed[transformed < 0] = 0
    return transformed


def adjust_image(img, dtype=16, adjust_gamma=True, gamma=1):
    from skimage import exposure
    """
    adjust image data depth and gamma value
    :param img: input image
    :param dtype: bit depth, 8, 12 or 16
    :param adjust_gamma: whether or not correct gamma
    :param gamma: gamma value
    :return: adjusted image
    """
    n_range = (0,255)
    if isinstance(dtype, int) & (dtype > 2):
        n_range = (0, 2 ** dtype - 1)
    else:
        print("Illegal input found where an integer no less than 2 was expected.")
    outimg = exposure.rescale_intensity(img, out_range=n_range)
    if adjust_gamma:
        outimg = exposure.adjust_gamma(outimg, gamma=gamma)
    return outimg


def coord2mesh(x, y, selem,
               xmax=10000000,
               ymax=10000000):
    """

    :param x:
    :param y:
    :param selem:
    :param xmax:
    :param ymax:
    :return:
    """
    dx, dy = np.where(selem)
    dx -= int(selem.shape[0] / 2)
    dy -= int(selem.shape[1] / 2)
    x_mesh = x[:, np.newaxis] + dx[np.newaxis, :]
    x_mesh[x_mesh < 0] = 0
    x_mesh[x_mesh > xmax - 1] = xmax - 1
    y_mesh = y[:, np.newaxis] + dy[np.newaxis, :]
    y_mesh[y_mesh < 0] = 0
    y_mesh[y_mesh > ymax - 1] = ymax - 1
    return x_mesh, y_mesh


def simplify_polygon(polygon,
                     tolerance=0.95,
                      interp_distance=1,
                      min_segment_count=2):
    """
    Simplifies a given polygon using the Ramer-Douglas-Peucker algorithm and linear interpolation.

    :param polygon: A 2D numpy array of shape (n, 2) representing the polygon.
    :param tolerance: A float representing the maximum distance for a point to be considered as a candidate for reduction, default value is 0.95.
    :param interp_distance: A float representing the distance between interpolated points, default value is 1.
    :param min_segment_count: An integer representing the minimum number of interpolation points, default value is 2.
    :return: A 2D numpy array of the simplified and interpolated polygon.
    """
    from skimage.measure import approximate_polygon
    if (polygon[0]-polygon[-1]).sum() == 0:
        closed = True
    else:
        closed = False
    approximation = approximate_polygon(polygon,tolerance=tolerance)
    linear_interpolated_segments=[]
    for i in range(len(approximation)-1):
        p1 = approximation[i]
        p2 = approximation[i+1]
        dist = distance(p1,p2)
        steps = max(int(round(dist/interp_distance)),min_segment_count)
        linear_interpolated_segments.append(np.array([np.linspace(p1[0],p2[0],steps),np.linspace(p1[1],p2[1],steps)]).T[:-1])
    linear_interpolated_segments.append(np.array([approximation[-1]]))
    approximated = np.concatenate(linear_interpolated_segments)
    if int(measure_length(approximated)/interp_distance) >= 5:
        approximated = spline_approximation(approximated,
                                            n=int(measure_length(approximated)/interp_distance),
                                            closed=closed,smooth_factor=0)
    return approximated


def dilate_by_coords(coords, image_shape,
                     selem=morphology.disk(2)):
    h, w = image_shape
    if len(selem) == 1:
        return coords
    elif len(selem) > 1:
        x, y = coord2mesh(coords[:, 0],coords[:, 1],selem=selem)
        x[x < 0] = 0
        x[x > h - 1] = h - 1
        y[y < 0] = 0
        y[y > w - 1] = w - 1
        xy = np.unique(np.array([x.ravel(), y.ravel()]).T, axis=0)
        return xy


def get_neighbors(labeled_mask, label):
    """
    Retrieves the neighboring labels around a given label in a labeled image mask.

    :param labeled_mask: 2D ndarray where each unique integer represents a unique region
    :type labeled_mask: numpy.ndarray
    :param label: The label for which to find neighbors
    :type label: int
    :return: List of unique neighboring labels
    :rtype: list
    """
    _x, _y = np.where(labeled_mask == label)
    x, y = dilate_by_coords(np.array([_x, _y]).T, labeled_mask.shape).T
    return [i for i in np.unique(labeled_mask[x, y]) if i not in [0, label]]


"""
===================================. miscellaneous .================================
"""
def dict2list(dict_f):
    """
    This function takes a dictionary as input and returns a list of lists,
    where each inner list contains a key-value pair from the dictionary.

    :param dict_f: input dictionary
    :return: output list
    """
    return [[k, v] for k, v in dict_f.items()]


def identical_shapes(image_list):
    """
    Check if all images in the list have identical shapes.
    :param image_list: A list of numpy arrays representing images.
    :return: True if all images have identical shapes, False otherwise.
    """
    x, y = 0, 0
    shape_matched = True
    for i, img in enumerate(image_list):
        if i == 0:
            x, y = img.shape
        else:
            if x != img.shape[0] or y != img.shape[1]:
                shape_matched = False
    return shape_matched


def value_constrain(value, min_value, max_value):
    """
    Keeps the value between a user defined range.

    :param value: The value to be constrained.
    :param min_value: The minimum value.
    :param max_value: The maximum value.
    :return: The constrained value.
    """
    return max(min_value, min(value, max_value))


def in_range(val, min_val, max_val):
    """
    check whether a value is within a user defined range
    :param val:
    :param min_val:
    :param max_val:
    :return:
    """
    if (val < min_val) or (val > max_val):
        return False
    else:
        return True


def norm_ratio(measure_of_two):
    """
    Calculate the normalized ratio of two values.
    Basically it does: min(measure_of_two) / max(measure_of_two)

    :param measure_of_two: The two values to be normalized.
    :return: The normalized ratio of the two values.
    """
    sorted_v = np.sort(measure_of_two)
    return sorted_v[0] / sorted_v[1]


def kwarg_or_default(key, kwarg, default):
    """
    Get the value of a keyword argument or return a default value.

    :param key: str, the key of the keyword argument.
    :param kwarg: dict, the dictionary of keyword arguments.
    :param default: The default value to return if the key is not in the dictionary.
    :return: The value of the keyword argument or the default value.
    """
    if key in kwarg:
        return kwarg[key]
    else:
        return default



def reorient_data(data, skew='left'):
    skewed_direction = 'right'
    interpolated = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(data)), data)
    if interpolated[:50].mean() > interpolated.mean():
        skewed_direction = 'left'
    if skewed_direction == skew:
        return False
    else:
        return True


def current_datetime():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    return dt_string


def rank2group(n, n_group=50):
    """
    Divides a range of numbers into groups. The numbers range from 0 to n-1.
    The groups are created in a way to divide the range into equal parts as much as possible.

    :param n: The total number of elements
    :type n: int, optional
    :param n_group: The total number of groups
    :type n_group: int, optional
    :return: A list of numpy arrays where each array represents a group of indices
    :rtype: list
    """

    ext = int(0.5 * (n % n_group))
    step = int(n / n_group)
    indices = []
    for i in range(n_group):
        group = np.arange(i * step, i * step + step) + ext
        group = group[group < n - 1]
        indices.append(group)
    return indices


def min_max(data):
    """
    Min-max normalization of n-D data
    :param data: list or numpy array.
    :return: min-max normalized numpy array.
    """
    data = np.array(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def correlation_map(x, y):
    """
    Correlate each n with each m.
    @abcd
    @https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30145770#30145770
    :param x: numpy array, Shape N X T.
    :param y: numpy array, Shape M X T.
    :return: numpy array, N X M array in which each element is a correlation coefficient.
    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n-1)
    s_y = y.std(1, ddof=n-1)
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis],mu_y[np.newaxis, :])
    return cov/np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def dist_prob(x, d=0.1, t=1, a=0.1):
    """
    Gaussian distance probability distribution

    :param x: float, input value
    :param d: float, diffusion coefficient (default=0.1)
    :param t: float, time (default=1)
    :param a: float, constant (default=0.1)
    :return: float, probability density
    """
    return (1 / np.sqrt(4 * np.pi * t * d)) * np.exp(-(a * x) ** 2 / (4 * d * t))


def difference_matrix(v1, v2, ref=0):
    """
    Computes the normalized difference matrix between two vectors.
    The normalized difference is computed as the absolute difference divided by the mean.
    This results in a similarity score where a score of 1 means the vectors are identical,
    and a score of 0 means they are completely dissimilar.

    :param v1: First input vector
    :type v1: numpy.ndarray
    :param v2: Second input vector
    :type v2: numpy.ndarray
    :param ref: Reference value to initialize the difference matrix, default is 0
    :type ref: int or float, optional
    :return: Normalized difference matrix
    :rtype: numpy.ndarray
    """
    diff = np.abs(v1[:, np.newaxis, :] - v2[np.newaxis, :, :])
    mean = (v1[:, np.newaxis, :] + v2[np.newaxis, :, :]) / 2
    norm_similarity = 1 - np.mean(diff / mean, axis=-1)
    norm_similarity[norm_similarity < 0] = 0
    return norm_similarity

