from .generic import *
from scipy import stats, ndimage as ndi
import cv2
import warnings
import pandas as pd

__all__ = ['multiscale_image_feature',
           'local_stat',
           'ratio_of_gaussian',
           'normalized_difference_of_gaussian',
           'gabor_kernel_bank',
           'basic_intensity_features',
           'sinuosity']

def multiscale_image_feature(image,
                             sigmas=(0.5, 0.8, 1.5, 3.5),
                             rog_sigmas=(0.5, 5),
                             num_workers=None,
                             mode='reflect',
                             order='rc',
                             shapeindex=True,
                             sobel=True,
                             ridge=True,
                             rog=True,
                             cval=0,
                             ext_kernels={}):
    """
    Computes multiscale features for an image including Gaussian filtered images, shape index, Sobel edges, ridges, etc.

    :param image: Input image
    :type image: numpy.ndarray
    :param sigmas: The standard deviations for the Gaussian filter
    :type sigmas: tuple, optional
    :param rog_sigmas: The standard deviations for the ratio of Gaussians (RoG)
    :type rog_sigmas: tuple, optional
    :param num_workers: The number of worker threads for parallel processing
    :type num_workers: int, optional
    :param mode: How to handle the borders for the filters
    :type mode: str, optional
    :param order: The order for the axes (reversed by default)
    :type order: str, optional
    :param shapeindex: Whether to compute the shape index or not
    :type shapeindex: bool, optional
    :param sobel: Whether to compute the Sobel edges or not
    :type sobel: bool, optional
    :param ridge: Whether to compute the ridges or not
    :type ridge: bool, optional
    :param rog: Whether to compute the ratio of Gaussians or not
    :type rog: bool, optional
    :param cval: The constant value for padding the image before filtering
    :type cval: int, optional
    :param ext_kernels: Additional kernels to filter the image with
    :type ext_kernels: dict, optional
    :return: A dictionary of Gaussian filtered images and a dictionary of image features
    :rtype: dict, dict
    """

    from concurrent.futures import ThreadPoolExecutor

    if num_workers is None:
        num_workers = len(sigmas)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        out_sigmas = list(
            ex.map(
                lambda s: _uniscale_image_feature(image, s,
                                                  mode=mode,
                                                  order=order,
                                                  shapeindex=shapeindex,
                                                  sobel=sobel,
                                                  ridge=ridge,
                                                  cval=cval,
                                                  ext_kernels=ext_kernels),
                sigmas,
            )
        )
    multiscale_gaussian = {}
    multiscale_features = {}
    for i, pair in enumerate(out_sigmas):
        gaussian_filtered, image_features = pair
        multiscale_gaussian[sigmas[i]] = gaussian_filtered
        multiscale_features[sigmas[i]] = image_features
    reformated_features = {'Sigma{}-{}'.format(sigma, k): v for sigma, dic in multiscale_features.items() for k, v in
                           dic.items()}

    # compute rog
    if rog:
        if rog_sigmas[0] in sigmas:
            g1 = multiscale_gaussian[rog_sigmas[0]]
        else:
            g1 = filters.gaussian(image, rog_sigmas[0])
        if rog_sigmas[1] in sigmas:
            g2 = multiscale_gaussian[rog_sigmas[1]]
        else:
            g2 = filters.gaussian(image, rog_sigmas[1])
        reformated_features['RoG'] = g2 / g1

    return multiscale_gaussian, reformated_features


def local_stat(target_images,
               x_coords,
               y_coords,
               selem=morphology.disk(2),
               as_dataframe=True):
    """
    Computes local statistics (min, max, std, mean) for a list of images at given coordinates.

    :param target_images: Dictionary of images
    :type target_images: dict
    :param x_coords: X-coordinates
    :type x_coords: numpy.ndarray
    :param y_coords: Y-coordinates
    :type y_coords: numpy.ndarray
    :param selem: The neighborhood expressed as a 2-D array of booleans
    :type selem: numpy.ndarray, optional
    :param as_dataframe: If True, return as pandas DataFrame, else return as numpy array
    :type as_dataframe: bool, optional
    :return: Local statistics for each image
    :rtype: pandas.DataFrame or numpy.ndarray
    """

    stacked_measures = []
    col_names = []

    if selem is None:
        for name, img in target_images.items():
            col_names.append('{}-intensity'.format(name))
            stacked_measures.append(img[x_coords, y_coords].reshape(-1, 1))
    else:
        x_mesh, y_mesh = coord2mesh(x_coords, y_coords, selem,
                                    target_images[list(target_images.keys())[0]].shape[0],
                                    target_images[list(target_images.keys())[0]].shape[1])

        for name, img in target_images.items():
            mesh_data = img[x_mesh, y_mesh]
            mesh_measures = np.array([img[x_coords, y_coords],
                                      np.min(mesh_data, axis=1),
                                      np.max(mesh_data, axis=1),
                                      np.std(mesh_data, axis=1),
                                      np.mean(mesh_data, axis=1)]).T
            col_names += ['{}_{}'.format(name, stat) for stat in ['intensity', 'min', 'max', 'std', 'mean']]
            stacked_measures.append(mesh_measures)
    if as_dataframe:
        return pd.DataFrame(np.hstack(stacked_measures), columns=col_names)
    else:
        return np.hstack(stacked_measures)


def _uniscale_image_feature(image,
                            sigma,
                            mode='reflect',
                            order='rc',
                            shapeindex=True,
                            sobel=True,
                            ridge=True,
                            ext_kernels={},
                            cval=0):
    """
    Computes uniscale features for an image including Gaussian filtered images, shape index, Sobel edges, ridges, etc.

    :param image: Input image
    :type image: numpy.ndarray
    :param sigma: The standard deviation for the Gaussian filter
    :type sigma: float
    :param mode: How to handle the borders for the filters
    :type mode: str, optional
    :param order: The order for the axes (reversed by default)
    :type order: str, optional
    :param shapeindex: Whether to compute the shape index or not
    :type shapeindex: bool, optional
    :param sobel: Whether to compute the Sobel edges or not
    :type sobel: bool, optional
    :param ridge: Whether to compute the ridges or not
    :type ridge: bool, optional
    :param ext_kernels: Additional kernels to filter the image with
    :type ext_kernels: dict, optional
    :param cval: The constant value for padding the image before filtering
    :type cval: int, optional
    :return: Gaussian filtered image and a dictionary of image features
    :rtype: numpy.ndarray, dict
    """

    from itertools import combinations_with_replacement
    from skimage.feature import hessian_matrix_eigvals

    gaussian_filtered = filters.gaussian(image,
                                         sigma=sigma,
                                         mode=mode,
                                         cval=cval)
    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)
    if order == 'rc':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]

    # correct for scale
    H_elems = [(sigma ** 2) * e for e in H_elems]
    l1, l2 = hessian_matrix_eigvals(H_elems)

    image_features = {'H_eig_1': l1, 'H_eig_2': l2}
    if shapeindex:
        # shape index
        shpid = (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))
        shpid[np.isnan(shpid)] = 0
        image_features['shapeindex'] = shpid
    if sobel:
        # sobel edge
        image_features['sobel'] = filters.sobel(gaussian_filtered)

    if ridge:
        image_features['ridge'] = filters.sato(image, sigmas=(0.1, sigma), black_ridges=False, mode='constant')

    if len(ext_kernels) > 0:
        for name, kernel in ext_kernels.items():
            image_features[name] = cv2.filter2D(image,-1,kernel)
    return gaussian_filtered, image_features


def ratio_of_gaussian(img, s1=0.5, s2=5):
    g1 = filters.gaussian(img,sigma=s1)
    g2 = filters.gaussian(img,sigma=s2)
    return g1/g2


def hu_log10_transform(hu_moments):
    warnings.filterwarnings("ignore")
    abs_vals = np.abs(hu_moments)
    hu_log10 = np.copysign(np.log10(abs_vals), hu_moments)
    corrected = np.zeros(hu_moments.shape)
    corrected[np.isfinite(hu_log10)] = hu_log10[np.isfinite(hu_log10)]
    return corrected


def median_over_gaussian(data,windowsize=3):
    return (ndi.median_filter(data,size=windowsize)/filters.gaussian(data,sigma=windowsize,preserve_range=True))


def normalized_difference_of_gaussian(data, s1=0, s2=10):
    g1 = filters.gaussian(data, sigma=s1)
    g2 = filters.gaussian(data, sigma=s2)
    return (g1-g2)/g1


def masked_percentile(regionmask, intensity):
    return np.percentile(intensity[regionmask], q=(25, 50, 75))


def masked_cv(regionmask, intensity):
    return 100*np.std(intensity[regionmask])/np.mean(intensity[regionmask])


def masked_skewness(regionmask, intensity):
    return stats.skew(intensity[regionmask])


def masked_kurtosis(regionmask, intensity):
    return stats.kurtosis(intensity[regionmask])


def masked_std(regionmask, intensity):
    return np.std(intensity[regionmask])


def gabor_kernel_bank(n_theta=4,
                      sigmas=[2],
                      lamdas = [np.pi/4],
                      gammas = [0.5],
                      kernel_size=5):
    """
    Generate a bank of Gabor filters with different orientations and parameters.
    Annotated by GPT-4 on 2023/05/21
    :param n_theta: The number of different orientations
    :type n_theta: int, optional
    :param sigmas: The standard deviation(s) of the gaussian envelope
    :type sigmas: list of float, optional
    :param lamdas: The wavelength(s) of the sinusoidal factor
    :type lamdas: list of float, optional
    :param gammas: The spatial aspect ratio(s)
    :type gammas: list of float, optional
    :param kernel_size: The size of the gabor kernel
    :type kernel_size: int, optional
    :return: A dictionary of gabor kernels
    :rtype: dict
    """
    gabor_kernels = {}
    counter=0
    for i in range(n_theta):
        theta = (i/n_theta)*np.pi
        for sigma in sigmas:
            for lamda in lamdas:
                for gamma in gammas:
                    kernel = cv2.getGaborKernel((kernel_size,kernel_size),
                                                sigma,theta,lamda,gamma,0,
                                                ktype=cv2.CV_64F)
                    gabor_kernels['Gabor{}'.format(counter)]=kernel
                    counter+=1
    return gabor_kernels

def basic_intensity_features(data):
    from scipy.stats import kurtosis,skew
    names = ['mean','median','max','min','Q1','Q3','std','CV','skewness','kurtosis']
    data_stats = [np.mean(data),
                  np.median(data),
                  np.max(data),
                  np.min(data),
                  np.percentile(data,25),
                  np.percentile(data,75),
                  np.std(data),
                  np.std(data)/np.mean(data),
                  skew(data.flatten()),
                  kurtosis(data.flatten())]
    return names, data_stats


def sinuosity(midline):
    if (midline[0]-midline[-1]).sum() == 0:
        raise ValueError('Midline coordinates appear to be closed!')
    end_to_end_dist = distance(midline[0],midline[-1])
    length = measure_length(midline)
    ret = round(length/end_to_end_dist, 3)
    if ret < 1:
        ret = 1.0
    return ret