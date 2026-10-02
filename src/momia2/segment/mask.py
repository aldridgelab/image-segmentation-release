from ._helper import *
from ..utils import *
import pandas as pd

__all__ = ['BinarizeLegend','Patchifier','MakeMask']

def mask_custom(img, th1=0, th2='auto',
                return_threshold=False, config=None, ):
    if th2 == 'auto':
        th2 = filters.threshold_isodata(img)
    binary = ((img < th2) & (img > th1)) * 1
    if not return_threshold:
        return binary.astype(int)
    else:
        return binary.astype(int), (th1, th2)


def mask_sauvola(img,
                 dark_foreground=True,
                 return_threshold=False,
                 config=None,
                 **kwargs):
    th = filters.threshold_sauvola(img, **kwargs)
    if dark_foreground:
        binary = (img < th) * 1
    else:
        binary = (img > th) * 1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)


def mask_isodata(img,
                 dark_foreground=True,
                 return_threshold=False,
                 config=None,
                 **kwargs):
    th = filters.threshold_isodata(img, **kwargs)
    if dark_foreground:
        binary = (img < th) * 1
    else:
        binary = (img > th) * 1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)


def mask_local(img,
               dark_foreground=True,
               return_threshold=False,
               block_size=15,
               config=None,
               **kwargs):
    th = filters.threshold_local(img, block_size=block_size, **kwargs)
    if dark_foreground:
        binary = (img < th) * 1
    else:
        binary = (img > th) * 1
    if return_threshold:
        return binary.astype(int), th
    else:
        return binary.astype(int)


def mask_legacy(img,
                block_sizes=(15, 81),
                method='gaussian',
                dilation_selem=morphology.disk(1),
                opening_selem=morphology.disk(1),
                config=None):
    """
    binarization method used in OMEGA (MOMIA1), deprecated.
    """
    local_mask1 = mask_local(img, block_sizes[0], method=method)
    local_mask2 = mask_local(img, block_sizes[1], method=method)
    glob_mask = mask_isodata(img)
    glob_mask = morphology.binary_dilation(glob_mask, dilation_selem)
    binary = ((local_mask1 + local_mask2) * glob_mask) > 0
    binary = morphology.binary_opening(binary, opening_selem)
    return binary.astype(int)


def mask_composite(img, config=None):
    # reset configurations when provided
    if config is not None:
        window_size = int(config['segment']['sauvola_window_size'])
        k = float(config['segment']['sauvola_k'])
        min_particle_size = int(config['segment']['min_size'])
        opening = bool(int(config['segment']['binary_opening']))
        overlap_threshold = float(config['segment']['overlap_threshold'])
        dilation = bool(int(config['segment']['binary_dilation']))
    else:
        window_size = 9  # sauvola local threshold window size
        k = 0.05  # sauvola local threshold regulation
        min_particle_size = 10  # minimal particle size included
        opening = False  # opening to delink neighboring particles
        overlap_threshold = 0.5  # threshold of overlap between masks rendered by local and global thresholding
        dilation = True

    # local threshold
    mask1 = mask_sauvola(img, window_size=window_size, k=k)
    # isodata threshold
    mask2 = mask_isodata(img)
    # composite mask
    init_mask = morphology.remove_small_objects((mask1 + mask2) > 0, min_size=min_particle_size)
    if opening:
        init_mask = morphology.binary_opening(init_mask)

    # label composite mask
    labels = measure.label(init_mask * 1)
    filtered_labels = labels.copy()

    # measure particles, define foreground/background
    region_props = pd.DataFrame(
        measure.regionprops_table(labels, intensity_image=mask2, properties=['coords', 'mean_intensity']))
    foreground = (region_props['mean_intensity'] >= overlap_threshold) * 1
    region_props['foreground'] = foreground

    # remove background particles
    x, y = np.vstack(region_props[foreground == 0]['coords'].values).T
    filtered_labels[x, y] = 0
    binary = (filtered_labels > 0) * 1
    if dilation:
        binary = morphology.binary_dilation(binary)
    return binary.astype(int)


def _make_mask(img, method=3, **kwargs):
    if method == 0:
        return mask_custom(img, **kwargs)
    if method == 1:
        return mask_isodata(img, **kwargs)
    if method == 2:
        return mask_sauvola(img, **kwargs)
    if method == 3:
        return mask_local(img, **kwargs)
    if method == 4:
        return mask_legacy(img, **kwargs)
    if method == 5:
        return mask_composite(img, **kwargs)


class Patchifier:
    """
    A simple way to convert 2D images to patches and stitch them back into one
    Currently it only works with images with shapes like (height, width) or (height, width, channel), it doesn't work
    on image series such as (frame, height, width, channel)
    The smoothing function for overlap edges is simply the mean values of the overlapping pixels. Future updates may
    consider implementing 2D spline interpolation based smoothing method, such as:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
    """

    def __init__(self, img_shape=(512, 512), patch_size=128, pad=32):

        """
        :param img_shape: the original shape of the large input image
        :param patch_size: the height and width of the square-shaped patch
        :param pad: half-width of the overlapping region of neighboring patches
        """

        self._shape = img_shape
        self.size = patch_size
        self.pad = pad
        self.shape = (max(self._shape[0], self.size),
                      max(self._shape[1], self.size))
        self.pad_h = max(0, self.size - self._shape[0])
        self.pad_w = max(0, self.size - self._shape[1])
        self.ref_coords = self.generate_patch_coords()

    def generate_patch_coords(self):
        """
        funtion to generate patch coords
        :return:
        """
        h, w = self.shape
        xs = list(np.arange(0, h - self.size, self.size - 2 * self.pad)) + [h - self.size]
        if len(xs) > 1:
            if xs[-1] == xs[-2]:
                xs = xs[:-1]
        ys = list(np.arange(0, w - self.size, self.size - 2 * self.pad)) + [w - self.size]
        if len(ys) > 1:
            if ys[-1] == ys[-2]:
                ys = ys[:-1]
        ref_coords = np.array([[x, y, np.random.randint(2)] for x in xs for y in ys])
        return ref_coords

    def pachify(self, img, random_rotate=False):
        """
        convert img to patches
        :param img: momia2 image
        :param random_rotate: if randomly rotate clips, this shouldn't be used for prediction but can be helpful for training
        :return: clipped patches
        """
        if self.shape != img.shape[:2]:
            self.__init__(img.shape[:2])
        pad_config = np.zeros((len(img.shape), 2))
        pad_config[0][1] = self.pad_h
        pad_config[1][1] = self.pad_w
        pad_config = pad_config.astype(int)
        if self.pad_h > 0 or self.pad_w > 0:
            padded_img = np.pad(img.copy(), pad_config, mode='constant')
        else:
            padded_img = img.copy()
        patches = []
        for x, y, t in self.ref_coords:
            p = padded_img[x:x + self.size, y:y + self.size]
            if random_rotate and t:
                p = p.T
            patches.append(p)
        return np.array(patches)

    def unpatchify(self, patches, n_channel):
        """
        stitch patches back into one
        :param patches: array of patches
        :param n_channel: number of channels, for instance, for a rgb image n_channel should be 3
        :return:
        """
        canvas = np.zeros(list(self.shape) + [n_channel])
        canvas_counter = np.zeros(self.shape)
        for i, p in enumerate(patches):
            x, y = self.ref_coords[i][0], self.ref_coords[i][1]
            canvas[x:x + self.size, y:y + self.size] += p
            canvas_counter[x:x + self.size, y:y + self.size] += 1
        mean_canvas = canvas / canvas_counter[:, :, np.newaxis]
        return mean_canvas[:self._shape[0], :self._shape[1]]


class MakeMask:

    def __init__(self, model,
                 description=None,
                 n_classes=1,
                 n_channels=1,
                 threshold=[0.5],
                 verbose=True,
                 **kwargs):
        """
        This function allows the users to conviniently use pre-implemented or separately trained models to do
        single- or multi-class mask prediction (semantic segment)
        """
        self.model = model
        self.description = description
        self.verbose = bool(verbose)
        self.threshold = threshold
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.kwargs = kwargs

    def predict(self, targets):
        """
        targets
        """
        if not isinstance(targets, list):
            targets = [targets]

        if self.verbose:
            if self.description is not None:
                print("Here's some extra information the user provided: {}.".format(self.description))

        # mask predictions are exported as list rather than np.array, considering instances where target images have different shapes
        predictions = []

        # check target channels
        for i, t in enumerate(targets):
            t = dimension_scaler(t)
            shape = t.shape
            expected_shape = tuple(list(shape[:-1]) + [self.n_channels])
            if shape[-1] != self.n_channels:
                error_msg = 'Expected image dimension is {} but {} was given.'.format(expected_shape, shape)
                raise ValueError(error_msg)
            probabilities = self.model.predict(t, **self.kwargs)
            binary = probabilities
            predictions.append(binary)
        return predictions


class BinarizeLegend:

    def __init__(self, method=5, **kwargs):
        self.method = method
        self.kwargs = kwargs

    def predict(self, target):
        target = np.squeeze(target)
        target = target.astype(float)
        target /= target.mean()
        return _make_mask(target, method=self.method, **self.kwargs)
