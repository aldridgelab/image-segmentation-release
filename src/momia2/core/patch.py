import numpy as np

from ..utils import *
from ..segment import *
from ..plot import *
from ..momia_IO import pickle_save, pickle_load
import pandas as pd
import PIL,io
from ..classify import run_particle_labeler
from matplotlib import pyplot as plt
from skimage import morphology, draw
from typing import Optional, Dict, Union, Any


__all__ = ['Patch']


class Patch:

    def __init__(self,
                 image_dict: Optional[Dict[str, Any]] = None,
                 configfile: Optional[str] = None,
                 image_id: Optional[int] = 0,
                 ref_channel: Union[int, str] = -1,
                 store_backup: bool = False) -> None:
        """
        Initialize a Patch object.
        # make sure that the number of images match the number of channels provided.

        :param image_dict: Dictionary containing image data.
        :param configfile: Path to the configuration file.
        :param image_id: Text ID of the image.
        :param ref_channel: Specify the name/index of the reference channel for the image.
        :param store_backup: Flag to store unprocessed data.
        """

        self.id = image_id
        self.data = {}
        self._data = {}
        self.particles = {}
        self.cache = PatchCache()
        self.seed = None
        self.labeled_seeds = None
        self.labeled_mask = None
        self.config = load_config(configfile)
        self.mask = None
        self.prob_mask = None
        self.pixel_microns = 0.065
        self.shape = (2048, 2048)
        self.offset = (0,0)
        self.channels = []
        self.ref_channel = ''
        self.bbox = ()
        self.store_backup = store_backup
        self.regionprops = pd.DataFrame()
        self.boundaries = None
        self.wd = './'
        if isinstance(image_dict,dict):
            self.load_data(image_dict,
                           ref_channel=ref_channel,
                           image_id=image_id)

    def load_data(self,
                  image_dict: Dict[str, np.array],
                  image_id: Optional[int] = 0,
                  ref_channel: Union[int,str]=-1):
        """
        Load image data into the Patch object.

        :param image_dict: Dictionary containing compiled image data: {'Channel_name':2D numpy array}.
        :param image_id: ID of the image.
        :param ref_channel: Reference channel for the image.
        """
        self.id = image_id
        self.data = image_dict
        self.channels = list(image_dict.keys())
        if isinstance(ref_channel, int):
            self.ref_channel = self.channels[ref_channel]
        elif isinstance(ref_channel, str):
            self.ref_channel = ref_channel
        else:
            raise ValueError('reference channel should be integer or string.')
        self.shape = self.data[self.ref_channel].shape
        if self.store_backup:
            self._data = self.data.copy()

    def revert(self) -> None:
        """
        Revert the image data to its original/unprocessed state.
        """
        if self.store_backup:
            self.data = self._data.copy()
        else:
            print('No raw data found. Set "store_backup" to True to backup raw data.')

    def load_seed(self,
                  seed: np.array) -> None:
        """
        Load seed image into the Patch object (to guide watershed segmentation).

        :param seed: Seed image.
        """
        if seed.shape != self.shape:
            raise ValueError("The shapes of seed image and target images don't match!")
        self.seed = seed

    def load_mask(self,
                  mask: np.array) -> None:
        """
        Load a precomputed mask into the Patch object.

        :param mask: Precomputed mask image.
        """
        if mask.shape != self.shape:
            raise ValueError("The shapes of mask image and target images don't match!")
        self.mask = mask

    def load_labeled_mask(self,
                          labeled_mask: np.array) -> None:
        """
        Load a precomputed labeled mask into the Patch object.

        :param labeled_mask: Labeled mask image.
        """
        if labeled_mask.shape != self.shape:
            raise ValueError("The shapes of labeled mask and target images don't match!")
        self.labeled_mask = labeled_mask

    def batch_process(self) -> None:
        return None

    def preprocess_images(self,
                          func=None,
                          channels = 'all', **kwargs) -> None:
        """
        Preprocess images based on a provided function and channels.

        :param func: Function to preprocess images.
        :param channels: Channels to apply the function on.
        :param kwargs: Additional keyword arguments for the function.
        """
        target_channels = []
        if isinstance(channels,str):
            if channels in ['ALL','All','all','-']:
                target_channels=self.channels
            elif channels in ['FL','fl','Fluorescence','fluorescence','F','f']:
                target_channels = [c for c in self.channels if c != self.ref_channel]
            elif channels in ['Ref','ref','Reference','reference','R','r']:
                target_channels = [self.ref_channel]
            elif channels in self.channels:
                target_channels = [channels]
            else:
                raise ValueError('Invalid channel variable "{}".'.format(channels))
        elif isinstance(channels,list):
            for c in channels:
                if c in self.channels:
                    target_channels.append(c)
                else:
                    raise ValueError('Channel "{}" not found.'.format(c))
        try:
            for c in target_channels:
                self.data[c] = func(self.data[c],**kwargs)
        except:
            print('Failed in applying function {} on these channels: {}'.format(func._name_, target_channels))

    def crop_edge(self,edge_fraction=None) -> None:
        """
        crop edge pixels to suppress edge aberrations.
        :param edge_fraction: proportion of edge pixels being cropped from each side, should be between 0 and 0.4.
        """
        if isinstance(edge_fraction,float):
            self.config['IMAGE']['EDGE'] = edge_fraction
        else:
            edge_fraction = self.config['IMAGE']['EDGE']
        if 0 <= edge_fraction < 0.4:
            crop_width = int(edge_fraction * self.shape[1])
            crop_height = int(edge_fraction * self.shape[0])
            w1, w2 = crop_width, self.shape[1] - crop_width
            h1, h2 = crop_height, self.shape[0] - crop_height
            self.shape = (h2 - h1, w2 - w1)
            self.offset = (crop_height, crop_width)
        else:
            raise ValueError('Edge fraction should be no higher than 0.4 (40% from each side)!')

        # crope edge
        for channel, data in self.data.items():
            self.data[channel] = self.data[channel][h1:h2, w1:w2]

    def correct_xy_drift(self,
                         reference_channel=None,
                         invert_ref=None,
                         max_drift=None) -> None:
        """
        Planar drift correction.
        :param reference_channel: specify reference channel, uses Patch.ref_channel by default.
        :param invert_ref: if invert reference channel data, set to True if reference channel has a light background, e.g. phase contrast.
        :param max_drift: maximum drift (in pixels) to correct for.
        """
        if reference_channel is None:
            reference_channel = self.config['IMAGE']['DRIFT_CORRECTION']['REFERENCE']
            reference_image = self.get_ref_image()
        elif isinstance(reference_channel,str):
            if reference_channel == 'default':
                reference_channel = self.ref_channel
                reference_image = self.get_ref_image()
            elif reference_channel in list(self.data.keys()):
                reference_image = self.data[reference_channel]
                self.config['IMAGE']['DRIFT_CORRECTION']['REFERENCE']=reference_channel
        else:
            raise ValueError('Channel {} not found!'.format(reference_channel))

        if invert_ref is None:
            invert_ref = self.config['IMAGE']['DRIFT_CORRECTION']['INVERT']
        elif isinstance(invert_ref,bool):
            self.config['IMAGE']['DRIFT_CORRECTION']['INVERT']=bool(invert_ref)
        else:
            raise ValueError('The variable "invert_ref" should be boolean, not {}!'.format(type(invert_ref)))

        if max_drift is None:
            max_drift = self.config['IMAGE']['DRIFT_CORRECTION']['MAX_DRIFT_PIX']
        elif isinstance(max_drift,(int,float)):
            self.config['IMAGE']['DRIFT_CORRECTION']['MAX_DRIFT_PIX']=bool(max_drift)
        else:
            raise ValueError('The variable "max_drift" should be float or int, not {}!'.format(type(max_drift)))

        # invert image if necessary
        if invert_ref:
            reference_image = 100 + reference_image.max() - reference_image
        # correct xy drift by phase cross correlation
        for channel, data in self.data.items():
            if channel != reference_channel:
                shift = drift_detection(reference_image,data,upsample_factor=10)
                if max(np.abs(shift)) <= max_drift:
                    offset_image = drift_correction(data, shift)
                    self.data[channel] = offset_image

    def generate_mask(self,
                      method=1,
                      min_particle_size=None,
                      min_hole_size=None,
                      **kwargs) -> None:
        """
        Generate a binary mask for the image.

        :param method: Numeric index or name of a pre-defined binarization method
        :param min_particle_size: Minimum size for particles in the mask.
        :param min_hole_size: Minimum size for holes in the mask.
        :param kwargs: Additional keyword arguments for the method.
        """

        # Default to env-tunable sizes if not provided
        from momia2.utils.corrections import MIN_PARTICLE_SIZE_DEFAULT, MIN_HOLE_SIZE_DEFAULT

        if min_particle_size is None:
            min_particle_size = MIN_PARTICLE_SIZE_DEFAULT
        if min_hole_size is None:
            min_hole_size = MIN_HOLE_SIZE_DEFAULT

        mask = BinarizeLegend(method, **kwargs).predict(self.get_ref_image())
        # ensure boolean mask before morphology ops to avoid skimage warnings
        mask = mask.astype(bool)
        mask = morphology.remove_small_holes(mask, min_hole_size)
        mask = morphology.remove_small_objects(mask, min_particle_size)
        mask = mask.astype(bool)
        self.mask = mask

    def label_mask(self) -> None:
        """
        Label the binary mask.
        """
        self.labeled_mask = measure.label(self.mask)

    def locate_particles(self,
                         use_intensity=True,
                         cache_mask = False,
                         cache_data = False,
                         edge_width = None) -> None:
        """
        Locate particles in the image.

        :param use_intensity: Use intensity data for locating particles.
        :param cache_mask: Cache the binary mask for future use.
        :param edge_width: Width in pixels for edge detection threshold.
        """
        if self.labeled_mask.max() > 0:
            if use_intensity:
                self.regionprops = get_regionprop(self.labeled_mask, intensity_image=self.get_ref_image())
            else:
                self.regionprops = get_regionprop(self.labeled_mask, intensity_image=None)
            self.regionprops.set_index('$label', inplace=True)
            self.regionprops['$image_id'] = [self.id] * len(self.regionprops)
            self.regionprops['$Cell_id'] = ['{}_{}'.format(self.id, nid) for nid in self.regionprops.index]
            self.regionprops[
                ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2', '$touching_edge']] = optimize_bbox_batch(
                                                                                    self.shape, self.regionprops, edge_width=edge_width)
            self.regionprops['$include'] = 1
            self.regionprops['$midlines'] = [[]] * len(self.regionprops)
            self.regionprops['$outline'] = [[]] * len(self.regionprops)
            self.regionprops['$refined_outline'] = [[]] * len(self.regionprops)
            self.regionprops['$widths_list'] = [[]] * len(self.regionprops)
            masks = []
            data_dict = {c:[] for c in self.channels}
            for l in self.regionprops.index:
                x1, y1, x2, y2 = self.regionprops.loc[l, ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2']]
                if cache_mask:
                    masks.append(self.labeled_mask[x1:x2, y1:y2] == l)
                if cache_data:
                    for c in self.channels:
                        data_dict[c].append(self.get_channel_data(c)[x1:x2, y1:y2])
            if cache_mask:
                self.regionprops['$mask'] = masks
            if cache_data:
                for c in self.channels:
                    self.regionprops['${}'.format(c)] = data_dict[c]

    def get_particle_data(self,particle_id):
        """
        return particle data by ID
        :param particle_id: user provided unqiue particle ID
        :return: particle data dict
        """
        if len(self.regionprops)==0:
            self.locate_particles()
        if particle_id not in self.regionprops.index:
            raise ValueError('Particle with label {} not found!'.format(particle_id))

        particle_dict = dict(self.regionprops.loc[particle_id])
        x1,y1,x2,y2 = [particle_dict[x] for x in ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2']]
        for c in self.channels:
            particle_dict[c] = self.data[c][x1:x2,y1:y2]
        if '$mask' not in particle_dict:
            particle_dict['$mask'] = (self.labeled_mask==particle_id)[x1:x2,y1:y2]
        return particle_dict

    def get_particle_mask(self,particle_id,cropped=True):
        """
        retrieve particle mask, cropped or not
        :param particle_id: user provided unqiue particle ID
        :param cropped: whether to crop the mask by optimized bbox
        :return: particle binary mask, cropped or not
        """
        if len(self.regionprops)==0:
            self.locate_particles()
        if particle_id not in self.regionprops.index:
            raise ValueError('Particle with label {} not found!'.format(particle_id))
        if not cropped:
            return (self.labeled_mask==particle_id)**1
        else:
            x1, y1, x2, y2 = self.regionprops.loc[particle_id,['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2']].values
            return (self.labeled_mask==particle_id)[x1:x2,y1:y2]

    def filter_particles(self,
                         filter_dict: Dict[str, Any]):
        """
        Filters particles in the region properties based on given criteria.

        This function performs two tasks:
        1. It removes particles that touch the edges of the image.
        2. It filters particles based on the criteria specified in the filter_dict.
           For keys starting with '$', particles are filtered based on exact value matching.
           For other keys, particles are filtered based on a value range.

        The results are stored in the '$include' column of the regionprops DataFrame,
        where particles that pass the filter are marked with a 1 and others with a 0.

        :param filter_dict: Dictionary specifying filtering criteria.
                            The keys should be feature names (possibly starting with '$')
                            and the values should be either:
                            a) An exact value for features with '$' key prefix.
                            b) A tuple (min_value, max_value) specifying a range for other features.

        :return: None. This method modifies the internal regionprops DataFrame in-place.
        """
        _accept = [self.regionprops['$touching_edge'].values == 0]
        # filter by range
        for k, r in filter_dict.items():
            if '$' in k:
                _accept.append(self.regionprops[k].values == r)
            else:
                vmin, vmax = r
                if k in self.regionprops:
                    v = self.regionprops[k].values
                    _accept.append((v > vmin) * (v < vmax))
                else:
                    print('feature ${}$ not found!'.format(k))
        self.regionprops['$include'] = (np.sum(_accept, axis=0) == len(_accept)) * 1

    def save_particles(self,
                       filename,
                       filtered_only=True,
                       as_dictionary=False):
        if filtered_only:
            subset=self.regionprops[self.regionprops['$include']==1].copy()
            id_list = self.regionprops[self.regionprops['$include']==1].index
        else:
            subset = self.regionprops.copy()
            id_list = self.regionprops.index
        if as_dictionary:
            particle_dict = {cid:self.get_particle_data(nid) for nid,cid in zip(id_list,
                                                                                self.regionprops['$Cell_id'].values)}
            pickle_save(particle_dict,filename)
        else:
            pickle_save(subset, filename)

    def _get_intensity_stats(self, channel):
        """
        Add basic intensity features to the regionprop table
        :params channel: specify channel name
        """
        intensity_stats = []
        if channel not in self.channels:
            raise ValueError('Channel {} not found!'.format(channel))

        channel_data = self.get_channel_data(channel)
        for coords in self.regionprops['$coords'].values:
            particle_data = channel_data[coords[:, 0], coords[:, 1]]
            _cols, particle_stat = basic_intensity_features(particle_data)
            intensity_stats.append(particle_stat)
        columns = ['{}_{}'.format(channel, c) for c in _cols]
        self.regionprops[columns] = intensity_stats

    def get_intensity_stats(self):
        """
        get particle intensity stats of all channels
        """
        for c in self.channels:
            self._get_intensity_stats(c)

    def get_morphology_stats(self):
        """

        get particle morphological stats of all particles
        -------

        """
        # measure widths
        widths_list = []
        length_profiles = []
        widths_profiles = []

        for midlines,outline,refined_outline in self.regionprops[['$midlines','$outline','$refined_outline']].values:
            widths = []
            if len(midlines)>0:
                if len(refined_outline) > 0:
                    _outline = refined_outline
                elif len(outline) > 0:
                    _outline = outline
                else:
                    # No valid outline, skip width calculation
                    widths_profiles.append([0,0,0,0,0])
                    length = np.sum([measure_length(midline) for midline in midlines]) * self.pixel_microns
                    length_profiles.append(length)
                    widths_list.append(widths)
                    continue
                for segment in midlines:
                    w = direct_intersect_distance(segment,_outline)
                    if len(w) > 0:
                        widths.append(w)
                # Filter valid width arrays with interior points
                valid_widths = [width[1:-1] for width in widths if len(width) > 2]
                if len(valid_widths) > 0:
                    width_concat = np.concatenate(valid_widths)
                    if len(width_concat) > 0:
                        widths_profiles.append([np.median(width_concat)*self.pixel_microns,
                                                np.std(width_concat)*self.pixel_microns,
                                                np.max(width_concat)*self.pixel_microns,
                                                np.percentile(width_concat,25)*self.pixel_microns,
                                                np.percentile(width_concat,75)*self.pixel_microns])
                    else:
                        widths_profiles.append([0,0,0,0,0])
                else:
                    widths_profiles.append([0,0,0,0,0])
                length = np.sum([measure_length(midline) for midline in midlines]) * self.pixel_microns
            else:
                widths_profiles.append([0,0,0,0,0])
                length = 0

            length_profiles.append(length)
            widths_list.append(widths)

        self.regionprops['$widths_list'] = widths_profiles
        self.regionprops['length [µm]'] = length_profiles
        self.regionprops[['width_median [µm]',
                          'width_std [µm]',
                          'width_max [µm]',
                          'width_Q1 [µm]',
                          'width_Q3 [µm]']] = widths_profiles
        sinuosities = self.regionprops['length [µm]'].values/(self.pixel_microns*self.regionprops['major_axis_length'])
        sinuosities[(sinuosities<1)&(sinuosities>0.5)]=1
        self.regionprops['sinuosity']=sinuosities

    def render_image_features(self, model='default', mode='default'):
        if isinstance(model, str):
            if (model == 'default') & (len(self.cache.feature_columns) == 0):
                self._image2pixelfeatures(mode=mode)
        else:
            model.predict(self)
        return None

    def annot2feature(self):
        use_Gaussian = bool(int(self.config['classification_pixel']['use_Gaussian']))
        pixel_feature_selem = generic.config2selem(str(self.config['classification_pixel']['pixel_feature_selem']))
        self.cache.extract_local_features(use_Gaussian=use_Gaussian, selem=pixel_feature_selem)

    def mask2feature(self, ):
        coords = np.where(self.mask > 0)
        self.cache.pixel_data['xcoord'] = coords[0]
        self.cache.pixel_data['ycoord'] = coords[1]
        self.annot2feature()

    def pixel_data(self):
        return self.cache.pixel_data

    def get_ref_image(self):
        return self.data[self.ref_channel]

    def get_channel_data(self, channel):
        if isinstance(channel,str):
            if channel in self.channels:
                return self.data[channel]
            else:
                raise ValueError('Invalid channel name:{}'.format(channel))
        elif isinstance(channel,int):
            if channel<len(self.channels) and channel >=-1:
                return self.data[self.channels[channel]]
            else:
                raise ValueError('Invalid numeric channel index:{}'.format(channel))
        else:
            raise ValueError('Invalid channel format: {}. Can only be a string or integer'.format(type(channel)))

    def find_outline(self,
                     level=0.1,
                     dilation=False,
                     erosion=False,
                     calculate_curvature=True,
                     angularity_window=15,
                     dark_background=False,
                     sigma=1,
                     approximate=False,
                     tolerance=0.3,
                     interp_distance=1,
                     min_segment_count=3,
                     update_mask=True):

        if len(self.regionprops) > 0:
            bending_stat = []
            masks = []
            outline_list = []
            if '$opt-x1' not in self.regionprops:
                self.regionprops[
                    ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2', '$touching_edge']] = corrections.optimize_bbox_batch(
                    self.shape,
                    self.regionprops)
            for label in self.regionprops.index:
                x1, y1, x2, y2, touching_edge = self.regionprops.loc[label][
                    ['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2', '$touching_edge']].values.astype(int)
                mask = (self.labeled_mask[x1:x2, y1:y2] == label).astype(int)
                if not touching_edge:
                    data = self.get_ref_image()[x1:x2, y1:y2]
                    outline = find_contour_marching_squares(mask, data,
                                                            level=level,
                                                            dilation=dilation,
                                                            erosion=erosion,
                                                            dark_background=dark_background,
                                                            sigma=sigma,
                                                            approximate=approximate,
                                                            tolerance=tolerance,
                                                            interp_distance=interp_distance,
                                                            min_segment_count=min_segment_count)
                    if len(outline) == 0:
                        # Empty outline - treat like touching_edge case
                        outline_list.append(np.array([]))
                        if calculate_curvature:
                            bending_stat.append([np.nan] * 5)
                        masks.append(mask)
                        continue
                    if calculate_curvature:
                        bending = -bend_angle_closed(outline, window=angularity_window)
                        bending_stat.append([bending.min(), bending.max(), bending.mean(),
                                             np.percentile(bending, 25), np.percentile(bending, 75)])
                    outline_list.append(outline)
                    if update_mask:
                        mask = draw.polygon2mask(mask.shape, outline).astype(int)
                else:
                    outline_list.append(np.array([]))
                    if calculate_curvature:
                        bending_stat.append([np.nan] * 5)
                masks.append(mask)
            self.regionprops['$outline'] = outline_list
            self.regionprops['$mask'] = masks
            if calculate_curvature:
                self.regionprops[['min_negative_curvature',
                                  'max_positive_curvature',
                                  'mean_curvature', 'Q1_curvature', 'Q3_curvature']] = bending_stat

    def extract_midlines(self,
                         min_branch_length=0.2,
                         pole_length=0.25,
                         method='zhang',
                         filtered_only=True):
        """

        Parameters
        ----------
        min_branch_length
        pole_length
        method
        filtered_only

        Returns
        -------

        """
        #
        midline_list = []
        cell_ids = self.regionprops.index
        for idx in cell_ids:
            cell = self.get_particle_data(idx)
            cent_x,cent_y = cell['$centroid-0']-cell['$opt-x1'],cell['$centroid-1']-cell['$opt-y1']
            mask = cell['$mask']
            mask_coords = np.array(np.where(mask>0)).T
            outline = cell['$outline']
            orientation = cell['orientation']
            solidity = cell['solidity']

            if not cell['$include'] and filtered_only:
                midlines = []
            elif len(outline) == 0:
                midlines = []
            # if cell is convex, use fast mode
            else:
                try:
                    if solidity > 0.8 and np.array([cent_x,cent_y]).astype(int) in mask_coords:
                        if len(outline) > 100:
                            upsampling_factor=1
                        else:
                            upsampling_factor=2
                        midlines = fast_midline(mask,outline,
                                                orientation=orientation,
                                                centroid=np.array([cent_x,cent_y]),
                                                pixel_microns=self.pixel_microns,
                                                pole_length=pole_length,
                                                upsampling_factor=upsampling_factor)
                    else:
                        midlines = extract_midline(mask, outline,
                                                    pixel_microns=self.pixel_microns,
                                                    min_branch_length=min_branch_length,
                                                    method=method)
                except:
                    midlines = []
            midline_list.append(midlines)
        self.regionprops['$midlines'] = midline_list

    def plot(self,
             cell_ids=[],
             figsize=(5, 5),
             outline_prop={},
             midline_prop={},
             cell_plot=False,
             channel=-1):
        print('Rendering cell plot(s)...')
        plot_cells(self,cell_ids,figsize,outline_prop,midline_prop,cell_plot,channel)

    def _image2pixelfeatures(self, mode='default'):
        if mode == 'default':
            sigmas = (0.5, 1.5, 5)
            use_RoG = True
            use_Ridge = True
            use_Sobel = False
            use_Gaussian = True
            use_Shapeindex = True
            num_workers = 12
        elif mode == 'use_config':
            # adopt parameters from configurations
            sigmas = np.array(self.config['classification_pixel']['sigmas'].split(',')).astype(float)
            use_RoG = bool(int(self.config['classification_pixel']['use_RoG']))
            use_Ridge = bool(int(self.config['classification_pixel']['use_Ridge']))
            use_Sobel = bool(int(self.config['classification_pixel']['use_Sobel']))
            use_Gaussian = bool(int(self.config['classification_pixel']['use_Gaussian']))
            use_Shapeindex = bool(int(self.config['classification_pixel']['use_Shapeindex']))
            num_workers = int(self.config['classification_pixel']['num_workers'])
        elif mode == 'fast':
            sigmas = (float(self.config['classification_pixel']['ridge_sigma']))
            use_RoG = True
            use_Ridge = True
            use_Sobel = False
            use_Gaussian = False
            use_Shapeindex = True
            num_workers = 12
        else:
            raise ValueError(
                'Illegal pixel feature extraction mode "{}", use "default", "fast", "use_config" instead.'.format(mode))
        # feature images
        self.cache.prepare_feature_images(self.get_ref_image(),
                                          num_workers=num_workers,
                                          sigmas=sigmas,
                                          rog=use_RoG,
                                          ridge=use_Ridge,
                                          sobel=use_Sobel,
                                          shapeindex=use_Shapeindex)

    def _seed2annot(self):
        from skimage.segmentation import watershed
        from skimage.measure import regionprops_table, label

        # label_seed
        self.labeled_seeds = label(self.seed)

        # adopt parameters from configurations
        ridge_sigma = float(self.config['classification_pixel']['ridge_sigma'])
        min_seed_fraction = float(self.config['classification_pixel']['min_seed_fraction'])
        erosion_selem = generic.config2selem(str(self.config['classification_pixel']['seeded_mask_erosion_selem']))
        dilation_selem = generic.config2selem(str(self.config['classification_pixel']['seeded_mask_dilation_selem']))

        # watershed segment
        ridge = filters.sato(self.get_ref_image(),
                             sigmas=(ridge_sigma),
                             black_ridges=False)
        labeled_mask = watershed(ridge, markers=self.labeled_seeds,
                                 mask=self.mask,
                                 watershed_line=True,
                                 compactness=1)
        rp = pd.DataFrame(regionprops_table(labeled_mask,
                                            intensity_image=(self.seed > 0) * 1,
                                            properties=['coords', 'mean_intensity']))
        if rp['mean_intensity'].max() < min_seed_fraction:
            x, y = np.vstack(rp[rp['mean_intensity'] < min_seed_fraction]['coords'].values).T
            labeled_mask[x, y] = 0
        self.labeled_mask = labeled_mask

        # extract relavent pixel coordinates
        seeded_mask = (labeled_mask > 0) * 1
        core = morphology.binary_erosion(seeded_mask, erosion_selem) * 1
        dilated = morphology.binary_dilation(seeded_mask, dilation_selem) * 1
        residual = ((self.mask - dilated) == 1) * 1
        edge = dilated - core

        # mask to coords
        core_coords = np.array(np.where(core > 0))
        edge_coords = np.array(np.where(edge > 0))
        residual_coords = np.array(np.where(residual > 0))

        # concat coords and annotations
        coords = np.hstack([core_coords, edge_coords, residual_coords])
        annot = [2] * len(core_coords[0]) + \
                [1] * len(edge_coords[0]) + \
                [0] * len(residual_coords[0])
        self.mask = np.zeros(self.mask.shape)
        self.mask[edge == 1] = 1
        self.mask[core == 1] = 2

        self.cache.pixel_data['xcoord'] = coords[0]
        self.cache.pixel_data['ycoord'] = coords[1]
        self.cache.pixel_data['annotation'] = annot

    def annotate_particles(self,
                           percentile_norm_params=(0, 100, 2000, 10000),
                           show_outline=False,
                           outline_color='r',
                           classes={1: 'single cell', 2: 'microcolony', 3: 'others'},
                           saturation=0.85):
        norm = (generic.percentile_normalize(self.get_ref_image(),
                                             percentile_norm_params[0],
                                             percentile_norm_params[1],
                                             percentile_norm_params[2],
                                             percentile_norm_params[3]) * int(saturation * 255)).astype(np.uint8)
        thumbnails = []
        if '$annotation' in self.regionprops.columns:
            annotation = self.regionprops['$annotation'].values
        else:
            annotation = np.zeros(len(self.regionprops))
        for (x0, y0, x1, y1, c) in self.regionprops[['$opt-x1', '$opt-y1', '$opt-x2', '$opt-y2', '$outline']].values:
            if show_outline:
                fig = plt.figure(figsize=(4, 4))
                plt.imshow(self.get_ref_image()[x0:x1, y0:y1], cmap='gist_gray')
                plt.plot(c[0][:, 1], c[0][:, 0], color=outline_color, lw=0.8)
                plt.axis('off')
                img_buf = io.BytesIO()
                plt.savefig(img_buf, bbox_inches='tight')
                thumbnails.append(PIL.Image.open(img_buf))
                plt.close()
            else:
                thumbnails.append(PIL.Image.fromarray(norm[x0:x1, y0:y1]))
        run_particle_labeler(thumbnails, label_dict=classes, annot=annotation)
        self.regionprops['$annotation'] = annotation

    def generate_multi_mask(self, model='default',
                              method=1,
                              channels=[], **kwargs):
        """

        Parameters
        ----------
        model
        method
        channels
        kwargs

        Returns
        -------

        """
        if len(channels) > len(self.channels):
            raise ValueError('Patch image had {} channels but {} channels were provided'.format(len(self.channels),
                                                                                                len(channels)))
        if len(channels) == 0:
            input_images = [self.get_ref_image()]

        elif len(channels) >= 1:
            stack = []
            for c in channels:
                if c in self.data:
                    stack.append(self.data[c])
                else:
                    raise ValueError('Patch image does not have channel "{}".'.format(c))
            input_images = [np.array(stack)]

        if isinstance(model, str):
            if model == 'default':
                model = BinarizeLegend(method=method, config=self.config)
            else:
                raise ModuleNotFoundError('Mask method {} not found!'.format(model))
        mask = MakeMask(model, **kwargs).predict(input_images)[0]
        self.mask = mask


class PatchCache:

    def __init__(self):
        self.gaussians = {}
        self.feature_images = {}
        self.feature_columns = []
        self.pixel_data = pd.DataFrame()

    def prepare_feature_images(self, img, **kwargs):
        self.gaussians, self.feature_images = multiscale_image_feature(img, **kwargs)

    def extract_local_features(self, use_Gaussian=True, **kwargs):
        if len(self.pixel_data) == 0:
            raise ValueError('Pixel coordinates not defined!')
        xcoord, ycoord = self.pixel_data[['xcoord', 'ycoord']].values.T
        feature_images = self.feature_images
        if use_Gaussian:
            for k, g in self.gaussians.items():
                feature_images['Gaussian_{}'.format(k)] = g
        local_pixel_features = local_stat(feature_images, xcoord, ycoord, **kwargs)
        self.feature_columns = local_pixel_features.columns
        self.pixel_data[self.feature_columns] = local_pixel_features.values

def get_regionprop(labeled_image, intensity_image=None):
    regionprop_properties = ['label','bbox','coords','centroid',
                             'area','convex_area','filled_area',
                             'eccentricity','solidity',
                             'major_axis_length','minor_axis_length',
                             'perimeter','equivalent_diameter',
                             'extent','orientation',
                             'inertia_tensor','moments_hu','weighted_moments_hu',
                             'mean_intensity','max_intensity','min_intensity']

    if intensity_image is not None:
        rp_table = pd.DataFrame(measure.regionprops_table(labeled_image,
                                                          intensity_image,
                                                          properties=regionprop_properties))
    else:
        rp_table = pd.DataFrame(measure.regionprops_table(labeled_image,
                                                          intensity_image=(labeled_image>0)*1,
                                                          properties=regionprop_properties))
    areas = rp_table['area'].values
    perimeter = rp_table['perimeter'].values
    rp_table['compactness'] = (perimeter**2)/(4*np.pi*areas)
    discriminant = np.maximum(perimeter**2 - 16 * areas, 0)
    rp_table['rough_length'] = (perimeter - np.sqrt(discriminant)) / 4
    rp_table['aspect_ratio'] = np.round(rp_table['minor_axis_length'].values/rp_table['major_axis_length'].values,3)
    rp_table.rename(columns={'bbox-0':'$bbox-0',
                             'bbox-1':'$bbox-1',
                             'bbox-2':'$bbox-2',
                             'bbox-3':'$bbox-3',
                             'coords':'$coords',
                             'label':'$label',
                             'centroid-0':'$centroid-0',
                             'centroid-1':'$centroid-1'},
                    inplace=True)
    return rp_table

    """
    def load_config(self, config):
        if 'ConfigParser' in str(type(config)):
            self.config = config
        else:
            self.config = Config(config).config
    """


