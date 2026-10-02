import numpy as np

from ..utils import *
from ..core.patch import Patch

class Particle:

    def __init__(self,
                 bbox,
                 mask,
                 data,
                 ref_channel,
                 particle_id,
                 offset,
                 pixel_microns=0.065,
                 outline=np.array([[]])):
        self.bbox = bbox
        self.outline = outline
        self.mask = mask
        self.x,self.y = np.where(mask>0)
        self.data = data
        self.ref_channel = ref_channel
        self.pixel_microns = pixel_microns
        self.id = particle_id
        self.sobel = None
        self.branched = False
        self.discarded = False
        self.skeleton = None
        self.skeleton_coords = {}
        self.midlines = {}
        self.branch_lengths = {}
        self.width_lists = {}
        self.length = 0
        self.config = None
        self.offset = offset
        self.basic_feature_attributes = ['area', 'eccentricity', 'solidity', 'touching_edge',
                                         'convex_area', 'filled_area', 'eccentricity', 'solidity',
                                         'major_axis_length', 'minor_axis_length', 'perimeter',
                                         'equivalent_diameter', 'extent','circularity', 'aspect_ratio',
                                         'moments_hu-0', 'moments_hu-1', 'moments_hu-2', 'moments_hu-3',
                                         'moments_hu-4', 'moments_hu-5', 'moments_hu-6', 'aspect_ratio',
                                         'bending_std', 'bending_90', 'bending_10', 'bending_max',
                                         'bending_min', 'bending_skewness', 'bending_median', 'touching_edge']
        self.basic_features = []

    def _load_basic_features(self, parent_regionprop, basic_attributes):
        self.basic_feature_attributes = basic_attributes
        self.basic_features = parent_regionprop[basic_attributes].values

    def get_ref_image(self):
        return self.data[self.ref_channel]