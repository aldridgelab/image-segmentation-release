from scipy.ndimage import median_filter
from ..segment import prediction2foreground, prediction2seed
from ..utils import *
from skimage import filters, morphology, measure, feature, segmentation
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class CellTracker:

    def __init__(self,
                 sorted_patches,  # load patches patches as list
                 time_points,
                 pixel_classifier,
                 seed_prob_min=0.2,
                 edge_prob_max=0.9,
                 background_prob_max=0.2,
                 min_overlap_threshold=0.5,
                 min_overlap_area=100,
                 min_size_similarity=0.75,
                 min_cell_size=200,
                 min_seed_size=10,
                 max_iter=5,
                 cache_image_features=False,
                 verbose=True,
                 backtrack_generations=4,
                 no_merge=False,
                 no_split=False,
                 hard_split=False):

        if len(time_points) != len(sorted_patches):
            raise ValueError("The number of time points given doesn't match the number of image patches!")
        self.timepoints = time_points
        self.frames = sorted_patches
        self.pixel_classifier = pixel_classifier
        self.seed_prob_min = seed_prob_min
        self.edge_prob_max = edge_prob_max
        self.background_prob_max = background_prob_max
        self.min_seed_size = min_seed_size
        self.cache_in = cache_image_features

        self.min_overlap_threshold = min_overlap_threshold
        self.min_overlap_area = min_overlap_area
        self.min_size_similarity = min_size_similarity
        self.min_cell_size = min_cell_size
        self.backtrack_generations = backtrack_generations
        self.max_iter = max_iter

        self.regionprops = None
        self.overlap_stats = None
        self.to_be_merged = [-1]
        self.to_be_split = [-1]
        self.verbose = verbose
        self.no_merge = no_merge
        self.no_split = no_split
        self.hard_split = hard_split

    def locate_spurious_events(self):
        to_be_split, to_be_merged = _locate_spurious_events(self.regionprops, verbose=self.verbose)
        self.to_be_split = to_be_split
        self.to_be_merged = to_be_merged

    def update_masks(self):
        if len(self.to_be_merged) > 0:
            if not self.no_merge:
                for c in self.to_be_merged:
                    self._merge_cells(c)
        if len(self.to_be_split) > 0:
            if not self.no_split:
                for c in self.to_be_split:
                    self._split_cells(c)

    def update_frames(self):
        _regionprops = []
        for i, p in enumerate(self.frames):
            p.locate_particles()
            p.regionprops['$cell'] = ['{}_{}'.format(i, l) for l in p.regionprops.index]
            p.regionprops['$time'] = i
            _regionprops.append(p.regionprops)
        self.regionprops = pd.concat(_regionprops).set_index('$cell')

    def _update_regionprops(self):
        _regionprops = []
        for i, p in enumerate(self.frames):
            _regionprops.append(p.regionprops)
            p.regionprops['$cell'] = ['{}_{}'.format(i, l) for l in p.regionprops.index]
        _regionprops = pd.concat(_regionprops).set_index('$cell')
        for col in _regionprops.columns:
            if col not in self.regionprops.columns:
                self.regionprops[col] = _regionprops.loc[self.regionprops.index][col]
        del _regionprops

    def trace_by_overlap(self):
        counter = 0
        while counter <= self.max_iter:
            self.update_frames()
            self.link_cells_by_overlap()
            self.locate_spurious_events()
            self.update_masks()
            if self._end_iter():
                self.update_frames()
                self.link_cells_by_overlap()
                break
            counter += 1

    def link_cells_by_overlap(self):
        self.regionprops, self.overlap_stats = overlap_match(self.frames,
                                                             self.regionprops,
                                                             min_cell_size=self.min_cell_size,
                                                             min_size_similarity=self.min_size_similarity,
                                                             min_overlap_threshold=self.min_overlap_threshold,
                                                             min_overlap_area=self.min_overlap_area)

    def _end_iter(self):
        end = False
        if (len(self.to_be_merged) + len(self.to_be_split)) == 0:
            end = True
        return end

    def _merge_cells(self, cell_list):

        coords = self.regionprops.loc[np.array(cell_list)]['$coords']
        dilated_coords = [dilate_by_coords(x, self.frames[0].shape, morphology.disk(1)) for x in coords]
        unique_coords, pix_count = np.unique(np.vstack(dilated_coords), return_counts=True, axis=0)
        border_coords = unique_coords[pix_count >= 2]
        merged_coords = np.vstack(list(coords) + [border_coords])
        time = np.unique([int(x.split('_')[0]) for x in cell_list])[0]
        new_label = np.min([int(x.split('_')[1]) for x in cell_list])
        self.frames[time].labeled_mask[merged_coords[:, 0], merged_coords[:, 1]] = new_label

    def _split_cells(self, cell_ref):
        cell, n_daughter = cell_ref
        time, label = np.array(cell.split('_')).astype(int)
        x, y = self.regionprops.loc[cell]['$coords'].T
        cropped_prob = np.zeros(self.frames[time].prob_mask.shape)
        cropped_mask = np.zeros(self.frames[time].mask.shape)
        cropped_mask[x, y] = 1
        cropped_prob[x, y, :] = self.frames[time].prob_mask[x, y, :]

        for th in np.linspace(self.seed_prob_min, 1, 5):
            seed = prediction2seed(cropped_prob,
                                   seed_min=th,
                                   edge_max=self.edge_prob_max,
                                   min_seed_size=self.min_seed_size)

            if seed.max() >= n_daughter:
                break

        if seed.max() != n_daughter and self.hard_split:
            seed = np.zeros(self.frames[time].mask.shape)
            current_mask = self.frames[time].labeled_mask == label
            prev_frame = (self.frames[time - 1].labeled_mask) * current_mask
            counter = 0
            for d in np.unique(prev_frame):
                if d != 0:
                    if (np.sum(prev_frame == d) / len(x)) > 0.1:
                        seed[prev_frame == d] = counter + 1
                        counter += 1
        if seed.max() == n_daughter:
            seed[seed == 1] = self.frames[time].labeled_mask.max() + 1
            seed[seed == 2] = self.frames[time].labeled_mask.max() + 2
            cropped_watershed = segmentation.watershed(image=cropped_prob[:, :, 1],
                                                       mask=cropped_mask,
                                                       markers=seed,
                                                       connectivity=1,
                                                       compactness=0.1,
                                                       watershed_line=False)
            new_labeled_mask = self.frames[time].labeled_mask.copy()
            new_labeled_mask[x, y] = cropped_watershed[x, y]
            self.frames[time].labeled_mask = new_labeled_mask

    def plot_cell(self, cell_id):
        show_highlight(cell_id, self.frames)

    def trace_lineage(self):
        rev_lineage = {}
        init_cell_counter = 1
        for c in self.regionprops.index:
            daughters = self.regionprops.loc[c]['daughter(s)']
            if c not in rev_lineage:
                rev_lineage[c] = str(init_cell_counter)
                init_cell_counter += 1
            if len(daughters) == 1:
                rev_lineage[daughters[0]] = rev_lineage[c]
            else:
                for i, d in enumerate(daughters):
                    rev_lineage[d] = '{}.{}'.format(rev_lineage[c], i + 1)
        lineage = [rev_lineage[c] for c in self.regionprops.index]
        self.regionprops['cell_lineage'] = lineage

    def refine_trace(self):
        _refine_trajectories(self,
                             seed_threshold=self.seed_prob_min,
                             force_split=self.hard_split,
                             trace_generations=self.backtrack_generations)

    def link_cells(self):
        _link_cells(self)


def _refine_trajectories(tracker,
                         seed_threshold=0.7,
                         force_split=True,
                         trace_generations=3):
    from skimage import measure, segmentation, morphology
    to_be_split, to_be_merged = _locate_spurious_events(tracker.regionprops, verbose=False)
    new_labels = [f.labeled_mask.copy() for f in tracker.frames]  # copy mask
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    for pair in to_be_merged:
        t, l1 = pair[0].split('_')
        t, l2 = pair[1].split('_')
        t = int(t)
        coords1 = tracker.regionprops.loc[pair[0], '$coords']
        coords2 = tracker.regionprops.loc[pair[1], '$coords']
        x, y = np.vstack([coords1, coords2]).T
        max_label = new_labels[t].max()
        new_labels[t][x, y] = max_label + 1

    for cell in to_be_split:
        t, l1 = cell.split('_')
        t = int(t)
        prob = tracker.frames[t].prob_mask[:, :, 2]
        mask = tracker.frames[t].labeled_mask == int(l1)
        eroded_mask = morphology.erosion(mask)
        masked_prob = prob * eroded_mask
        sink = 1 - prob
        seed_found = False
        for th in np.linspace(seed_threshold, 1, 10):
            seeds = masked_prob > th
            labeled_seeds = measure.label(seeds)
            if labeled_seeds.max() == 2:
                seed_found = True
                break
        if not seed_found and force_split:
            parents = back_track(tracker.regionprops, cell, n_generations=trace_generations)
            daughters = forward_track(tracker.regionprops, cell, n_generations=trace_generations)
            nearest_split_event = [np.inf, []]
            for p in parents:
                if len(p) == 2:
                    _t = int(p[0].split('_')[0])
                    dt = np.abs(_t - t)
                    if dt < nearest_split_event[0]:
                        nearest_split_event = [_t, p]
                    break
            for p in daughters:
                if len(p) == 2:
                    _t = int(p[0].split('_')[0])
                    dt = np.abs(_t - t)
                    if dt < nearest_split_event[0]:
                        nearest_split_event = [_t, p]
                    break
            seeds = np.zeros(mask.shape)
            if len(nearest_split_event[1]) > 0:
                x1, y1 = tracker.regionprops.loc[nearest_split_event[1][0], '$coords'].T
                x2, y2 = tracker.regionprops.loc[nearest_split_event[1][1], '$coords'].T
                seeds[x1, y1] = 1
                seeds[x2, y2] = 2
                labeled_seeds = measure.label(seeds) * mask
                seed_found = True
        if seed_found:
            watersheded = segmentation.watershed(image=sink, markers=labeled_seeds, mask=mask, compactness=100)
            for i in np.unique(watersheded):
                max_label = new_labels[t].max()
                if i > 0:
                    new_labels[t][watersheded == i] = max_label + i
        else:
            print('Rectification failed for cell {}.'.format(cell))
    for i, p in enumerate(tracker.frames):
        # remove tiny chunks that don't belong
        labeled_mask = morphology.remove_small_objects(new_labels[i].copy(), tracker.min_seed_size)
        p.labeled_mask = labeled_mask
        p.regionprops = None
        p.locate_particles(precompute_contours=False)


def back_track(regionprops,
               cell_id,
               n_generations=3):
    # the backtrack algorithm is based on the naive hypothesis that the fragments of an oversegmented cell
    # should have a common ancester tracing back n generations whereas an undersegmented cell should not.
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    time, label = np.array(cell_id.split('_')).astype(int)
    elderlist = []
    current_cell = np.array([cell_id])
    while True:
        if n_generations == 0 or time == 0:
            break
        mothers = np.unique(np.hstack(regionprops.loc[current_cell]['mother(s)'].values))
        current_cell = mothers
        n_generations -= 1
        time -= 1
        if len(mothers) > 0:
            elderlist.append(mothers)
        else:
            break
    return elderlist


def forward_track(regionprops,
                  cell_id,
                  n_generations=3):
    # the backtrack algorithm is based on the naive hypothesis that the fragments of an oversegmented cell
    # should have a common ancester tracing back n generations whereas an undersegmented cell should not.
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    time, label = np.array(cell_id.split('_')).astype(int)
    younger_list = []
    current_cell = np.array([cell_id])
    while True:
        if n_generations == 0 or time == regionprops['$time'].max():
            break
        daughters = np.unique(np.hstack(regionprops.loc[current_cell]['daughter(s)'].values))
        current_cell = daughters
        n_generations -= 1
        time += 1
        if len(daughters) > 0:
            younger_list.append(daughters)
        else:
            break
    return younger_list


def forward_track_exhaustive(regionprops,
                             cell_id,
                             additional_search=5,
                             max_n_daughter=4):
    # the backtrack algorithm is based on the naive hypothesis that the fragments of an oversegmented cell
    # should have a common ancester tracing back n generations whereas an undersegmented cell should not.
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    time, label = np.array(cell_id.split('_')).astype(int)
    younger_list = []
    current_cell = np.array([cell_id])
    start_count = False
    while True:
        if time == regionprops['$time'].max():
            break
        if additional_search == 0:
            break
        daughters = np.unique(np.hstack(regionprops.loc[current_cell]['daughter(s)'].values))
        current_cell = daughters
        time += 1
        if len(daughters) >= max_n_daughter:
            start_count = True
        if start_count:
            additional_search -= 1
        if len(daughters) > 0:
            younger_list.append(daughters)
        else:
            break
    return younger_list


def _link_cells(tracker):
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    init_stat = {}
    mother_list = []
    daughter_list = []
    for t in range(len(tracker.frames) - 1):
        linking_matrix, trans_matrix, overlap_stat, labels1, labels2 = silly_link(tracker, t)
        for i, l in enumerate(labels1):
            k = '{}_{}'.format(t, l)
            if k not in init_stat:
                init_stat[k] = [[], []]
            daughters = ['{}_{}'.format(t + 1, labels2[j]) for j in np.where(linking_matrix[i] > 0)[0]]
            init_stat[k][1] = daughters
        for j, l in enumerate(labels2):
            k = '{}_{}'.format(t + 1, l)
            if k not in init_stat:
                init_stat[k] = [[], []]
            mothers = ['{}_{}'.format(t, labels1[i]) for i in np.where(linking_matrix[:, j] > 0)[0]]
            init_stat[k][0] = mothers
        tracker.frames[t].regionprops['$Tracker_ID'] = ['{}_{}'.format(t, l) for l in
                                                        tracker.frames[t].regionprops.index]
        tracker.frames[t].regionprops['$time'] = t
    tracker.frames[-1].regionprops['$Tracker_ID'] = ['{}_{}'.format(t + 1, l) for l in
                                                     tracker.frames[-1].regionprops.index]
    tracker.frames[-1].regionprops['$time'] = t + 1
    merged_rp = pd.concat([f.regionprops for f in tracker.frames]).set_index('$Tracker_ID').copy()
    mother_lists = [[init_stat[k][0], len(init_stat[k][0])] for k in merged_rp.index]
    daughter_lists = [[init_stat[k][1], len(init_stat[k][1])] for k in merged_rp.index]
    merged_rp[['mother(s)', 'n_mother']] = mother_lists
    merged_rp[['daughter(s)', 'n_daughter']] = daughter_lists
    tracker.regionprops = merged_rp
    return None


def area_ratio(areas):
    a1, a2 = areas
    if a1 > a2:
        ratio = a2 / a1
    else:
        ratio = a1 / a2
    return ratio


def show_highlight(cell_id, frames):
    t, label = np.array(cell_id.split('_')).astype(int)
    mask = (frames[t].labeled_mask > 0) * 1
    mask[frames[t].labeled_mask == label] = 3
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(mask)


def overlap_match(frames, regionprops,
                  min_overlap_threshold=0.5,
                  min_overlap_area=100,
                  min_size_similarity=0.75,
                  min_cell_size=200):
    # link cells by overlap
    init_stat = {k: [[], []] for k in regionprops.index}
    mother_list = []
    daughter_list = []
    rp = regionprops.copy()
    overlap_stat = []
    for t in range(0, len(frames) - 1):
        df = _calculate_overlap_matrix(frames[t].labeled_mask,
                                       frames[t + 1].labeled_mask, t, t + 1)
        filtered_df = df[
            np.max(df[['frame1_overlap_frac', 'frame2_overlap_frac']].values, axis=1) > min_overlap_threshold]
        for (cell1, cell2) in filtered_df[['frame1_id', 'frame2_id']].values:
            init_stat[cell1][1] += [cell2]
            init_stat[cell2][0] += [cell1]
        overlap_stat.append(df)

    overlap_stat = pd.concat(overlap_stat)

    # find missing mother(s) for cells that moved a bit
    orphans_candidates = rp[(rp['$time'] > 0) & (rp['$touching_edge'] == 0)].index
    for o in orphans_candidates:
        if len(init_stat[o][0]) == 0 and len(init_stat[o][1]) > 0:
            subset = overlap_stat[overlap_stat['frame2_id'] == o].values
            missing_mother = []
            for s in subset:
                cond1 = s[2] > min_overlap_area
                cond2 = area_ratio((s[3], s[4])) > min_size_similarity
                cond3 = min(s[3], s[4]) > min_cell_size
                if cond1 * cond2 * cond3:
                    missing_mother += [s[0]]
                    init_stat[s[0]][1] += [o]
            init_stat[o][0] = missing_mother
    rp[['mother(s)', 'n_mother']] = [[np.unique(init_stat[k][0]), len(np.unique(init_stat[k][0]))] for k in rp.index]
    rp[['daughter(s)', 'n_daughter']] = [[np.unique(init_stat[k][1]), len(np.unique(init_stat[k][1]))] for k in
                                         rp.index]
    return rp, overlap_stat


def _calculate_overlap_matrix(frame1_labeled_mask,
                              frame2_labeled_mask,
                              frame1_label,
                              frame2_label):
    f1, f2 = frame1_labeled_mask.ravel(), frame2_labeled_mask.ravel()
    f1f2 = np.array([f1, f2]).T[f1 * f2 != 0]
    f1_counts = _unique2dict1D(f1)
    f2_counts = _unique2dict1D(f2)
    neighbors, overlap = np.unique(f1f2, return_counts=True, axis=0)
    f1_areas = np.array([f1_counts[i] for i in neighbors[:, 0]])
    f2_areas = np.array([f2_counts[i] for i in neighbors[:, 1]])

    f1_id = np.array(['{}_{}'.format(frame1_label, i) for i in neighbors[:, 0]])
    f2_id = np.array(['{}_{}'.format(frame2_label, i) for i in neighbors[:, 1]])
    overlap_matrix = np.hstack([f1_id.reshape(-1, 1),
                                f2_id.reshape(-1, 1),
                                overlap.reshape(-1, 1),
                                f1_areas.reshape(-1, 1),
                                f2_areas.reshape(-1, 1),
                                (overlap / f1_areas).reshape(-1, 1),
                                (overlap / f2_areas).reshape(-1, 1),
                                (overlap / (f1_areas + f2_areas - overlap)).reshape(-1, 1)])
    overlap_df = pd.DataFrame()
    overlap_df['frame1_id'] = f1_id
    overlap_df['frame2_id'] = f2_id
    overlap_df['overlap_area'] = overlap
    overlap_df['frame1_area'] = f1_areas
    overlap_df['frame2_area'] = f2_areas
    overlap_df['frame1_overlap_frac'] = overlap / f1_areas
    overlap_df['frame2_overlap_frac'] = overlap / f2_areas
    overlap_df['iou'] = overlap / (f1_areas + f2_areas - overlap)
    return overlap_df


def _unique2dict1D(array, nonzero=True):
    copied = array.copy()
    if nonzero:
        copied = copied[copied != 0]
    vals, counts = np.unique(copied, return_counts=True)
    return {v: c for v, c in zip(vals, counts)}


def _locate_spurious_events(regionprops,
                            n_generations=10,
                            window_size=5,
                            verbose=True):
    from scipy.stats import mode
    import warnings
    warnings.filterwarnings("ignore")
    to_be_merged = []
    to_be_split = []

    for cell in regionprops[regionprops['n_mother'] >= 2].index:
        back_track_record = np.flip(back_track(regionprops, cell, n_generations=n_generations), axis=0)
        forward_track_record = forward_track_exhaustive(regionprops, cell, max_n_daughter=2, additional_search=10)

        back_divergence = [len(x) for x in back_track_record]
        forward_divergence = [len(x) for x in forward_track_record]
        trace = list(back_track_record) + [np.array([cell])] + list(forward_track_record)
        counts = np.array(back_divergence + [1] + forward_divergence).astype(float)
        window_size = min(int(len(counts) / 2) * 2 + 1, window_size)
        if len(counts) >= window_size:
            convolved_counts = counts.copy()
            pad = int(window_size / 2)
            convolved_counts[pad:-pad] = np.convolve(counts, np.ones(window_size) / window_size, mode='valid')
            convolved_counts = np.round(convolved_counts).astype(int)
            # to_be_merged.append(convolved_counts)
            # to_be_split.append(counts)

            l = int(len(convolved_counts) / 2)
            v1 = convolved_counts[:l].mean()
            v2 = convolved_counts[-l:].mean()

            if v1 < 0.6 * v2:  # there should be a division event
                try:
                    turn_point = find_turn_point(convolved_counts)
                except:
                    turn_point = -1
                if turn_point >= 0:
                    if counts[:(turn_point + 1)].mean() < 1.5:
                        for i in np.where(counts[:(turn_point + 1)] > 1)[0]:
                            to_be_merged.append(trace[i].astype("<U22"))

                        for j in \
                        np.where((convolved_counts[(turn_point + 1):] <= 2) & (counts[(turn_point + 1):] == 1))[0]:
                            to_be_split.append(trace[j + turn_point + 1])
                    else:
                        for j in \
                        np.where((convolved_counts[:(turn_point + 1)] <= 2) & (counts[:(turn_point + 1)] == 1))[0]:
                            to_be_split.append(trace[j].astype("<U22"))
            else:  # not detected division event
                count_Mode = mode(counts)[0][0]
                if count_Mode == 1:
                    for i in np.where(counts > 1)[0]:
                        if len(trace[i]) == 2:
                            to_be_merged.append(trace[i].astype("<U22"))
                elif count_Mode == 2:
                    for j in np.where(counts == 1)[0]:
                        to_be_split.append(trace[j].astype("<U22"))
    new_split = []
    for x in to_be_split:
        x = x[0]
        if x not in new_split:
            new_split.append(x)
    new_merge = []
    for x in to_be_merged:
        x = list(x)
        if x not in new_merge:
            new_merge.append(x)
    # to_be_split = np.unique(np.array(to_be_split).astype("<U22"))
    # to_be_merged = np.unique(np.array(to_be_merged).astype("<U22"), axis=0)
    return new_split, new_merge


def find_turn_point(data, kernel=np.array([0.5, 0.5, 0, -0.5, -0.5])):
    pad = int(len(kernel) / 2)
    convolved = np.convolve(data, kernel, mode='valid')
    convolved = np.pad(convolved, ((pad, pad),))
    turn_point = np.where(convolved == 1)[0][0]
    return turn_point


def generate_correlation_map(x, y):
    """Correlate each n with each m.
    @abcd
    @https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30145770#30145770
    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                   mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def transition_matrix(frames, t,
                      moments=['moments_hu-0', 'moments_hu-1', 'moments_hu-2',
                               'moments_hu-3', 'moments_hu-4', 'moments_hu-5', 'moments_hu-6'],
                      features=['area', 'major_axis_length', 'minor_axis_length', 'extent', 'aspect_ratio'],
                      normalize=True):
    """
    calcluate transition matrix
    channel0: Intersect(A[t],A[t+1])/Union(A[t],A[t+1])
    channel1: Intersect(A[t],A[t+1])/A[t]
    channel2: Intersect(A[t],A[t+1])/A[t+1]
    channel3: hu moments correlation matrix
    channel4: feature difference matrix
    channel5: centroid distance matrix
    channel6: nearest neighbor

    :params frames: momia.core.CellTracker.frames object (list)
    :params time: time
    :params features: rotation invariant features used to estimate pairwise correlation
    :return estimated transtion matrix and the two cell_id to index referenc dictionaries.
    """
    from sklearn.preprocessing import StandardScaler as ss
    f1 = frames[t]
    f2 = frames[t + 1]
    n1 = len(f1.regionprops)
    n2 = len(f2.regionprops)
    f1name_dict = {'{}_{}'.format(t, x): i for i, x in enumerate(f1.regionprops.index)}
    f2name_dict = {'{}_{}'.format(t + 1, x): i for i, x in enumerate(f2.regionprops.index)}
    area1 = f1.regionprops['area'].values
    area2 = f2.regionprops['area'].values

    # weights by overlap, channels 0-2
    transMatrix = np.zeros((n1, n2, 7))
    overlap_stat = _calculate_overlap_matrix(f1.labeled_mask, f2.labeled_mask, t, t + 1)
    for idx1, idx2, iou, frac1, frac2 in overlap_stat[
        ['frame1_id', 'frame2_id', 'iou', 'frame1_overlap_frac', 'frame2_overlap_frac']].values:
        transMatrix[f1name_dict[idx1], f2name_dict[idx2], 0] = iou
        transMatrix[f1name_dict[idx1], f2name_dict[idx2], 1] = frac1
        transMatrix[f1name_dict[idx1], f2name_dict[idx2], 2] = frac2

    # weights by hu moments
    m1 = f1.regionprops[moments].values
    m2 = f2.regionprops[moments].values

    if normalize:
        norm_v = ss().fit_transform(np.vstack([m1, m2]))
        m1 = norm_v[:n1]
        m2 = norm_v[-n2:]
    corr = generate_correlation_map(m1, m2)
    corr[corr < 0] = 0
    transMatrix[:, :, 3] = corr

    v1 = f1.regionprops[features].values
    v2 = f2.regionprops[features].values
    transMatrix[:, :, 4] = difference_matrix(v1, v2)

    cent1 = np.array([np.mean(x, axis=0) for x in f1.regionprops['$coords'].values])
    cent2 = np.array([np.mean(x, axis=0) for x in f2.regionprops['$coords'].values])
    dist = linalg.distance_matrix(cent1, cent2)
    argmin_dist = np.argmin(dist, axis=1)
    transMatrix[:, :, 5] = dist_prob(dist)
    transMatrix[:, :, 6] = 1 - (
                2 * np.abs(area1[:, np.newaxis] - area2[np.newaxis]) / (area1[:, np.newaxis] + area2[np.newaxis]))
    return transMatrix, f1name_dict, f2name_dict, overlap_stat


def silly_link(tracker_obj, t,
               iou_threshold=0.7,
               overlap_threshold=0.7,
               similarity_threshold=0.9,
               diff_threshold=0.2,
               logprob_threshold=4,
               min_iou_threshold=0.1):
    frames = tracker_obj.frames
    trans_matrix, f1n, f2n, overlap_stat = transition_matrix(frames, t, normalize=True)
    f1 = frames[t]
    f2 = frames[t + 1]
    n1 = len(f1.regionprops)
    n2 = len(f2.regionprops)
    labels1 = f1.regionprops.index
    labels2 = f2.regionprops.index
    matched_l1 = np.zeros(n1)
    matched_l2 = np.zeros(n2)
    area1 = f1.regionprops['area'].values
    area2 = f2.regionprops['area'].values

    linking_matrix = np.zeros((n1, n2))
    cond1 = trans_matrix[:, :, 0] > iou_threshold
    cond2 = trans_matrix[:, :, 4] > similarity_threshold
    cond3 = trans_matrix[:, :, 6] > similarity_threshold
    linking_matrix[cond1 & cond2 & cond3] = 1
    matched_l1[np.sum(linking_matrix, axis=1) > 0] = 1
    matched_l2[np.sum(linking_matrix, axis=0) > 0] = 1
    f1_remnant = np.where(matched_l1 == 0)[0]
    f2_remnant = np.where(matched_l2 == 0)[0]

    log_prob_matrix = np.sum(np.log2(trans_matrix + 1), axis=-1)
    log_dist_matrix = np.sum(np.log2(trans_matrix[:, :, np.array([0, 1, 2, 5])] + 1), axis=-1)

    # fig=plt.figure(figsize=(4,4))
    # plt.imshow(linking_matrix)
    for i in f1_remnant:
        cond1 = log_prob_matrix[i] > logprob_threshold
        cond2 = matched_l2 == 0
        cond3 = trans_matrix[i, :, 0] > min_iou_threshold
        cond4 = trans_matrix[i, :, 6] > similarity_threshold
        future_self_id = np.where(cond1 & cond2 & cond3 & cond4)[0]
        if len(future_self_id) >= 1:
            j = future_self_id[np.argmax(log_prob_matrix[i, future_self_id])]
            linking_matrix[i, j] = 1
            matched_l1[i] = 1
            matched_l2[j] = 1

    f1_remnant = np.where(matched_l1 == 0)[0]
    # cases when 2 or more particles on frame t map to one particle on frame t+1
    for i in f1_remnant:
        cond1 = trans_matrix[i, :, 1] > overlap_threshold
        cond2 = trans_matrix[i, :, 1] > trans_matrix[i, :, 2]
        cond3 = matched_l2 != 1
        future_self_id = np.where(cond1 & cond2 & cond3)[0]
        if len(future_self_id) == 1:
            j = future_self_id[0]
            matched_l1[i] = 1
            matched_l2[j] = 2  # N->1 event, note that the sibling particle won't necessarily overlap with j
            linking_matrix[i, j] = 1

    # if areas match, linkage is complete
    for j in np.where(matched_l2 == 2)[0]:
        f2_area = area2[j].sum()
        f1_ids = np.where(linking_matrix[:, j] == 1)[0]
        if len(f1_ids) > 1:
            f1_area = area1[f1_ids].sum()
            if np.abs(2 * (f1_area - f2_area) / (f1_area + f2_area)) < diff_threshold:
                matched_l2[j] = 1
        elif len(f1_ids) == 1:
            l1_1 = f1_ids[0]
            l1_1_coords = f1.regionprops.loc[labels1[l1_1], '$coords']
            for l1_2 in np.flip(np.argsort(log_dist_matrix[:, j])):
                cond1 = l1_2 != l1_1
                cond2 = matched_l1[l1_2] == 0
                # l2_1 and l2_2 should be sitting next to each other
                if cond1 and cond2:
                    l1_2_coords = f1.regionprops.loc[labels1[l1_2], '$coords']
                    if linalg.distance_matrix(l1_1_coords, l1_2_coords).min() < 2:
                        f1_area = area1[np.array([l1_1, l1_2])].sum()
                        if np.abs(2 * (f1_area - f2_area) / (f1_area + f2_area)) < diff_threshold:
                            matched_l1[l1_2] = 1
                            matched_l2[j] = 1
                            linking_matrix[l1_2, j] = 1
                            break

    # cases when 2 or more particles on frame t+1 map to one particle on frame t
    f2_remnant = np.where(matched_l2 == 0)[0]
    for j in f2_remnant:
        cond1 = trans_matrix[:, j, 2] > overlap_threshold
        cond2 = trans_matrix[:, j, 1] < trans_matrix[:, j, 2]
        cond2 = matched_l1 != 1
        prev_self_id = np.where(cond1 & cond2)[0]
        if len(prev_self_id) == 1:
            i = prev_self_id[0]
            matched_l1[i] = 2  # 1->N event, note that the sibling particle won't necessarily overlap with i
            matched_l2[j] = 1
            linking_matrix[i, j] = 1

    for i in np.where(matched_l1 == 2)[0]:
        f2_ids = np.where(linking_matrix[i, :] == 1)[0]
        f1_area = area1[i].sum()
        # if areas match, linkage is complete
        if len(f2_ids) > 1:
            f2_area = area2[f2_ids].sum()
            if np.abs(2 * (f1_area - f2_area) / (f1_area + f2_area)) < diff_threshold:
                matched_l1[i] = 1

        # find sibling, then if areas match, linkage is complete
        elif len(f2_ids) == 1:
            l2_1 = f2_ids[0]
            l2_1_coords = f2.regionprops.loc[labels2[l2_1], '$coords']
            for l2_2 in np.flip(np.argsort(log_dist_matrix[i])):
                cond1 = l2_2 != l2_1
                cond2 = matched_l2[l2_2] == 0
                # l2_1 and l2_2 should be sitting next to each other
                if cond1 and cond2:
                    l2_2_coords = f2.regionprops.loc[labels2[l2_2], '$coords']
                    if linalg.distance_matrix(l2_1_coords, l2_2_coords).min() < 2:
                        f2_area = area2[np.array([l2_1, l2_2])].sum()
                        if np.abs(2 * (f1_area - f2_area) / (f1_area + f2_area)) < diff_threshold:
                            matched_l1[i] = 1
                            matched_l2[l2_2] = 1
                            linking_matrix[i, l2_2] = 1
                            break

    f1_remnant = np.where(matched_l1 == 0)[0]
    f2_remnant = np.where(matched_l2 == 0)[0]
    # vsnap or any movement can be really annoying
    for i in f1_remnant:
        for j in np.flip(np.argsort(log_dist_matrix[i]))[:3]:
            if j in f2_remnant:
                if trans_matrix[i, j, 6] > similarity_threshold - 0.1:
                    matched_l1[i] = 1
                    matched_l2[j] = 1
                    linking_matrix[i, j] = 1
                    break
                elif area1[i] > area2[j]:
                    matched_l2[j] = 1
                    linking_matrix[i, j] = 1
                    f2_area = area2[np.where(linking_matrix[i] > 0)].sum()
                    if np.abs(2 * (area1[i] - f2_area) / (area1[i] + f2_area)) < diff_threshold:
                        matched_l1[i] = 1
                        break
                elif area1[i] < area2[j]:
                    matched_l1[i] = 1
                    linking_matrix[i, j] = 1
                    f1_area = area1[np.where(linking_matrix[:, j] > 0)].sum()
                    if np.abs(2 * (area2[j] - f1_area) / (area2[j] + f1_area)) < diff_threshold:
                        matched_l2[j] = 1
                        break

    # now we deal with the remaining unassigned particles
    f1_remnant = np.where(matched_l1 == 0)[0]
    f2_remnant = np.where(matched_l2 == 0)[0]

    for i in f1_remnant:
        if len(f2_remnant) > 0:
            feature_maxj = f2_remnant[np.argmax(trans_matrix[i, f2_remnant, 4])]
            moments_maxj = f2_remnant[np.argmax(trans_matrix[i, f2_remnant, 3])]
            if feature_maxj == moments_maxj and feature_maxj in f2_remnant:
                if trans_matrix[i, feature_maxj, 4] > similarity_threshold:
                    feature_maxi = f1_remnant[np.argmax(trans_matrix[f1_remnant, feature_maxj, 4])]
                    moments_maxi = f1_remnant[np.argmax(trans_matrix[f1_remnant, feature_maxj, 3])]
                    if feature_maxi == moments_maxi == i:
                        linking_matrix[i, j] = 1

    return linking_matrix, trans_matrix, overlap_stat, labels1, labels2