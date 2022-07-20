"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on two repositories:

Xinshuo Weng's AB3DMOT code at:
https://github.com/xinshuoweng/AB3DMOT/blob/master/evaluation/evaluate_kitti3dmot.py

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
import os
from typing import List, Dict, Callable, Tuple
import unittest



import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion

from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricDataList
from nuscenes.utils.data_classes import Box
import seaborn

import sklearn
import tqdm
import shutil

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

from nuscenes.eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks


class TrackingVisualizer:
    """
    This is the official nuScenes tracking evaluation code.
    Results are written to the provided output_dir.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/tracking for more details.
    """
    def __init__(self,
                 config: TrackingConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str,
                 nusc,
                 num_scenes):
        """
        Initialize a TrackingEval object.
        :param config: A TrackingConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param nusc_version: The version of the NuScenes dataset.
        :param nusc_dataroot: Path of the nuScenes dataset on disk.
        :param verbose: Whether to print to stdout.
        :param render_classes: Classes to render to disk or None.
        """
        self.cfg = config
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.num_scenes = num_scenes

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        print('Initializing nuScenes visualization')
        pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, TrackingBox, verbose=True)
        gt_boxes = load_gt(nusc, self.eval_set, TrackingBox, verbose=True)

        assert set(pred_boxes.sample_tokens) == set(gt_boxes.sample_tokens), \
            "Samples in split don't match samples in predicted tracks."

        # Add center distances.
        pred_boxes = add_center_dist(nusc, pred_boxes)
        gt_boxes = add_center_dist(nusc, gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, self.cfg.class_range, verbose=True)
        gt_boxes = filter_eval_boxes(nusc, gt_boxes, self.cfg.class_range, verbose=True)
        self.sample_tokens = gt_boxes.sample_tokens

        # Convert boxes to tracks format.
        self.tracks_gt = create_tracks(gt_boxes, nusc, self.eval_set, gt=True)
        self.tracks_pred = create_tracks(pred_boxes, nusc, self.eval_set, gt=False)

    def visualize(self, class_name):
        def accumulate_class(curr_class_name):
            curr_ev = TrackingVisualizerHelper(self.tracks_gt, self.tracks_pred, curr_class_name, self.cfg.dist_fcn_callable,
                                         self.cfg.dist_th_tp, self.cfg.min_recall,
                                         num_thresholds=TrackingMetricData.nelem,
                                         metric_worst=self.cfg.metric_worst,
                                         verbose=True,
                                         output_dir=self.output_dir)
            curr_md = curr_ev.visualize(self.num_scenes)

        accumulate_class(class_name)




class TrackingVisualizerHelper(object):
    def __init__(self,
                 tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
                 tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
                 class_name: str,
                 dist_fcn: Callable,
                 dist_th_tp: float,
                 min_recall: float,
                 num_thresholds: int,
                 metric_worst: Dict[str, float],
                 verbose: bool = True,
                 output_dir: str = None):
        """
        """
        self.tracks_gt = tracks_gt
        self.tracks_pred = tracks_pred
        self.class_name = class_name
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.num_thresholds = num_thresholds
        self.metric_worst = metric_worst
        self.verbose = verbose
        self.output_dir = output_dir
        self.n_scenes = len(self.tracks_gt)

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def visualize(self, num_scenes):
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        accumulators = []
        thresh_metrics = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        _, scores = self.accumulate_threshold(threshold=None, num_scenes=num_scenes)

    def accumulate_threshold(self, threshold: float = None, num_scenes: int = 1) -> Tuple[pandas.DataFrame, List[float]]:
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        accs = []
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        scene_iter = 0
        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
            os.makedirs(save_path, exist_ok=True)
            renderer = TrackingRenderer(save_path)

            prev_frame_gt = None
            prev_frame_pred = None
            prev_prev_frame_gt = None
            prev_prev_frame_pred = None
            has_preds = False
            for timestamp in scene_tracks_gt.keys():
                # Select only the current class.
                frame_gt = scene_tracks_gt[timestamp]
                frame_pred = scene_tracks_pred[timestamp]
                frame_gt = [f for f in frame_gt if f.tracking_name == self.class_name]
                frame_pred = [f for f in frame_pred if f.tracking_name == self.class_name]

                # Threshold boxes by score. Note that the scores were previously averaged over the whole track.
                if threshold is not None:
                    frame_pred = [f for f in frame_pred if f.tracking_score >= threshold]

                # Abort if there are neither GT nor pred boxes.
                gt_ids = [gg.tracking_id for gg in frame_gt]
                pred_ids = [tt.tracking_id for tt in frame_pred]
                if len(pred_ids) == 0:
                    continue
                has_preds = True

                # Calculate distances.
                # Note that the distance function is hard-coded to achieve significant speedups via vectorization.
                assert self.dist_fcn.__name__ == 'center_distance'
                if len(frame_gt) == 0 or len(frame_pred) == 0:
                    distances = np.ones((0, 0))
                else:
                    gt_boxes = np.array([b.translation[:2] for b in frame_gt])
                    pred_boxes = np.array([b.translation[:2] for b in frame_pred])
                    distances = sklearn.metrics.pairwise.euclidean_distances(gt_boxes, pred_boxes)

                # Distances that are larger than the threshold won't be associated.
                assert len(distances) == 0 or not np.all(np.isnan(distances))
                distances[distances >= self.dist_th_tp] = np.nan

                # Accumulate results.
                # Note that we cannot use timestamp as frameid as motmetrics assumes it's an integer.
                acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

                # Store scores of matches, which are used to determine recall thresholds.
                events = acc.events.loc[frame_id]
                matches = events[events.Type == 'MATCH']
                match_pred_ids = matches.HId.values
                match_gt_ids = matches.OId.values



                # Render the boxes in this frame.
                if frame_id > 1:
                    switches = events[events.Type == 'SWITCH']
                    switch_ids = switches.HId.values
                    gt_switch_ids = switches.OId.values
                    for i, switch_id in enumerate(switch_ids):
                        # get GT information of past 3 frames
                        cur_gt = [box for box in frame_gt if box.tracking_id == gt_switch_ids[i]]
                        prev_gt = [box for box in prev_frame_gt if box.tracking_id == gt_switch_ids[i]]
                        prev_prev_gt = [box for box in prev_prev_frame_gt if box.tracking_id == gt_switch_ids[i]]
                        gts = cur_gt + prev_gt + prev_prev_gt

                        # get most-recent previous objects assignment to this GT
                        first_assigned_preds = []
                        for lookback in [1, 2]:
                            frame = prev_frame_pred if lookback == 1 else prev_prev_frame_pred
                            lookback_events = acc.events.loc[frame_id - lookback]
                            gtids = np.concatenate([lookback_events[lookback_events.Type == 'MATCH'].OId.values, lookback_events[lookback_events.Type == 'SWITCH'].OId.values])
                            hids = np.concatenate([lookback_events[lookback_events.Type == 'MATCH'].HId.values, lookback_events[lookback_events.Type == 'SWITCH'].HId.values])
                            gtidx = np.where(gtids == gt_switch_ids[i])[0]
                            hids = hids[gtidx]
                            first_assigned_preds = [box for box in frame if box.tracking_id in hids]
                            if len(first_assigned_preds) > 0:
                                break

                        # use see where the previously-assigned object has been
                        if len(first_assigned_preds) > 0:
                            prev_id = first_assigned_preds[0].tracking_id
                            cur_origobj = [box for box in frame_pred if box.tracking_id == prev_id]
                            cur_origobj = cur_origobj if len(cur_origobj) > 0 else [None]
                            prev_origobj = [box for box in prev_frame_pred if box.tracking_id == prev_id]
                            prev_origobj = prev_origobj if len(cur_origobj) > 0 else [None]
                            prev_prev_origobj = [box for box in prev_prev_frame_pred if box.tracking_id == prev_id]
                            prev_prev_origobj = prev_prev_origobj if len(cur_origobj) > 0 else [None]
                            first_assigned_preds = cur_origobj + prev_origobj + prev_prev_origobj

                        # get history of our currently associated object
                        cur_pred = [box for box in frame_pred if box.tracking_id == switch_id]
                        cur_pred = cur_pred if len(cur_pred) > 0 else [None]
                        prev_pred = [box for box in prev_frame_pred if box.tracking_id == switch_id]
                        prev_pred = prev_pred if len(prev_pred) > 0 else [None]
                        prev_prev_pred = [box for box in prev_prev_frame_pred if box.tracking_id == switch_id]
                        prev_prev_pred = prev_prev_pred if len(prev_prev_pred) > 0 else [None]
                        preds = cur_pred + prev_pred + prev_prev_pred

                        # render
                        ex_timestamp = str(timestamp) + "_%s" % i
                        renderer.render(events, ex_timestamp, frame_gt, prev_frame_gt, prev_prev_frame_gt, frame_pred, prev_frame_pred, prev_prev_frame_pred, gts, preds, first_assigned_preds)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1
                prev_prev_frame_gt = prev_frame_gt
                prev_prev_frame_pred = prev_frame_pred
                prev_frame_gt = frame_gt
                prev_frame_pred = frame_pred

            if not has_preds:
                shutil.rmtree(os.path.join(self.output_dir, 'render', str(scene_id)))

            accs.append(acc)

            if scene_iter >= num_scenes:
                break
            if has_preds:
                scene_iter += 1

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        return acc_merged, scores






class TrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.
        self.n_clrs = 30
        self.gt_cur_clrs = seaborn.color_palette(palette='muted', n_colors=self.n_clrs)
        self.gt_prev_clrs = seaborn.color_palette(palette='pastel', n_colors=self.n_clrs)
        # self.matched_clrs = seaborn.color_palette(palette='bright', n_colors=self.n_clrs)
        # self.gt_background_palette = seaborn.color_palette(palette='spring', n_colors=self.n_clrs)
        self.gt_background_palette = [self.gt_cur_clrs, self.gt_prev_clrs]

        # self.pred_background_palette = seaborn.color_palette(palette='winter', n_colors=self.n_clrs)
        # self.gt_background_clr = 'grey'
        self.pred_background_clr = [(1.0, 24/25, 23/25, 1.0), (241/255, 241/255, 1.0, 1.0)]
        self.gt_clrs = [(0, 0.5, 0, 1.0), (51/225, 205/255, 51/255, 1.0), (144/255, 238/255, 144/255, 1.0)]
        self.pred_clrs = [(1.0, 0, 0, 1.0), (1.0, 128/255, 114/255, 1.0), (1.0, 160/255, 123/255, 1.0)]
        self.other_clrs = [(0, 11/255, 1.0, 1.0), (65/255, 105/255, 1.0, 1.0), (100/255, 149/255, 237/255, 1.0)]

    def render(self, events, timestamp, gt_cur, gt_prev, gt_prev_prev, preds_cur, preds_prev, preds_prev_prev, gts, preds, first_assigned_preds):
        """
        Render function for a given scene timestamp
        :param events: motmetrics events for that particular
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        print('Rendering {}'.format(timestamp))
        fig, ax = plt.subplots()

        # get random ego-shift
        offset = preds[0].translation

        # Plot all pred boxes
        for i, pred_boxes in enumerate([preds_cur, preds_prev]):
            for b in pred_boxes:
                translation = (b.translation[0] - offset[0], b.translation[1] - offset[1], b.translation[2] - offset[2])
                box = Box(translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
                clr = self.pred_background_clr[i]
                # clr = (clr[0], clr[1], clr[2], 1.0)
                box.render(ax, view=np.eye(4), colors=(clr, clr, clr), linewidth=1.25)

        # Plot all gt boxes.
        for i, gt_boxes in enumerate([gt_cur, gt_prev]):
            for b in gt_boxes:
                translation = (b.translation[0] - offset[0], b.translation[1] - offset[1], b.translation[2] - offset[2])
                box = Box(translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
                clr = self.gt_background_palette[i][hash(b.tracking_id) % self.n_clrs]
                clr = (clr[0], clr[1], clr[2], 1.0)
                box.render(ax, view=np.eye(4), colors=(clr, clr, clr), linewidth=0.45 + 0.1*i)

        # Plot GT track boxes.
        for i, b in enumerate(reversed(gts)):
            if b is None:
                continue
            # color = self.cur_clrs[hash(b.tracking_id) % self.n_clrs]
            clr = self.gt_clrs[(-i-1) % 3]
            translation = (b.translation[0] - offset[0], b.translation[1] - offset[1], b.translation[2] - offset[2])
            box = Box(translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            box.render(ax, view=np.eye(4), colors=(clr, clr, clr), linewidth=1.7)

        # # Plot prediction track boxes.
        for i, b in enumerate(reversed(preds)):
            if b is None:
                continue
            # color = self.cur_clrs[hash(b.tracking_id) % self.n_clrs]
            clr = self.pred_clrs[(-i-1) % 3]
            translation = (b.translation[0] - offset[0], b.translation[1] - offset[1], b.translation[2] - offset[2])
            box = Box(translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            box.render(ax, view=np.eye(4), colors=(clr, clr, clr), linewidth=1.2 - 0.1*i)
            ax.arrow(translation[0], translation[1], b.velocity[0], b.velocity[1])

        # Plot prediction track boxes.
        for i, b in enumerate(reversed(first_assigned_preds)):
            if b is None:
                continue
            # color = self.cur_clrs[hash(b.tracking_id) % self.n_clrs]
            clr = self.other_clrs[(-i-1) % 3]
            translation = (b.translation[0] - offset[0], b.translation[1] - offset[1], b.translation[2] - offset[2])
            box = Box(translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            box.render(ax, view=np.eye(4), colors=(clr, clr, clr), linewidth=1.0 - 0.1*i)
            ax.arrow(translation[0], translation[1], b.velocity[0], b.velocity[1])


        # # Plot predicted boxes.
        # for b in frame_pred:
        #     box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
        #
        #     # Determine color for this tracking id.
        #     if b.tracking_id not in self.id2color.keys():
        #         self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
        #                                         float(hash(b.tracking_id + 'g') % 256) / 255,
        #                                         float(hash(b.tracking_id + 'b') % 256) / 255)
        #
        #     # Render box. Highlight identity switches in red.
        #     if b.tracking_id in switch_ids:
        #         color = self.id2color[b.tracking_id]
        #         box.render(ax, view=np.eye(4), colors=('r', 'r', color))
        #     else:
        #         color = self.id2color[b.tracking_id]
        #         box.render(ax, view=np.eye(4), colors=(color, color, color))

        # Plot ego pose.
        # plt.scatter(0, 0, s=48, facecolors='none', edgecolors='none', marker='o')
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)), format="png", dpi=700)
        plt.close(fig)
