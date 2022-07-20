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
from typing import List, Dict, Callable
import unittest

import numpy as np
import sklearn
import tqdm

try:
    import pandas
except ModuleNotFoundError:
    raise unittest.SkipTest('Skipping test as pandas was not found!')

from nuscenes.eval.tracking.constants import MOT_METRIC_MAP, TRACKING_METRICS
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricData
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
# from nuscenes.eval.tracking.render import TrackingRenderer
from nuscenes.eval.tracking.utils import create_motmetrics
from scripts.vis_nusc_tracking.our_nusc_renderer import TrackingRenderer


class TrackingEvaluation(object):
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
                 output_dir: str = None,
                 render_classes: List[str] = None):
        """
        Create a TrackingEvaluation object which computes all metrics for a given class.
        :param tracks_gt: The ground-truth tracks.
        :param tracks_pred: The predicted tracks.
        :param class_name: The current class we are evaluating on.
        :param dist_fcn: The distance function used for evaluation.
        :param dist_th_tp: The distance threshold used to determine matches.
        :param min_recall: The minimum recall value below which we drop thresholds due to too much noise.
        :param num_thresholds: The number of recall thresholds from 0 to 1. Note that some of these may be dropped.
        :param metric_worst: Mapping from metric name to the fallback value assigned if a recall threshold
            is not achieved.
        :param verbose: Whether to print to stdout.
        :param output_dir: Output directory to save renders.
        :param render_classes: Classes to render to disk or None.

        Computes the metrics defined in:
        - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.
          MOTA, MOTP
        - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows.
          MT/PT/ML
        - Weng 2019: "A Baseline for 3D Multi-Object Tracking".
          AMOTA/AMOTP
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
        self.render_classes = [] if render_classes is None else render_classes

        self.n_scenes = len(self.tracks_gt)

        # Specify threshold naming pattern. Note that no two thresholds may have the same name.
        def name_gen(_threshold):
            return 'thr_%.4f' % _threshold
        self.name_gen = name_gen

        # Check that metric definitions are consistent.
        for metric_name in MOT_METRIC_MAP.values():
            assert metric_name == '' or metric_name in TRACKING_METRICS

    def accumulate(self, render) -> TrackingMetricData:
        """
        Compute metrics for all recall thresholds of the current class.
        :return: TrackingMetricData instance which holds the metrics for each threshold.
        """
        # Init.
        if self.verbose:
            print('Computing metrics for class %s...\n' % self.class_name)
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
        mh = create_motmetrics()

        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it without trying all thresholds.
        acc, scores, scene_motas = self.accumulate_threshold(threshold=0.377, render=render)  # todo: update me accordingly!
        # recall = 91.1
        # ours = 0.33769
        # nms = 0.27104944984118146
        # ours-only-boxes = 0.2805

        # ours:
        # Compute metrics for current threshold.
        # nms = 0.347
        summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name="all-scenes")

        return summary, scene_motas

    def accumulate_threshold(self, threshold: float = None, render=True):
        """
        Accumulate metrics for a particular recall threshold of the current class.
        The scores are only computed if threshold is set to None. This is used to infer the recall thresholds.
        :param threshold: score threshold used to determine positives and negatives.
        :return: (The MOTAccumulator that stores all the hits/misses/etc, Scores for each TP).
        """
        accs = []
        scores = []  # The scores of the TPs. These are used to determine the recall thresholds initially.
        scene_metrics = {}

        # Go through all frames and associate ground truth and tracker results.
        # Groundtruth and tracker contain lists for every single frame containing lists detections.
        # scene_ids = ['a2b005c4dd654af48194ada18662c8ca', '5eaff323af5b4df4b864dbd433aeb75c', '6a24a80e2ea3493c81f5fdb9fe78c28a']
        # scene_ids = ['6a24a80e2ea3493c81f5fdb9fe78c28a', '64bfc5edd71147858ce7446892d7f864', '7061c08f7eec4495979a0cf68ab6bb79', '3ada261efee347cba2e7557794f1aec8']
 #        scene_ids = ['a2b005c4dd654af48194ada18662c8ca', '6a24a80e2ea3493c81f5fdb9fe78c28a', '64bfc5edd71147858ce7446892d7f864',
 # '7061c08f7eec4495979a0cf68ab6bb79', '3ada261efee347cba2e7557794f1aec8',
 # '07aed9dae37340a997535ad99138e243', 'ed242d80ccb34b139aaf9ab89859332e',
 # 'ec7b7459461e4da1a236ba23e22377c9', '21a7ba093614493b83838b9656b3558d',
 # '265f002f02d447ad9074813292eef75e', '3dd2be428534403ba150a0b60abc6a0a',
 # '325cef682f064c55a255f2625c533b75', 'c3ab8ee2c1a54068a72d7eb4cf22e43d',
 # 'cb3e964697d448b3bc04a9bc06c9e131', 'f5b29a1e09d04355adcd60ab72de006b',
 # 'ca6e45c25d954dc4af6e12dd7b23454d']
        # scene_ids = ['3ada261efee347cba2e7557794f1aec8']
        scene_ids = []
        for scene_id in tqdm.tqdm(self.tracks_gt.keys(), disable=not self.verbose, leave=False):
            # if scene_id not in scene_ids:
            #     continue

            # Initialize accumulator and frame_id for this scene
            acc = MOTAccumulatorCustom()
            frame_id = 0  # Frame ids must be unique across all scenes

            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            # Visualize the boxes in this frame.
            if self.class_name in self.render_classes and render:
                save_path = os.path.join(self.output_dir, 'render', str(scene_id), self.class_name)
                os.makedirs(save_path, exist_ok=True)
                renderer = TrackingRenderer(save_path)
            else:
                renderer = None

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
                if len(gt_ids) == 0 and len(pred_ids) == 0:
                    continue

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
                match_ids = matches.HId.values
                match_scores = [tt.tracking_score for tt in frame_pred if tt.tracking_id in match_ids]
                scores.extend(match_scores)

                # Render the boxes in this frame.
                if self.class_name in self.render_classes and render:
                    renderer.render(timestamp, frame_gt, frame_pred)

                # Increment the frame_id, unless there are no boxes (equivalent to what motmetrics does).
                frame_id += 1

            if scene_id in scene_ids:
                t_ranges = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10, 11], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                            [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                            [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]]
                for k, t_range in enumerate(t_ranges):
                    if self.class_name in self.render_classes:
                        save_path = os.path.join(self.output_dir, 'render', self.class_name)
                        os.makedirs(save_path, exist_ok=True)
                        renderer = TrackingRenderer(save_path)
                        gt_boxes = []
                        for i, t in enumerate(scene_tracks_gt):
                            if i not in t_range:
                                continue
                            # print(self.class_name)
                            # print(scene_tracks_gt[t])
                            gt_of_class = [box for box in scene_tracks_gt[t] if box.tracking_name == self.class_name]
                            # print(gt_of_class)
                            gt_boxes.extend(gt_of_class)
                        pred_boxes = []
                        for i, t in enumerate(scene_tracks_pred):
                            if i not in t_range:
                                continue
                            preds_of_conf = [box for box in scene_tracks_pred[t] if box.tracking_score >= threshold]
                            pred_boxes.extend(preds_of_conf)
                        renderer.render("%s_%s" % (str(scene_id), k), gt_boxes, pred_boxes)

            # compute scene metrics
            mh = create_motmetrics()
            thresh_summary = mh.compute(acc, metrics=MOT_METRIC_MAP.keys(), name="all_thresh")
            mota = thresh_summary['mota_custom'].to_numpy().item()
            scene_metrics[scene_id] = mota
            # for col in thresh_summary.columns:
            #     print(col)
            #     print(thresh_summary[col])

            accs.append(acc)

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        return acc_merged, scores, scene_metrics

