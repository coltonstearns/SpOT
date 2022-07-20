import os
import time
from spot.ops.data_utils import quaternion_yaw
from spot.io.wandb_utils import get_wandb_boxes, get_boxdir_points

from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionBox
from typing import Callable
import wandb

from nuscenes.eval.common.utils import center_distance
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionMetricDataList
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

Axis = Any




class VisualizeDetectionError:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 baseline_result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.baseline_boxes, _ = load_prediction(baseline_result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        assert set(self.baseline_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
        self.baseline_boxes = add_center_dist(nusc, self.baseline_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        self.baseline_boxes = filter_eval_boxes(nusc, self.baseline_boxes, self.cfg.class_range, verbose=verbose)

        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        all_pnts = []
        for box in self.gt_boxes.all:
            if box.detection_name == "car":
                all_pnts.append(box.num_pts)
        total_pnts = len(all_pnts)
        all_pnts = np.array(all_pnts)
        pnt_pcts = []
        for i in range(200):
           pct = np.sum(all_pnts >= i) / total_pnts
           pnt_pcts.append(pct)

        plt.plot(np.arange(200), pnt_pcts)
        plt.savefig(os.path.join(output_dir, "point-dist.png"))
        plt.clf()


        self.sample_tokens = self.gt_boxes.sample_tokens

    def run(self, class_name):
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()
        for dist_th in self.cfg.dist_ths:
            md = self.visualize(class_name, self.cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    def visualize(self,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMetricData:
        """
        Average Precision over predefined different recall thresholds for a single distance threshold.
        The recall/conf thresholds and other raw metrics will be used in secondary metrics.
        :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
        :param pred_boxes: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param dist_fcn: Distance function used to match detections and ground truths.
        :param dist_th: Distance threshold for a match.
        :param verbose: If true, print debug messages.
        :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
        """
        # ---------------------------------------------
        # Organize input and initialize accumulators.
        # ---------------------------------------------

        # Count the positives.
        npos = len([1 for gt_box in self.gt_boxes.all if gt_box.detection_name == class_name])
        if self.verbose:
            print("Found {} GT of class {} out of {} total across {} samples.".
                  format(npos, class_name, len(self.gt_boxes.all), len(self.gt_boxes.sample_tokens)))

        # For missing classes in the GT, return a data structure corresponding to no predictions.
        if npos == 0:
            return DetectionMetricData.no_predictions()

        # Organize the predictions in a single list.
        pred_boxes_list = [box for box in self.pred_boxes.all if box.detection_name == class_name]
        baseline_boxes_list = [box for box in self.baseline_boxes.all if box.detection_name == class_name]
        pred_confs = [box.detection_score for box in pred_boxes_list]
        baseline_confs = [box.detection_score for box in baseline_boxes_list]

        if verbose:
            print("Found {} PRED of class {} out of {} total across {} samples.".
                  format(len(pred_confs), class_name, len(self.pred_boxes.all), len(self.pred_boxes.sample_tokens)))

        # Sort by confidence.
        prediction_gt2errs, gt2predbox, gt2gtbox1 = perform_gt_matching(pred_boxes_list, pred_confs, self.gt_boxes, class_name, dist_fcn, dist_th)
        baseline_gt2errs, gt2baselinebox, gt2gtbox2 = perform_gt_matching(baseline_boxes_list, baseline_confs, self.gt_boxes, class_name, dist_fcn, dist_th)
        gt2gtbox = {**gt2gtbox1, **gt2gtbox2}
        num_our_misses = 0
        num_our_gains = 0
        our_misses = []
        our_gains = []
        err_sorting = []
        for gt_id in set(list(baseline_gt2errs.keys()) + list(prediction_gt2errs.keys())):
            if gt_id not in baseline_gt2errs:
                num_our_gains += 1
                our_gains.append(gt_id)
                continue
            elif gt_id not in prediction_gt2errs:
                num_our_misses += 1
                our_misses.append(gt_id)
                continue
            else:
                pred_err = prediction_gt2errs[gt_id]
                baseline_err = prediction_gt2errs[gt_id]
                diff = pred_err - baseline_err
                err_sorting.append((diff, gt_id))
        err_sorting.sort(reverse=True)

        print("===================")
        print("==================")
        print("FP To TP Summary:")
        print(num_our_gains)
        print("TP to FP Summary:")
        print(num_our_misses)

        # todo: plot distance from car?

        error_dir = os.path.join(self.output_dir, "Error_Analysis", str(dist_th))
        if not os.path.exists(error_dir):
            os.makedirs(error_dir)

        compute_dist_histograms(self.nusc,
                                our_misses,
                                our_gains,
                                gt2gtbox,
                                savepath=os.path.join(error_dir, 'error_histogram_%s.png' % str(dist_th)))

        # for i, gt_id in enumerate(our_misses):
        #     sample_token = gt_id[0]
        #     predicted_boxes = self.pred_boxes[sample_token]
        #     gt_box = gt2gtbox[gt_id]
        #     baseline_box = gt2baselinebox[gt_id]
        #     visualize_single_sample(self.nusc,
        #                             sample_token,
        #                             gt_box,
        #                             predicted_boxes,
        #                             baseline_box,
        #                             savepath=os.path.join(error_dir, '{}.png'.format(gt_id)))
            # visualize_miss(self.nusc,
            #                sample_token,
            #                gt_box,
            #                predicted_boxes,
            #                baseline_box,
            #                savepath=os.path.join(error_dir, '{}.png'.format(gt_id))
            #                )
            # if i > 25:
            #     break

        # for i in range(40):
            # gt_id = err_sorting[i][1]
            # sample_token = gt_id[0]
            # pred_box = gt2predbox[gt_id]
            # gt_box = gt2gtbox[gt_id]
            # baseline_box = gt2baselinebox[gt_id]
            # visualize_single_sample(self.nusc,
            #                         sample_token,
            #                         gt_box,
            #                         pred_box,
            #                         baseline_box,
            #                         savepath=os.path.join(error_dir, '{}.png'.format(gt_id)))





def perform_gt_matching(pred_boxes_list, pred_confs, gt_boxes, class_name, dist_fcn, dist_th):
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    gt2pred_center_errs = {}
    gt2pred_boxes = {}
    gtid2gtbox = {}

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Compute center error for this match
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            gtid2gtbox[(pred_box.sample_token, match_gt_idx)] = gt_box_match
            gt2pred_boxes[(pred_box.sample_token, match_gt_idx)] = pred_box
            gt2pred_center_errs[(pred_box.sample_token, match_gt_idx)] = center_distance(gt_box_match, pred_box)

    return gt2pred_center_errs, gt2pred_boxes, gtid2gtbox


def compute_dist_histograms(nusc: NuScenes,
                            our_misses,
                            our_gains,
                            gt2gtbox,
                            savepath):
    miss_numpts = []
    for miss_id in our_misses:
        sample_token = miss_id[0]
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Get boxes in ego
        gt_box = gt2gtbox[miss_id]
        num_pts = gt_box.num_pts

        # Map GT boxes to lidar.
        # gt_box = boxes_to_sensor([gt_box], pose_record, cs_record)[0]
        # ego_dist = (gt_box.center[0]**2 + gt_box.center[1]**2)**(1/2)
        miss_numpts.append(num_pts)

    print("Avg points on a missed-detection:")
    print(sum(miss_numpts) / len(miss_numpts))
    print(np.median(miss_numpts))

    gain_numpts = []
    for gain_id in our_gains:
        sample_token = gain_id[0]
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Get boxes in ego
        gt_box = gt2gtbox[gain_id]
        num_pts = gt_box.num_pts

        # Map GT boxes to lidar.
        # gt_box = boxes_to_sensor([gt_box], pose_record, cs_record)[0]
        # ego_dist = (gt_box.center[0]**2 + gt_box.center[1]**2)**(1/2)
        gain_numpts.append(num_pts)

    print("Avg points on a missed-detection:")
    print(sum(gain_numpts) / len(gain_numpts))
    print(np.median(gain_numpts))

    bins = np.linspace(0, 150, 35)

    plt.hist(gain_numpts, bins, alpha=0.5, label='gains')
    plt.hist(miss_numpts, bins, alpha=0.5, label='losses')
    plt.legend(loc='upper right')
    plt.savefig(savepath)
    plt.clf()

    # make curve of accuracy vs num-pts
    total_err = len(miss_numpts) # - len(gain_numpts)
    gain_numpts.sort()
    miss_numpts.sort()
    gain_numpts = np.array(gain_numpts)
    miss_numpts = np.array(miss_numpts)
    num_unique_pts = np.unique(miss_numpts)

    pnt_threshs = []
    diffs = []
    for n in num_unique_pts:
        gains = np.sum(gain_numpts >= n)
        misses = np.sum(miss_numpts >= n)
        diff = misses
        pnt_threshs.append(n)
        diffs.append(diff / total_err)
    closest_to_eighty = np.argmin(np.abs(np.array(diffs) - 0.2))
    print("At 80% error threshold, we have:")
    print(diffs[closest_to_eighty])
    print(pnt_threshs[closest_to_eighty])
    plt.plot(pnt_threshs, diffs)
    plt.xlim([-1, 200])
    plt.axhline(y=0.0, color='r', linestyle='-')

    plt.savefig(savepath + "2.png")



def visualize_single_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_box,
                     predicted_boxes,
                     baseline_box,
                     nsweeps: int = 10,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Map GT boxes to lidar.
    gt_box = boxes_to_sensor([gt_box], pose_record, cs_record)[0]
    predicted_boxes = boxes_to_sensor(predicted_boxes, pose_record, cs_record)
    baseline_box = boxes_to_sensor([baseline_box], pose_record, cs_record)[0]

    # Get point cloud in lidar frame.
    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
    points = pc.points[:3, :].T
    point_clrs = np.ones((points.shape[0], 3)) * 155

    gt_box = nuscbox2nparray(gt_box)
    predicted_boxes = np.vstack([nuscbox2nparray(pred_box) for pred_box in predicted_boxes])
    baseline_box = nuscbox2nparray(baseline_box)
    boxes = np.vstack([gt_box, baseline_box, predicted_boxes])
    colors = np.array([[0, 255, 0], [255, 0, 0]])
    pred_colors = np.array([[0, 0, 255]]).repeat(predicted_boxes.shape[0], axis=0)
    colors = np.vstack([colors, pred_colors])
    wandb_boxes = get_wandb_boxes(boxes, colors=colors)
    wandb_box_dirpnts, wandb_box_clrs = get_boxdir_points(boxes, colors)

    # final wandb vis
    wandb_pc = np.hstack([np.vstack([points, wandb_box_dirpnts]), np.vstack([point_clrs, wandb_box_clrs])])
    wandb.log({"Error Analysis": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})

    # Init axes.
    # _, ax = plt.subplots(1, 1, figsize=(9, 9))
    #
    # # Show point cloud.
    # points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / 50)
    # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
    #
    # # Show ego vehicle.
    # ax.plot(0, 0, 'x', color='black')
    #
    # # Show GT boxes.
    # gt_box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)
    # pred_box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)
    # baseline_box.render(ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=1)
    #
    # # Limit visible range.
    # axes_limit = 50 + 3  # Slightly bigger to include boxes that extend beyond the range.
    # ax.set_xlim(-axes_limit, axes_limit)
    # ax.set_ylim(-axes_limit, axes_limit)
    #
    # # Show / save plot.
    # if verbose:
    #     print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    # if savepath is not None:
    #     plt.savefig(savepath)
    #     plt.close()
    # else:
    #     plt.show()

def nuscbox2nparray(nusc_box):
    yaw = quaternion_yaw(nusc_box.orientation)
    yaw = np.array([yaw])
    center = np.array(nusc_box.center).flatten()
    size = np.array(nusc_box.wlh).flatten()
    size[[0, 1]] = size[[1, 0]]
    np_box = np.concatenate([center, size, yaw])
    return np_box.reshape((1, -1))

def visualize_miss(nusc: NuScenes,
                     sample_token: str,
                     gt_box,
                     pred_boxes,
                     baseline_box,
                     nsweeps: int = 40,
                     verbose: bool = True,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Map GT boxes to lidar.
    gt_box = boxes_to_sensor([gt_box], pose_record, cs_record)[0]
    pred_boxes = boxes_to_sensor(pred_boxes, pose_record, cs_record)
    baseline_box = boxes_to_sensor([baseline_box], pose_record, cs_record)[0]

    # Get point cloud in lidar frame.
    pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    colors = np.minimum(1, dists / 50)
    ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    gt_box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=2)
    baseline_box.render(ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=1)
    for box in pred_boxes:
        box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

    # Limit visible range.
    axes_limit = 50 + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(-axes_limit, axes_limit)
    ax.set_ylim(-axes_limit, axes_limit)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
