import pytorch3d.loss
import torch
import numpy as np
from third_party.mmdet3d.ops.iou3d.iou3d_utils import boxes_iou_bev
from spot.ops.transform_utils import lidar2bev_box

torch.set_printoptions(precision=14)

DEBUG = False


class AssociationL1Distance:

    def __init__(self, association_params, trajectory_params, object_class):
        self.distance_type = association_params['box-affinity']['distance-metric']
        self.center_comparison = association_params['matching']['compare-which-frames']
        if self.distance_type == "iou":
            self.distance_threshold = -association_params['box-affinity']['iou-threshold']
        elif self.distance_type == "l2":
            self.distance_threshold = association_params['box-affinity']['l2-distance-thresholds'][object_class]
        self.trajectory_params = trajectory_params

    def compute_l1_matrix(self, tracks, detections):
        # get timesteps for comparison
        if self.center_comparison == "earliest-detection":
            comparison_timesteps = detections.get_object_timesteps("min")
        elif self.center_comparison == "all-detections":
            comparison_timesteps = detections.get_object_timesteps("all")
        elif self.center_comparison == "centerpoint-lookback":
            comparison_timesteps = tracks.get_object_timesteps("max")
            comparison_timesteps = torch.max(comparison_timesteps).unsqueeze(0).repeat(len(tracks))

        else:
            raise RuntimeError("Center comparison must be one of ['all-detections', 'earliest-detection']")
        u_comparison_timesteps, comparison_timestep_map = torch.unique(comparison_timesteps, return_inverse=True)
        T, M, N = u_comparison_timesteps.size(0), len(tracks), len(detections)

        # forecast tracks and detections to equivalent timesteps
        track_boxes, _ = tracks.estimate_boxes(u_comparison_timesteps, **self.trajectory_params)
        track_boxes = torch.stack([track_box.box for track_box in track_boxes])
        detection_boxes, _ = detections.estimate_boxes(u_comparison_timesteps, **self.trajectory_params)
        detection_boxes = torch.stack([det_box.box for det_box in detection_boxes])

        # compute distance matrix
        if self.distance_type == "l2":
            distance_matrix = torch.cdist(detection_boxes[:, :, :2].double(), track_boxes[:, :, :2].double(), p=2)  # t-timesteps x n-dets x m-tracks
        else:
            bev_offset = detection_boxes[0, 0:1, :2]  # for numerical stability
            track_boxes_bev = lidar2bev_box(track_boxes[:, :, :7].view(-1, 7), bev_offset)  # T*M x 5
            det_boxes_bev = lidar2bev_box(detection_boxes[:, :, :7].view(-1, 7), bev_offset)  # T*N x 5
            ious = boxes_iou_bev(det_boxes_bev, track_boxes_bev)  # (T*N, T*M)  todo: this is Tx slow because we're repeating across all timesteps!
            ious = ious.view(T, N, T, M)[torch.arange(T), :, torch.arange(T), :]  # T x N x M
            distance_matrix = -ious

        # filter to a single distance value per track-detection pair
        if self.center_comparison == "earliest-detection":
            distance_matrix = distance_matrix[comparison_timestep_map, torch.arange(len(detections))]  # n-dets x m-tracks
        elif self.center_comparison == "centerpoint-lookback":
            distance_matrix = distance_matrix[comparison_timestep_map, :, torch.arange(len(tracks))].T  # n-dets x m-tracks
        else:  # self.center_comparison == "all-detections":
            distance_matrix = torch.mean(distance_matrix, dim=0)  # n-dets x m-tracks
        distance_matrix[distance_matrix > self.distance_threshold] = np.inf

        return distance_matrix


class ConfidenceRescorer:

    def __init__(self, sort_by):
        self.sort_by = sort_by

    def rescore(self, tracks, detections, distance_matrix):
        if self.sort_by == "detections":
            detection_confidences = detections.get_object_confidences(which="mean")
            sorted_idxs = torch.argsort(detection_confidences, descending=True)
            distance_matrix = distance_matrix[sorted_idxs, :]
        elif self.sort_by == "tracks":
            track_confidences = tracks.get_object_confidences(which="mean")
            sorted_idxs = torch.argsort(track_confidences, descending=True)
            distance_matrix = distance_matrix[:, sorted_idxs]
        else:
            raise RuntimeError("Sort-By must be one of ['tracks', 'detections'].")

        return distance_matrix, sorted_idxs