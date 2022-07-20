lidarseg_name2idx_mapping = {'noise': 0,
 'animal': 1,
 'human.pedestrian.adult': 2,
 'human.pedestrian.child': 3,
 'human.pedestrian.construction_worker': 4,
 'human.pedestrian.personal_mobility': 5,
 'human.pedestrian.police_officer': 6,
 'human.pedestrian.stroller': 7,
 'human.pedestrian.wheelchair': 8,
 'movable_object.barrier': 9,
 'movable_object.debris': 10,
 'movable_object.pushable_pullable': 11,
 'movable_object.trafficcone': 12,
 'static_object.bicycle_rack': 13,
 'vehicle.bicycle': 14,
 'vehicle.bus.bendy': 15,
 'vehicle.bus.rigid': 16,
 'vehicle.car': 17,
 'vehicle.construction': 18,
 'vehicle.emergency.ambulance': 19,
 'vehicle.emergency.police': 20,
 'vehicle.motorcycle': 21,
 'vehicle.trailer': 22,
 'vehicle.truck': 23,
 'flat.driveable_surface': 24,
 'flat.other': 25,
 'flat.sidewalk': 26,
 'flat.terrain': 27,
 'static.manmade': 28,
 'static.other': 29,
 'static.vegetation': 30,
 'vehicle.ego': 31}

class_name2lidarseg_names = {"car": ["vehicle.car"],
                             "pedestrian": ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.police_officer'],
                             "bicycle": ['vehicle.bicycle'],
                             'trailer': ['vehicle.trailer'],
                             'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
                             'motorcycle': ['vehicle.motorcycle'],
                             'truck': ['vehicle.truck']}


import os

import torch

import numpy as np
from third_party.mmdet3d.ops.roiaware_pool3d.points_in_boxes import points_in_boxes_batch

from preprocessing.nuscenes.utils import from_file_multisweep
from preprocessing.common.points import ObjectPoints


class KeyframeAndPrevSweepPoints:
    CLOSE_POINTS_RADIUS = 1.0

    def __init__(self, nusc, nuscenes_path, version, sample, sample_data, num_sweeps):
        # store basic info
        self.nusc = nusc
        self.sample_data = sample_data
        self.sample = sample
        self.num_sweeps = num_sweeps

        # compute point info
        self.points, self.point_times, self.close_point_masks = self._get_points()
        self.points, self.point_times = self.points.T, self.point_times.flatten()
        self.lidarseg = self._get_lidarseg(nuscenes_path, version)

    def _get_lidarseg(self, nuscenes_path, version):
        # Get lidarseg point labels;
        data_token = self.sample_data['token']
        lidar_seg_name = data_token + "_lidarseg.bin"
        lidar_seg_path = os.path.join(nuscenes_path, "lidarseg", version, lidar_seg_name)
        if version != "v1.0-test":
            points_labels = np.fromfile(lidar_seg_path, dtype=np.uint8)
            points_labels = points_labels[self.close_point_masks[0]]  # 0'th index indicates first sweep, ie sample keyframe
        else:
            points_labels = np.zeros((self.points.shape[0]))  # 0'th index indicates first sweep, ie sample keyframe

        return points_labels

    def _get_points(self):
        # Nuscenes method aggregates into single frame; our modified method does not
        pc, times, close_point_masks = from_file_multisweep(self.nusc, self.sample, self.sample_data['channel'],
                                                            'LIDAR_TOP', nsweeps=self.num_sweeps, min_distance=self.CLOSE_POINTS_RADIUS)
        points = pc.points[:3, :]
        return points, times, close_point_masks

    def points_in_boxes(self, boxes, scale_factor, time_threshold=0.001):
        success, points, num_pts, sweep_pc, crop_pnt_idxs = self._points_in_boxes(boxes, scale_factor, time_threshold)
        if not success:
            return success, None, None, None

        withgt_mask = np.array([box.with_gt_box for box in boxes.boxes])
        withgt_idxs = np.where(withgt_mask)[0]
        all_gt_masks = np.array([None] * len(boxes))
        if np.sum(withgt_mask) > 0:
            gt_pointidxs = self._compute_object_point_masks(boxes[withgt_mask], sweep_pc, compute_gt=True)

            # combine with lidarseg
            gt_pointidxs = self.apply_gt_lidarseg(gt_pointidxs, boxes[withgt_mask])

            # filter to only points in our object crop
            gt_masks = [gt_pointidxs[i][crop_pnt_idxs[withgt_idxs[i]]] for i in range(len(gt_pointidxs))]
            for i in range(len(gt_masks)):
                all_gt_masks[withgt_idxs[i]] = gt_masks[i].reshape(-1, 1)

        # format to be empty arrays instead of None
        all_gt_masks = all_gt_masks.tolist()
        all_gt_masks = [all_gt_masks[i] if all_gt_masks[i] is not None else np.array([]) for i in range(len(all_gt_masks))]

        out = ObjectPoints(points, num_pts, all_gt_masks, scale_factor)

        return success, out

    def _points_in_boxes(self, objs, scale_factor, time_threshold=0.001):
        # get points belonging to this timestamp
        keyframe_time = 1e-6 * self.sample_data['timestamp']
        objs_time_lag = keyframe_time - objs.timestep
        sweep_idxs = np.abs(self.point_times - objs_time_lag) < time_threshold
        sweep_pc = self.points[sweep_idxs, :]

        # check sweep pc has >0 points
        if sweep_pc.shape[0] == 0:
            print("No PC in Sweep Time Lag %s!" % objs_time_lag)
            return False, None, None, None, None
        if  len(objs.boxes) == 0:
            return False, None, None, None, None

        # get point assignments
        enlarged_objs = objs.expand_sizes(scale_factor)
        pointidxs = self._compute_object_point_masks(objs, sweep_pc)
        enlarged_pointidxs = self._compute_object_point_masks(enlarged_objs, sweep_pc)

        # extract necessary point info
        num_pts = [np.sum(pointidxs[i]).item() for i in range(len(pointidxs))]
        points = [sweep_pc[large_pointidx] for large_pointidx in enlarged_pointidxs]

        return True, points, num_pts, sweep_pc, enlarged_pointidxs

    def _compute_object_point_masks(self, boxes, points, compute_gt=False):
        boxes_mmdet = boxes.convert2mmdetboxes(gt=compute_gt)
        boxes_mmdet = torch.from_numpy(boxes_mmdet).float().cuda().view(len(boxes), 1, -1)[:, :, :7]
        points_mmdet = torch.from_numpy(points[:, :3]).unsqueeze(dim=0).float().cuda().repeat(len(boxes), 1, 1).contiguous()
        assignment_idxs = points_in_boxes_batch(points_mmdet, boxes_mmdet)  # returns B x N-pts x M-

        obj_masks = []
        for i in range(len(boxes)):
            obj_point_mask = assignment_idxs[i, :, 0] == 1
            obj_masks.append(obj_point_mask.to('cpu').data.numpy())
        return obj_masks

    def apply_gt_lidarseg(self, gt_masks, boxes):
        updated_gt_masks = []
        for i in range(len(boxes)):
            class_name = boxes.boxes[i].class_name
            lidarseg_mask = np.zeros(self.lidarseg.shape, dtype=np.bool)
            for lidarseg_name in class_name2lidarseg_names[class_name]:
                lidarseg_idx = lidarseg_name2idx_mapping[lidarseg_name]
                lidarseg_mask |= (self.lidarseg == lidarseg_idx)
            updated_gt_mask = gt_masks[i] & lidarseg_mask
            updated_gt_masks.append(updated_gt_mask)
        return updated_gt_masks