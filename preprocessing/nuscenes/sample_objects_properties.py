from preprocessing.nuscenes.utils import bboxes_glob2sensor
import numpy as np
from preprocessing.nuscenes.utils import quaternion_yaw
import copy

TRACKING_OBJECTS = {'vehicle.bicycle': 'bicycle',
                    'vehicle.bus.bendy': 'bus',
                    'vehicle.bus.rigid': 'bus',
                    'vehicle.car': 'car',
                    'vehicle.motorcycle': 'motorcycle',
                    'human.pedestrian.adult': 'pedestrian',
                    'human.pedestrian.child': 'pedestrian',
                    'human.pedestrian.construction_worker': 'pedestrian',
                    'human.pedestrian.police_officer': 'pedestrian',
                    'vehicle.trailer': 'trailer',
                    'vehicle.truck': 'truck'}

TRACKING_OBJECTS_DETECTION_NAMES = set(TRACKING_OBJECTS.values())


class KeyframeAndSweepBoxes:

    def __init__(self, keyframe_boxes, nusc, sample, sample_data, num_sweeps, xy_range=np.inf):
        self.nusc = nusc
        self.sample = sample
        self.sample_data = sample_data
        self.num_sweeps = num_sweeps
        self.xy_range = xy_range

        # initialize boxes based on this child class
        self.all_boxes = self._load_objects(keyframe_boxes)

    def _load_objects(self, keyframe_boxes):
        """
        Extract and format all trackable objects in this frame.
        """
        # format ground truth boxes
        keyframe_time = 1e-6 * self.sample_data['timestamp']
        temp_kf_boxes = OneSweepBoxes(self.nusc, keyframe_boxes, self.sample_data, keyframe_time, self.xy_range)
        temp_kf_boxes.format()

        # backtrack velocity to get previous-sweep predictions
        all_boxes = self._velocity_backtrack_keyframe(keyframe_boxes, temp_kf_boxes, keyframe_time)
        return all_boxes

    def _velocity_backtrack_keyframe(self, raw_boxes, processed_boxes, keyframe_time):
        cur_sample_data = self.sample_data
        sweep_boxes = []
        for i in range(self.num_sweeps):
            if cur_sample_data['sample_token'] != self.sample['token']:
                break
            time_delta = keyframe_time - 1e-6 * cur_sample_data['timestamp']
            backtracked_boxes = processed_boxes.velocity_backtrack(raw_boxes, time_delta, cur_sample_data, remove_gt=i > 0)
            sweep_boxes.append(backtracked_boxes)
            if cur_sample_data['prev'] == '':
                break
            cur_sample_data = self.nusc.get('sample_data', cur_sample_data['prev'])
        return sweep_boxes

    def num_valid_sweeps(self):
        return len(self.all_boxes)

    def get_sweep_boxes(self, sweep_idx):
        if sweep_idx < len(self.all_boxes):
            return self.all_boxes[sweep_idx]
        else:
            raise RuntimeError("Attempting to access objects from a sweep we don't have.")


class OneSweepBoxes:

    def __init__(self, nusc, boxes, sample_data, timestep, xy_range=np.inf, global2sensor=None):
        self.nusc = nusc
        self.timestep = timestep
        self.sample_data = sample_data
        self.xy_range = xy_range
        self.is_keyframe = False  # note: we handle keyframes later in the pipeline, so this is deprecated

        self.boxes = boxes
        self.global2sensor = global2sensor

    def format(self):
        # compute desired bounding box information
        self.boxes, self.global2sensor = self._format_boxes(self.boxes)

    def _format_boxes(self, boxes):
        boxes = self._filter_by_trackable_type(boxes)
        boxes, global2sensor = self._transform_to_sensor_refframe(boxes)
        boxes = self._filter_by_distance_from_origin(boxes)
        return boxes, global2sensor

    def _filter_by_trackable_type(self, boxes):
        boxes = [box for box in boxes if box.class_name in TRACKING_OBJECTS_DETECTION_NAMES]
        return boxes

    def _filter_by_distance_from_origin(self, boxes):
        if len(boxes) == 0:
            return boxes

        xy_locs = np.concatenate([b.center[:2, :] for b in boxes], axis=1)
        max_locs = np.max(np.abs(xy_locs), axis=0)
        dist_filter = max_locs < self.xy_range
        return np.array(boxes)[dist_filter].tolist()

    def _transform_to_sensor_refframe(self, boxes):
        sensor = self.nusc.get('calibrated_sensor', self.sample_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', self.sample_data['ego_pose_token'])
        return bboxes_glob2sensor(boxes, ego_pose, sensor)

    def velocity_backtrack(self, global_boxes, time_delta, cur_sample_data, remove_gt=True):
        displaced_boxes = []
        for box in global_boxes:
            velocity = box.velocity if not np.any(np.isnan(box.velocity)) else np.zeros(3, 1)
            displaced_box = box.copy()
            displaced_box.center[:2, :] -= velocity[:2, :] * time_delta
            if remove_gt:
                displaced_box.remove_gt_box()  # we not longer have GT information
            displaced_box.timestep -= time_delta
            displaced_boxes.append(displaced_box)

        obj_type = type(self)
        time = 1e-6 * cur_sample_data['timestamp']
        displaced_object_props = obj_type(self.nusc, displaced_boxes, cur_sample_data, time, np.inf)
        displaced_object_props.format()

        return displaced_object_props

    def convert2caspr(self):
        for box in self.boxes:
            box.wlh[[0, 1, 2], :] = box.wlh[[1, 0, 2], :]
            if box.with_gt_box:
                box.gt_wlh[[0, 1, 2], :] = box.gt_wlh[[1, 0, 2], :]

    def convert2mmdetboxes(self, gt=False):
        # Convert into arrays
        if len(self.boxes) == 0:
            return np.zeros((0, 7))

        center_attr = "center" if not gt else "gt_center"
        dims_attr = "wlh" if not gt else "gt_wlh"
        orientation_attr = "orientation" if not gt else "gt_orientation"

        locs = np.concatenate([getattr(b, center_attr).copy() for b in self.boxes], axis=1).transpose()
        dims = np.concatenate([getattr(b, dims_attr).copy() for b in self.boxes], axis=1).transpose()
        headings = np.array([quaternion_yaw(getattr(b, orientation_attr)) for b in self.boxes]).reshape(-1, 1)

        # Convert into mmdetection coordinate frame
        locs[:, 2] -= dims[:, 2] / 2  # mmdetection has center at bottom of the box
        headings *= -1
        headings -= np.pi / 2
        headings[headings > np.pi] -= 2 * np.pi
        headings = headings.reshape(-1, 1)

        # concatenate and return
        bounding_boxes = np.concatenate([locs, dims, headings], axis=1)
        return bounding_boxes

    def expand_sizes(self, scale):
        expanded_boxes = []
        for box in self.boxes:
            expanded_box = box.copy()
            expanded_box.wlh *= scale
            expanded_boxes.append(expanded_box)
        copied_properties = self.copy()
        copied_properties.boxes = expanded_boxes
        return copied_properties

    def copy(self):
        copy = self.__class__(self.nusc, [], self.sample_data, self.timestep, self.xy_range)
        copy.boxes = [box.copy() for box in self.boxes]
        return copy

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, item):
        new_boxes = np.array(self.boxes)[item].tolist()
        return type(self)(self.nusc, new_boxes, self.sample_data, self.timestep, np.inf, self.global2sensor)

    @staticmethod
    def identity_max(arr, axis):
        if arr.shape[0] == 0:
            return arr
        else:
            return np.max(arr, axis=axis)

