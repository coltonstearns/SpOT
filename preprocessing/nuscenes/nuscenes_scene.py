from preprocessing.nuscenes.sample_objects_properties import KeyframeAndSweepBoxes
from preprocessing.common.bounding_box import associate
from preprocessing.nuscenes.sample_points import KeyframeAndPrevSweepPoints
import time

MAX_ASSOCIATION_DIST = 2.0  # in meters


class NuScenesScene:

    def __init__(self, nusc, scene, scene_objects, nuscenes_path, version, num_sweeps, xy_range, bbox_size_factor):
        self.nusc = nusc
        self.scene = scene
        self.nuscenes_path = nuscenes_path
        self.version = version
        self.num_sweeps = num_sweeps
        self.xy_range = xy_range
        self.bbox_size_factor = bbox_size_factor
        self.scene_objects = scene_objects

    def process(self, pred_boxes, gt_boxes, valid_sampleids):
        samples_with_preds = set(pred_boxes.keys())
        sample = self.nusc.get('sample', self.scene['first_sample_token'])
        while True:
            sample_token = sample['token']
            if sample_token not in valid_sampleids:
                continue

            # get sample data
            sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # if we don't have any predictions, skip it and record empty frames
            if sample_token not in samples_with_preds:
                self.record_empty_frames(sample, sample_data)
                sample = self.nusc.get('sample', sample['next'])
                continue

            # associate ground truth and prediction boxes
            keyframe_boxes = associate(gt_boxes[sample_token], pred_boxes[sample_token], threshold=MAX_ASSOCIATION_DIST, distance_type="l2")

            # build all object representations
            all_boxes = KeyframeAndSweepBoxes(keyframe_boxes, self.nusc, sample, sample_data, self.num_sweeps)

            # get sample lidar points
            frame_points = KeyframeAndPrevSweepPoints(self.nusc, self.nuscenes_path, self.version, sample, sample_data, self.num_sweeps)

            # go through boxes, one-sweep-at-a-time, and record the info
            self.record_sweeps(all_boxes, frame_points, sample)

            # get next sample
            if sample['next'] == '':
                break
            sample = self.nusc.get('sample', sample['next'])

    def record_sweeps(self, all_boxes, frame_points, sample):
        num_valid_sweeps = all_boxes.num_valid_sweeps()
        for i in range(num_valid_sweeps):
            boxes = all_boxes.get_sweep_boxes(sweep_idx=i)
            success, box_points = frame_points.points_in_boxes(boxes, self.bbox_size_factor)
            sample_id, timestep = sample['token'], boxes.timestep
            if success:
                self.scene_objects.init_new_sample_if_not_exist(timestep, sample_id)
                boxes.convert2caspr()
                self.scene_objects.add_frame(boxes.boxes, box_points, keyframe_id=sample_id, world2sensor=boxes.global2sensor, is_keyframe=i == 0)

            else:
                print("No objects or no points in sweep.")
                self.scene_objects.init_new_sample_if_not_exist(timestep, sample_id)

    def record_empty_frames(self, sample, sample_data):
        cur_sample_data = sample_data
        for i in range(self.num_sweeps):
            if cur_sample_data['sample_token'] != sample['token']:
                break

            time = 1e-6 * cur_sample_data['timestamp']
            self.scene_objects.init_new_sample_if_not_exist(time, sample['token'])

            if cur_sample_data['prev'] == '':
                break
            cur_sample_data = self.nusc.get('sample_data', cur_sample_data['prev'])
