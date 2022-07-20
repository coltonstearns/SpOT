from nuscenes.utils import splits
import tqdm
from spot.data.loading.scene import SceneIO
import os

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox
from preprocessing.nuscenes.utils import load_gt_local
from preprocessing.nuscenes.nuscenes_scene import NuScenesScene
from preprocessing.nuscenes.nuscenes_ops import nusc_boxes2our_boxes
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from multiprocessing import Pool
import multiprocessing as mp


class ProcessCenterPointNuscenes:
    def __init__(self,
                 centerpoint_results_path,
                 version,
                 nuscenes_path,
                 save_dir,
                 xy_range=51,
                 sweeps_to_aggregate=5,
                 bbox_size_factor=1.25,
                 num_procs=1,
                 split='train'):
        """
        neighbors_radius: Radius in meters surrounding object center to consider "neighbors" at that frame
        """
        assert version in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']  # we don't support test bc no bounding boxes
        if version == 'v1.0-trainval':
            self.train_scenes = list(splits.train)
            self.val_scenes = list(splits.val)
        else:  # version == 'v1.0-mini'
            self.train_scenes = list(splits.mini_train)
            self.val_scenes = list(splits.mini_val)

        self.version = version
        self.nuscenes_path = nuscenes_path
        self.save_path = os.path.join(save_dir)
        self.centerpoint_results_path = centerpoint_results_path
        self.split = split
        self.num_procs = num_procs

        # bbox filtration parameters
        self.bbox_size_factor = bbox_size_factor
        self.xy_range = xy_range
        self.sweeps_to_aggregate = sweeps_to_aggregate

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(os.path.join(self.save_path, self.split)):
            os.makedirs(os.path.join(self.save_path, self.split))

        self.nusc = NuScenes(version=self.version, dataroot=self.nuscenes_path, verbose=True)

        # Check result file exists.
        assert os.path.exists(centerpoint_results_path), 'Error: The result file does not exist!'

        # Load data.
        self.pred_boxes, self.meta = load_prediction(centerpoint_results_path, 100000, TrackingBox, verbose=True)
        self.gt_boxes = load_gt_local(self.nusc, self.split, DetectionBox, verbose=True)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."
        self.split_samples = list(self.pred_boxes.sample_tokens)

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        print('Filtering predictions')
        cfg = config_factory('detection_cvpr_2019')
        # Important: filtering prediction boxes reduces performance on tracking! Only filter GT boxes!
        # self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, cfg.class_range, verbose=True)
        # print('Filtering ground truth annotations')
        if self.split != "test":
            self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, cfg.class_range, verbose=True)

        # convert into our bounding box format
        our_pred_boxes = {}
        for ind, sample_token in enumerate(self.pred_boxes.sample_tokens):
            timestep = self.nusc.get('sample', sample_token)['timestamp'] * 1e-6
            sample_pred_boxes = nusc_boxes2our_boxes(self.pred_boxes.boxes[sample_token], timestep, box_type="prediction")
            our_pred_boxes[sample_token] = sample_pred_boxes
        self.pred_boxes = our_pred_boxes

        # convert gt into our bounding box format
        our_gt_boxes = {}
        for ind, sample_token in enumerate(self.gt_boxes.sample_tokens):
            timestep = self.nusc.get('sample', sample_token)['timestamp']
            sample_gt_boxes = nusc_boxes2our_boxes(self.gt_boxes.boxes[sample_token], timestep, box_type="ground-truth")
            our_gt_boxes[sample_token] = sample_gt_boxes
        self.gt_boxes = our_gt_boxes

    def preprocess_parallel(self):
        print("start converting ...")
        with Pool(self.num_procs) as p:
            r = list(tqdm.tqdm(p.imap(self.process_one_scene, range(len(self.nusc.scene))), total=len(self.nusc.scene)))
        print("\nfinished ...")

    def process_one_scene(self, idx):
        scene2split = {self.nusc.scene[i]['token']: self.get_scene2split(self.nusc.scene[i]['name']) for i in range(len(self.nusc.scene))}

        # Save data information: Dict[vehicle_id] --> List[idx] --> Dict of vehicle info
        scene = self.nusc.scene[idx]
        if self.split != scene2split[scene['token']]:
            return

        # instantiate scene objects and wrapper
        scene_objects = SceneIO(scene['name'], base_dir=os.path.join(self.save_path, self.split))
        scene_wrapper = NuScenesScene(self.nusc, scene, scene_objects, self.nuscenes_path, self.version, self.sweeps_to_aggregate,
                                      self.xy_range, self.bbox_size_factor)
        scene_wrapper.process(self.pred_boxes, self.gt_boxes, self.split_samples)

        # close our scene objects to prepare for saving
        scene_objects.close_filesystem_pipe()

        # save scene metadata (dense object data already saved)
        serialized_scene_objects = scene_objects.dump_to_dict()
        with open(os.path.join(self.save_path, self.split, f'scene_%s_objects.json' % (idx+1)), 'w') as f:
            json.dump(serialized_scene_objects, f)

    def preprocess(self):
        progress_bar = tqdm.tqdm(total=len(self.nusc.scene), desc='create_info', dynamic_ncols=True)
        for idx in range(len(self.nusc.scene)):
            scene2split = {self.nusc.scene[i]['token']: self.get_scene2split(self.nusc.scene[i]['name']) for i in
                           range(len(self.nusc.scene))}

            # Save data information: Dict[vehicle_id] --> List[idx] --> Dict of vehicle info
            scene = self.nusc.scene[idx]
            if self.split != scene2split[scene['token']]:
                continue

            # instantiate scene objects and wrapper
            scene_objects = SceneIO(scene['name'], base_dir=os.path.join(self.save_path, self.split))
            scene_wrapper = NuScenesScene(self.nusc, scene, scene_objects, self.nuscenes_path, self.version,
                                          self.sweeps_to_aggregate,
                                          self.xy_range, self.bbox_size_factor)
            scene_wrapper.process(self.pred_boxes, self.gt_boxes, self.split_samples)

            # close our scene objects to prepare for saving
            scene_objects.close_filesystem_pipe()

            # save scene metadata (dense object data already saved)
            serialized_scene_objects = scene_objects.dump_to_dict()
            with open(os.path.join(self.save_path, self.split, f'scene_%s_objects.json' % (idx + 1)), 'w') as f:
                json.dump(serialized_scene_objects, f)

            progress_bar.update()

        progress_bar.close()

    def get_scene2split(self, scene_name):
        if scene_name in self.train_scenes:
            return "train"
        elif scene_name in self.val_scenes:
            return "val"
        else:
            return "test"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # mp.set_start_method('spawn')
    # data input/output arguments
    parser.add_argument('--data-path', type=str, default="/media/colton/ColtonSSD/nuscenes-raw", help="Path to raw nuscenes dataset")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--save-dir', type=str, default="/home/colton/Documents/datasets/preprocess_debug")

    # data processing parameters
    parser.add_argument('--xy-range', type=float, default=51, help="Outside of +- this range, we ignore objects.")
    parser.add_argument('--sweeps', type=int, default=10, help="Number of lidar sweeps use per keyframe. This should be 10 if ego-aggregate is false.")
    parser.add_argument('--bbox-size-factor', type=float, default=1.25, help="A multiplicative factor of each bounding box wlh s.t. we obtain points within this larger box.")
    parser.add_argument('--track-results-path', type=str, default="/home/colton/Documents/cp-baseline/cp-tracking-result/tracking_result.json", help="Path to centerpoint tracking predictions.")
    parser.add_argument('--split', type=str, default="val", help="One of ['train', 'val', 'test'] for which to process")
    parser.add_argument('--num-processes', type=int, default=8, help="Number of parallel processes to spawn.")
    args = parser.parse_args()

    nuscenes_processor = ProcessCenterPointNuscenes(centerpoint_results_path=args.track_results_path,
                                                    version=args.version,
                                                    nuscenes_path=args.data_path,
                                                    save_dir=args.save_dir,
                                                    xy_range=args.xy_range,
                                                    sweeps_to_aggregate=args.sweeps,
                                                    bbox_size_factor=args.bbox_size_factor,
                                                    split=args.split,
                                                    num_procs=args.num_processes)
    nuscenes_processor.preprocess()
