from nuscenes.utils import splits
import tqdm
from spot.data.loading.scene import SceneObjects
import os

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from preprocessing.nuscenes.utils import load_gt_local
from preprocessing.nuscenes.nuscenes_scene import NuScenesScene
import json
from pyquaternion import Quaternion
import numpy as np
from nuscenes.eval.tracking.evaluate import TrackingEval
import matplotlib.pyplot as plt
# TRACKING_CLASSES = ["bicycle", "motorcycle", "pedestrian", "bus", "car", "trailer", "truck"]
TRACKING_CLASSES = ["bicycle", "motorcycle", "pedestrian", "bus", "car", "trailer", "truck"]


class BestTrackingEstimator:
    def __init__(self,
                 centerpoint_results_path,
                 version,
                 nuscenes_path,
                 save_dir,
                 split='train',
                 correct_confidences=False,
                 fully_swap_with_gt=False):
        """
        neighbors_radius: Radius in meters surrounding object center to consider "neighbors" at that frame
        """
        assert version in ['v1.0-trainval', 'v1.0-mini']  # we don't support test bc no bounding boxes
        if version == 'v1.0-trainval':
            self.train_scenes = splits.train
            self.val_scenes = splits.val
        else:  # version == 'v1.0-mini'
            self.train_scenes = splits.mini_train
            self.val_scenes = splits.mini_val
        self.global_counter = 0

        self.version = version
        self.nuscenes_path = nuscenes_path
        self.save_path = os.path.join(save_dir)
        self.centerpoint_results_path = centerpoint_results_path
        self.split = split
        self.correct_confidences = correct_confidences
        self.fully_swap_with_gt = fully_swap_with_gt
        self.correct_confidences = True if self.fully_swap_with_gt else self.correct_confidences

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.nusc = NuScenes(version=self.version, dataroot=self.nuscenes_path, verbose=True)
        self.result_path = centerpoint_results_path
        self.cfg = config_factory('detection_cvpr_2019')

        # Check result file exists.
        assert os.path.exists(centerpoint_results_path), 'Error: The result file does not exist!'

        # Load data.
        self.pred_boxes, self.meta = load_prediction(self.result_path, 100000, DetectionBox, verbose=True)
        self.gt_boxes = load_gt_local(self.nusc, self.split, DetectionBox, verbose=True)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(self.nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(self.nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.nusc, self.pred_boxes, self.cfg.class_range, verbose=True)
        print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.nusc, self.gt_boxes, self.cfg.class_range, verbose=True)
        self.sample_tokens = self.gt_boxes.sample_tokens

        # set up evaluation reps
        self.modality = dict(
            use_camera=False,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False)
        self.annos = {token: [] for token in self.sample_tokens}

    def run_tracking_eval(self):
        # generate a list of every scene in our data folder
        scenes = self.nusc.scene
        scene2split = {scenes[i]['token']: self.get_scene2split(scenes[i]['name']) for i in range(len(scenes))}

        # Save data information: Dict[vehicle_id] --> List[idx] --> Dict of vehicle info
        progress_bar = tqdm.tqdm(total=len(self.nusc.scene), desc='create_info', dynamic_ncols=True)
        for i, scene in enumerate(self.nusc.scene):
            if self.split != scene2split[scene['token']]:
                continue
            # processes + records scene annotations in self.annos
            self.process_predictions(scene)
            progress_bar.update()

        nusc_submissions = {
            'meta': self.modality,
            'results': self.annos,
        }

        # todo: plot histogram here based on self.annos
        for tracking_class in TRACKING_CLASSES:
            self.plot_tracklen_histogram(tracking_class)

        # track_results_outpath = os.path.join(self.save_path, "tracking_results.json")
        # with open(track_results_outpath, "w") as f:
        #     json.dump(nusc_submissions, f)
        # track_cfg = config_factory('tracking_nips_2019')
        # track_eval = TrackingEval(config=track_cfg, result_path=track_results_outpath, eval_set="val",
        #                           output_dir=os.path.join(self.save_path, "tracking"), nusc_version=self.version,
        #                           nusc_dataroot=self.nuscenes_path, verbose=True)  # , render_classes=["pedestrian"]
        # track_eval.main(render_curves=True)

        progress_bar.close()

    def process_predictions(self, scene):
        samples_with_preds = set(self.pred_boxes.sample_tokens)
        sample = self.nusc.get('sample', scene['first_sample_token'])
        while True:
            if sample['token'] in samples_with_preds:
                samp_gt_boxes, samp_pred_boxes = self.gt_boxes[sample['token']], self.pred_boxes[sample['token']]
                sample_annos = self.associate_boxes(samp_gt_boxes, samp_pred_boxes, sample['token'])
                self.annos[sample['token']] = sample_annos

            if sample['next'] == '':
                break
            sample = self.nusc.get('sample', sample['next'])

    def associate_boxes(self, gt_boxes, pred_boxes, sample_token):
        """
        Args:
            gt_boxes: List of NuScenes box objects
            pred_boxes: List of NuScenes box objects
        Returns:

        """
        # ---------------------------------------------
        # Organize input and initialize accumulators.
        # ---------------------------------------------

        cfg = config_factory('detection_cvpr_2019')

        # Organize the predictions in a single list.
        pred_confs = [box.detection_score for box in pred_boxes]

        # Sort by confidence.
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

        # ---------------------------------------------
        # Match and accumulate match data.
        # ---------------------------------------------
        matched_pred_bboxes, matched_gt_bboxes, unmatched_pred_bboxes = [], [], []
        taken = set()  # Initially no gt bounding box is matched.
        for ind in sortind:
            pred_box = pred_boxes[ind]
            min_dist = np.inf
            match_gt_idx = None

            for gt_idx, gt_box in enumerate(gt_boxes):
                # Find closest match among ground truth boxes
                if gt_box.detection_name == pred_box.detection_name and not (pred_box.sample_token, gt_idx) in taken:
                    this_distance = cfg.dist_fcn_callable(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx

            # If the closest match is close enough according to threshold we have a match!
            MAX_ASSOCIATION_DIST = 2.0
            is_match = min_dist < MAX_ASSOCIATION_DIST

            if is_match:
                taken.add((pred_box.sample_token, match_gt_idx))
                matched_pred_bboxes.append(pred_box)
                matched_gt_bboxes.append(gt_boxes[match_gt_idx])
            else:
                unmatched_pred_bboxes.append(pred_box)

        annos = []
        # Add appropriate matches
        for i in range(len(matched_pred_bboxes)):
            if matched_pred_bboxes[i].detection_name not in TRACKING_CLASSES:
                continue
            nusc_anno = dict(
                sample_token=sample_token,
                translation=list(matched_pred_bboxes[i].translation) if not self.fully_swap_with_gt else list(matched_gt_bboxes[i].translation),
                size=list(matched_pred_bboxes[i].size) if not self.fully_swap_with_gt else list(matched_gt_bboxes[i].size),
                rotation=Quaternion(list(matched_pred_bboxes[i].rotation)).elements.tolist() if not self.fully_swap_with_gt else Quaternion(list(matched_gt_bboxes[i].rotation)).elements.tolist(),
                velocity=list(matched_pred_bboxes[i].velocity) if not self.fully_swap_with_gt else list(matched_gt_bboxes[i].velocity),
                tracking_name=matched_pred_bboxes[i].detection_name,
                tracking_score=matched_pred_bboxes[i].detection_score if not self.correct_confidences else 1.0,
                tracking_id=matched_gt_bboxes[i].instance_token)
            annos.append(nusc_anno)

        # convert unmatched boxes to regular box format
        for i in range(len(unmatched_pred_bboxes)):
            if unmatched_pred_bboxes[i].detection_name not in TRACKING_CLASSES or self.fully_swap_with_gt:
                continue
            nusc_anno = dict(
                sample_token=sample_token,
                translation=list(unmatched_pred_bboxes[i].translation),
                size=list(unmatched_pred_bboxes[i].size),
                rotation=Quaternion(list(unmatched_pred_bboxes[i].rotation)).elements.tolist(),
                velocity=list(unmatched_pred_bboxes[i].velocity),
                tracking_name=unmatched_pred_bboxes[i].detection_name,
                tracking_score=unmatched_pred_bboxes[i].detection_score if not self.correct_confidences else 0.0,
                tracking_id=str(self.global_counter))
            self.global_counter += 1
            annos.append(nusc_anno)

        return annos

    def get_scene2split(self, scene_name):
        if scene_name in self.train_scenes:
            return "train"
        elif scene_name in self.val_scenes:
            return "val"
        else:
            return "test"

    def plot_tracklen_histogram(self, tracking_class):
        all_instance_ids = []
        for sample_id, annos in self.annos.items():
            anno_track_ids = [anno['tracking_id'] for anno in annos if anno['tracking_name'] == tracking_class]
            all_instance_ids += anno_track_ids
        _, occurances = np.unique(np.array(all_instance_ids), return_counts=True)
        # occurances = occurances.tolist()

        plt.hist(occurances)
        plt.xlabel('Track Length')
        plt.ylabel('Num Objects')
        plt.title('Class %s Instances' % tracking_class)
        # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(40, 160)
        # plt.ylim(0, 0.03)
        # plt.grid(True)
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # data input/output arguments
    parser.add_argument('--data-path', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/tri-nuscenes", help="Path to raw nuscenes dataset")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--save-dir', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/cp-upperbound-results-confs")

    # data processing parameters
    parser.add_argument('--cp-results-path', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/nuscenes-processed-cp/results_val_nusc.json", help="Path to centerpoint predictions.")
    parser.add_argument('--split', type=str, default="val", help="One of ['train', 'val', 'test'] for which to process")
    parser.add_argument('--correct-confidences', action="store_true", help="If provided, set TP tracks to confidence 1 and FP tracks to confidence 0.")
    parser.add_argument('--fully_swap_with_gt', action='store_true', help='Removes false positives. Swaps true positives with their GT values.')

    args = parser.parse_args()

    nuscenes_processor = BestTrackingEstimator(centerpoint_results_path=args.cp_results_path,
                                                    version=args.version,
                                                    nuscenes_path=args.data_path,
                                                    save_dir=args.save_dir,
                                                    split = args.split,
                                               correct_confidences=args.correct_confidences,
                                               fully_swap_with_gt=args.fully_swap_with_gt)
    nuscenes_processor.run_tracking_eval()





