from spot.io.globals import NUSCENES_IDX2CLASSES, DEFAULT_ATTRIBUTE, NUSCENES_CLASSES, WAYMO_NAME2CLASSPROTO
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import numpy as np
from nuscenes.eval.common.loaders import load_gt
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes import NuScenes
from spot.tracking_utils.track_scene import SceneTracking
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
import wandb
import os
from torch_geometric.loader import DataListLoader
from spot.io.logging_utils import log_runtime_stats

import tqdm
import gc
DEBUG = False

class TrackAndEvaluate:

    def __init__(self, dataloader, dataloader_omitted, class_name, dataset_source,
                 management_cfg, trajectory_cfg, association_cfg, out_dir):
        # defines tracking dataset
        self.dataset_source = dataset_source
        self.num_scenes = dataloader.dataset.num_scenes()
        self.total_samples = sum(dataloader.dataset.num_samples(i) for i in range(self.num_scenes))
        self.class_name = class_name
        self.dataloader = dataloader
        self.dataloader_omitted = dataloader_omitted
        self.out_dir = os.path.join(out_dir, "media")
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # datset-level parameters to keep track of
        self.num_initiated_tracks = 0
        self.observations = {}  # fill with frame-id --> all-detections info
        self.trackable_observation = {}
        self.runtimes = []

        # parameters to pass into SceneTracking
        self.track_management_params = management_cfg
        self.track_assoc_params = association_cfg
        self.track_trajectory_params = trajectory_cfg

    def track_and_evaluate(self, tpointnet, device, parallel, max_pnts_in_batch):
        tpointnet.eval()

        if self.track_management_params['refinement']['refinement-strategy'] == "none":
            print("NOTE: ONLY RUNNING WITH ORIGINAL BASELINE INFO. NOT USING OUR OBJECT REPRESENTATION.")

        pbar = tqdm.tqdm(total=self.num_scenes)
        for scene_idx in range(self.num_scenes):
            if self.dataset_source == "waymo" and (scene_idx+1) % 10 == 0:
                self._release_dataloader_memory()

            # run tracking on this scene
            assert self.dataloader.dataset.cur_scene_idx == scene_idx
            scene_tracking = SceneTracking(scene_idx, self.dataloader, self.dataloader_omitted, tpointnet, self.track_management_params, self.track_trajectory_params, self.track_assoc_params,
                                           device, parallel, max_pnts_in_batch, self.num_initiated_tracks, self.class_name, self.dataset_source, self.out_dir)
            scene_outputs = scene_tracking.run()
            scene_samp_tokens, scene_boxes, scene_confs, scene_vels, scene_trackids, scene_is_tracking, scene_ntracks, runtimes = scene_outputs

            # format for nuscenes output
            for j, frame_id in enumerate(scene_samp_tokens):
                frame_boxes = self.convert_frame_to_boxes(scene_idx, frame_id, scene_boxes[j], scene_confs[j], scene_trackids[j], scene_vels[j], scene_is_tracking[j])
                self.observations[frame_id] = frame_boxes
                self.trackable_observation[frame_id] = scene_is_tracking
            self.runtimes += runtimes

            # record
            self.num_initiated_tracks += scene_ntracks
            pbar.update()
            gc.collect()

        pbar.close()

        log_runtime_stats(self.runtimes)
        self.confidence_histogram()
        detection_submission, tracking_submission = self.format_observations()
        return detection_submission, tracking_submission

    def convert_frame_to_boxes(self, scene_idx, frame_id, pred_boxes, scores, track_ids, velocities, is_tracking):
        raise NotImplementedError("Method for Child Class.")

    def format_observations(self):
        raise NotImplementedError("Method for Child Class.")

    def confidence_histogram(self):
        raise NotImplementedError("Method for Child Class.")

    def _release_dataloader_memory(self):
        cur_dataset = self.dataloader.dataset
        cur_omitted_dataset = self.dataloader_omitted.dataset
        num_workers = self.dataloader.num_workers
        del self.dataloader
        del self.dataloader_omitted
        gc.collect()
        self.dataloader = DataListLoader(cur_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
        self.dataloader_omitted = DataListLoader(cur_omitted_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)


class TrackAndEvaluateNuscenes(TrackAndEvaluate):

    def __init__(self, dataloader, dataloader_omitted, class_name,
                 nusc_version, nusc_path, management_cfg, trajectory_cfg, association_cfg, out_dir):
        super().__init__(dataloader, dataloader_omitted, class_name, "nuscenes", management_cfg, trajectory_cfg, association_cfg, out_dir)

        # defines tracking dataset
        self.class_idx = NUSCENES_CLASSES[class_name]
        self.nusc_version = nusc_version
        self.nusc_path = nusc_path

        # datset-level parameters to keep track of
        self.modality = dict(
            use_camera=False,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False)

    def convert_frame_to_boxes(self, scene_idx, frame_id, pred_boxes, scores, track_ids, velocities, is_tracking):
        return output_to_nusc_box(pred_boxes, scores, self.class_idx, track_ids, velocities, is_tracking)

    def format_observations(self):
        detection_submission = self._format_observations(self.nusc_version, self.nusc_path, eval_type="detection")
        tracking_submission = self._format_observations(self.nusc_version, self.nusc_path, eval_type="tracking")
        return detection_submission, tracking_submission

    def confidence_histogram(self):
        histogram_nusc_confidences(self.observations)

    def _format_observations(self, nusc_version, nusc_path, eval_type="tracking"):
        nusc_annos = {}
        for sample_token, boxes in self.observations.items():
            annos = []
            for i, box in enumerate(boxes):
                if eval_type=="tracking" and not box.tracking:  # removes detections with less than min_age frames associated
                    continue
                name = NUSCENES_IDX2CLASSES[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        'car',
                        'construction_vehicle',
                        'bus',
                        'truck',
                        'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = DEFAULT_ATTRIBUTE[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = DEFAULT_ATTRIBUTE[name]

                if eval_type == "tracking":
                    nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=str(box.token))
                else:
                    nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        gt_boxes, all_tokens = get_all_nuscenes_samples(nusc_version, nusc_path)
        for token in all_tokens:
            if token not in nusc_annos:
                nusc_annos[token] = []

        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        return nusc_submissions


class TrackAndEvaluateWaymo(TrackAndEvaluate):

    def __init__(self, dataloader, dataloader_omitted, class_name, management_cfg, trajectory_cfg, association_cfg, out_dir):
        super().__init__(dataloader, dataloader_omitted, class_name, "waymo", management_cfg, trajectory_cfg, association_cfg, out_dir)


    def convert_frame_to_boxes(self, scene_idx, frame_id, pred_boxes, scores, track_ids, velocities, is_tracking):
        scene_name = self.dataloader.dataset.get_scene_name(scene_idx=scene_idx)
        waymo_timestamp = int(frame_id)
        return output_to_waymo_box(pred_boxes, scores, track_ids, velocities, is_tracking, self.class_name, waymo_timestamp, scene_name)

    def format_observations(self):
        combined_detection, combined_tracking = metrics_pb2.Objects(), metrics_pb2.Objects()
        for scene_context, objects in self.observations.items():
            detection_objects, tracking_objects = objects
            for o in detection_objects.objects:
                combined_detection.objects.append(o)
            for o in tracking_objects.objects:
                combined_tracking.objects.append(o)

        return combined_detection, combined_tracking

    def confidence_histogram(self):
        histogram_waymo_confidences(self.observations)


def output_to_nusc_box(pred_boxes, scores, label_idx, track_ids, velocities, is_tracking):
    if pred_boxes is None:
        return []

    box_gravity_center = pred_boxes[:, :3]
    box_dims = pred_boxes[:, 3:6]
    box_dims[:, [0, 1]] = box_dims[:, [1, 0]]
    box_yaw = pred_boxes[:, 6]

    box_list = []
    for i in range(pred_boxes.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if np.any(box_dims[i] <= 0):
            print("Size Error!")
            print(box_dims[i])
            print("!!!!!!!!!!!!!!!!!!!")
            continue
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=label_idx,
            score=scores[i],
            velocity=velocities[i],
            token=track_ids[i])
        box.tracking = is_tracking[i]
        box_list.append(box)
    return box_list


def output_to_waymo_box(pred_boxes, scores, track_ids, velocities, is_tracking, class_name, frame_timestep, scene_name):
    detection_objects = metrics_pb2.Objects()
    tracking_objects = metrics_pb2.Objects()

    if pred_boxes is None:
        return (detection_objects, tracking_objects)

    for i in range(pred_boxes.shape[0]):
        length = round(pred_boxes[i, 3].item(), 4)
        height = round(pred_boxes[i, 5].item(), 4)
        width = round(pred_boxes[i, 4].item(), 4)
        x = round(pred_boxes[i, 0].item(), 4)
        y = round(pred_boxes[i, 1].item(), 4)
        z = round(pred_boxes[i, 2].item(), 4)
        yaw = round(pred_boxes[i, 6].item(), 4)
        score = round(scores[i], 4)

        box = label_pb2.Label.Box()
        box.center_x = x
        box.center_y = y
        box.center_z = z
        box.length = length
        box.width = width
        box.height = height
        box.heading = yaw

        metadata = label_pb2.Label.Metadata()
        metadata.speed_x = round(velocities[i, 0].item(), 4)
        metadata.speed_y = round(velocities[i, 1].item(), 4)

        o = metrics_pb2.Object()
        o.object.box.CopyFrom(box)
        o.object.type = WAYMO_NAME2CLASSPROTO[class_name]
        o.object.id = str(track_ids[i])
        o.object.metadata.CopyFrom(metadata)
        o.score = score

        o.context_name = scene_name
        o.frame_timestamp_micros = int(frame_timestep)
        detection_objects.objects.append(o)
        if is_tracking[i]:
            tracking_objects.objects.append(o)

    return (detection_objects, tracking_objects)


def histogram_waymo_confidences(all_boxes):
    confidences = []
    for frame_boxes in all_boxes.values():
        frame_confidences = [o.score for o in frame_boxes[0].objects]
        confidences += frame_confidences
    confidences = np.array(confidences)

    plot_histogram(confidences)


def histogram_nusc_confidences(all_boxes):
    confidences = []
    for frame_boxes in all_boxes.values():
        frame_confidences = [o.score for o in frame_boxes]
        confidences += frame_confidences
    confidences = np.array(confidences)

    plot_histogram(confidences)

def plot_histogram(confidences):
    confidences = confidences.reshape(-1, 1).tolist()
    table = wandb.Table(data=confidences, columns=["confidences"])
    wandb.log({'Confidence Distribution': wandb.plot.histogram(table, "confidences")})



def get_all_nuscenes_samples(nusc_version, nusc_path):
    nusc = NuScenes(version=nusc_version, dataroot=nusc_path)
    if nusc_version != "v1.0-test":
        gt_boxes = load_gt(nusc, "val", DetectionBox, verbose=True)
    else:
        gt_boxes = get_nuscenes_test_samples(nusc)
    return gt_boxes, set(gt_boxes.sample_tokens)


def get_nuscenes_test_samples(nusc):
    from nuscenes.utils.splits import create_splits_scenes
    from nuscenes.eval.common.data_classes import EvalBoxes

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits['test']:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()
    for sample_token in tqdm.tqdm(sample_tokens, leave=True):
        all_annotations.add_boxes(sample_token, [])
    return all_annotations
