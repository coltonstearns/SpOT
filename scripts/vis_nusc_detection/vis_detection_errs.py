from nuscenes.eval.common.config import config_factory
import os
from spot.io.globals import DEFAULT_ATTRIBUTE
import json
from typing import Any

import numpy as np

from nuscenes import NuScenes
from scripts.vis_nusc_detection.nuscenes_error_analysis import VisualizeDetectionError
import wandb

Axis = Any

def convert_tracking_to_detection(results_file):
    with open(results_file, "r") as f:
        track_boxes = json.load(f)
    meta = track_boxes['meta']

    detection_results = {}
    for sample_token, boxes in track_boxes['results'].items():
        annos = []
        for i, box in enumerate(boxes):
            velocity = box["velocity"]
            name = box['tracking_name']
            if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) > 0.2:
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

            nusc_anno = dict(
                             sample_token=box['sample_token'],
                             translation=box['translation'],
                             size=box['size'],
                             rotation=box['rotation'],
                             velocity=velocity,
                             detection_name=name,
                             detection_score=box['tracking_score'],
                             attribute_name=attr)

            annos.append(nusc_anno)
        detection_results[sample_token] = annos


    nusc_submissions = {
        'meta': meta,
        'results': detection_results,
    }
    return nusc_submissions


wandb.init(project="nuscenes-ap-analysis", config={})

# run official NuScenes Detection Evaluation
version = "v1.0-trainval"
nusc_path = "/media/colton/ColtonSSD/nuscenes-raw"
outdir = "/home/colton/Documents/NuScenes-mAP-dive"
our_dets_path = "/home/colton/Documents/NuScenes-mAP-dive/fixed_our_detections.json"
cp_dets_path = "/home/colton/Documents/NuScenes-mAP-dive/fixed_baseline_detections.json"


nusc = NuScenes(version=version, dataroot=nusc_path, verbose=True)
det_cfg = config_factory('detection_cvpr_2019')
visualizer = VisualizeDetectionError(nusc, det_cfg, our_dets_path, cp_dets_path, eval_set="val",
                                     output_dir=os.path.join(outdir, "detection"), verbose=True)
visualizer.run("car")


# our_det_eval = DetectionEval(nusc, config=det_cfg, result_path=our_dets_path, eval_set="val",
#                          output_dir=os.path.join(outdir, "detection"), verbose=True)
# cp_det_eval = DetectionEval(nusc, config=det_cfg, result_path=cp_dets_path, eval_set="val",
#                          output_dir=os.path.join(outdir, "detection"), verbose=True)
# detection_summary = our_det_eval.main(plot_examples=100, render_curves=True)

# random.seed(42)
# sample_tokens = list(our_det_eval.sample_tokens)
# random.shuffle(sample_tokens)
# sample_tokens = sample_tokens[:50]
#
# # Visualize samples.
# example_dir = os.path.join(os.path.join(outdir, "detection"), 'examples')
# if not os.path.isdir(example_dir):
#     os.mkdir(example_dir)
# for sample_token in sample_tokens:
#     visualize_sample(nusc,
#                      sample_token,
#                      our_det_eval.gt_boxes,
#                      our_det_eval.pred_boxes,
#                      eval_range=max(det_cfg.class_range.values()),
#                      conf_th=0.0,
#                      savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))


# ==================== COUNT BOXES ===================
# num_our_boxes = 0
# pred_boxes, _ = load_prediction(our_dets_path, 100000, DetectionBox,verbose=True)
# pred_boxes = add_center_dist(nusc, pred_boxes)
# pred_boxes = filter_eval_boxes(nusc, pred_boxes, det_cfg.class_range, verbose=True)
# for ind, sample_token in enumerate(pred_boxes.sample_tokens):
#     num_our_boxes += len(pred_boxes[sample_token])

# num_cp_boxes = 0
# baseline_boxes, _ = load_prediction(cp_dets_path, 100000, DetectionBox,verbose=True)
# baseline_boxes = add_center_dist(nusc, baseline_boxes)
# baseline_boxes = filter_eval_boxes(nusc, baseline_boxes, det_cfg.class_range, verbose=True)
# for ind, sample_token in enumerate(baseline_boxes.sample_tokens):
#     num_cp_boxes += len(baseline_boxes[sample_token])


# print(num_our_boxes)
# print(num_cp_boxes)


