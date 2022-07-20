
from spot.io.nuscenes_rendering import TrackingVisualizer
import wandb
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--nuscenes-path', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/tri-nuscenes", help="Path to raw nuscenes dataset")
parser.add_argument('--nuscenes-version', type=str, default="v1.0-trainval")
parser.add_argument('--save-dir', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/out")
parser.add_argument('--results-path', type=str, default="/mnt/fsx/scratch/colton.stearns/pi/nuscenes-processed-cp/results_val_nusc.json", help="Path to centerpoint predictions.")

args = parser.parse_args()

track_cfg = config_factory('tracking_nips_2019')


# Run NuScenes Evaluation
track_eval = TrackingEval(config=track_cfg, result_path=args.results_path, eval_set="val",
                          output_dir=os.path.join(args.save_dir, "tracking"), nusc_version=args.nuscenes_version,
                          nusc_dataroot=args.nuscenes_path, verbose=True)  # , render_classes=["pedestrian"]
tracking_summary = track_eval.main(render_curves=True)