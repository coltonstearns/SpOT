import pickle
from preprocessing.waymo.load_objects import extract_predictions
from waymo_open_dataset import dataset_pb2 as open_dataset
import mmcv
import numpy as np
from glob import glob
from os.path import join
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from preprocessing.common.bounding_box import Box3D
from preprocessing.waymo.common import global_vel_to_ref
import wandb




def compare_preds_and_gt(preds_file, gt_file):
    wandb.init(project="waymo-data-preprocess", config={})

    loaded_predictions = extract_predictions(preds_file)
    loaded_gt = extract_predictions(gt_file)

    for scene_id in loaded_predictions.keys():
        for timestep in loaded_predictions[scene_id].keys():
            wandb_boxes = []
            for j in range(len(loaded_predictions[scene_id][timestep])):
                corners = loaded_predictions[scene_id][timestep][j].prediction_corners()
                wandb_box = {'corners': corners.tolist(),
                             "label": "",
                             "color": [255, 0, 255]}
                wandb_boxes.append(wandb_box)

            if scene_id not in loaded_gt:
                continue
            if timestep not in loaded_gt[scene_id]:
                continue
            for j in range(len(loaded_gt[scene_id][timestep])):
                corners = loaded_gt[scene_id][timestep][j].prediction_corners()
                wandb_box = {'corners': corners.tolist(),
                             "label": "",
                             "color": [0, 255, 0]}
                wandb_boxes.append(wandb_box)

            wandb_boxes = np.array(wandb_boxes)
            wandb.log({"Centerpoint Preprocessing": wandb.Object3D({"type": "lidar/beta", "points": np.zeros((1, 3)), "boxes": wandb_boxes})})



if __name__ == "__main__":
    dets_filepath = "/home/colton/Documents/CenterPointResults/waymo/val-detections.pkl"
    dets_binpath = "/home/colton/Documents/CenterPointResults/waymo/val/tracking_pred_0.bin"
    outpath = "/home/colton/Documents/val-detections-processed.bin"
    gt_path = "/home/colton/Documents/CenterPointResults/waymo/gt.bin"
    compare_preds_and_gt(dets_binpath, gt_path)

    # loaded_predictions = extract_predictions(dets_binpath)
    # predictions_to_output(predictions=loaded_predictions, outfile=outpath)


# with open(dets_filepath, 'rb') as f:
#     predictions = pickle.load(f)

