

import wandb
import numpy as np
from spot.ops.torch_utils import torch_to_numpy
import torch
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from spot.ops.data_utils import batched_index_select
from spot.utils.train_utils import format_pytorchgeo_data
from torch_geometric.data import Batch
from spot.ops.pc_util import group_first_k_values
from spot.tracking_utils.object_representation import MergedEnhancedAndBaseObjectRepresentations, EmptySTObjectRepresentations
from spot.data.box3d import LidarBoundingBoxes
from spot.io.run_utils import split_inputs, update_dict
import gc
from torch_geometric.loader import DataListLoader
import torch.nn.functional as F
import os

NOCS_OFFSET = np.array([[0.5, 0.5, 0.5]])
GT_NOCS_OFFSET = np.array([[0, 0, 0]])
PRED_NOCS_OFFSET = np.array([[3, 0, 0]])
PRED_AGG_NOCS_OFFSET = np.array([[6, 0, 0]])
INPUT_POINTS_OFFSET = np.array([[10, 0, 0]])
RAW_INPUT_POINTS_OFFSET = np.array([[15, 0, 0]])
PROC_POINTS_OFFSET = np.array([[-3, 0, 0]])
PROC_BBOX_OFFSET = np.array([[6, 0, 0]])
COORD_FRAME_OFFSET = np.array([[-3, 0, 0]])
L1_ERROR_COLOR_THRESH = 0.33

LIDAR_VIZ_DOWNSCALE = 1.0 / 4.0
SAMPLE_CONTOURS_RADII = [0.25, 0.5, 1.0, 1.5, 2.25, 3.0]
COLOR_PALLETE = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                 [70, 240, 240], [240, 50, 230], [210, 245, 60], [0, 128, 128],
                 [170, 110, 40], [128, 0, 0], [128, 128, 0], [0, 0, 128]]

WANDB_NOCS_BOXES = np.array([{"corners": [[0,0,0],
                                    [0,1,0],
                                    [0,0,1],
                                    [1,0,0],
                                    [1,1,0],
                                    [0,1,1],
                                    [1,0,1],
                                    [1,1,1]],
                                "label": "",
                                "color": [255,255,255]},
                          {"corners": [[3, 0, 0],
                                       [3, 1, 0],
                                       [3, 0, 1],
                                       [4, 0, 0],
                                       [4, 1, 0],
                                       [3, 1, 1],
                                       [4, 0, 1],
                                       [4, 1, 1]],
                           "label": "",
                           "color": [255, 255, 255]}
                          ,
                          {"corners": [[6, 0, 0],
                                         [6, 1, 0],
                                         [6, 0, 1],
                                         [7, 0, 0],
                                         [7, 1, 0],
                                         [6, 1, 1],
                                         [7, 0, 1],
                                         [7, 1, 1]],
                           "label": "",
                           "color": [255, 255, 255]}])

UNIT_BOX_CORNERS = np.array([[0,0,0],[0,1,0],[0,0,1], [1,0,0], [1,1,0], [0,1,1], [1,0,1], [1,1,1]])

NUM_DIR_PNTS = 75


def get_wandb_boxes(bboxes, colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)), name=""):
    """

    Args:
        bboxes: numpy array size mx7, for [center, lwh, yaw_rad]
        yaws: numpy array size mx1
        colors: one RGB color per bounding box; thus, numpy array of size mx3
        scale_factor: scale bounding box size by this. Useful if we're down or upsampling size for visualization
        viz_offset: translate bounding box center by this.
        name: label the bounding box

    Returns:

    """

    wandb_boxes = []
    for i in range(bboxes.shape[0]):
        # create box
        corners = (UNIT_BOX_CORNERS - 0.5) * bboxes[i:i+1, 3:6]
        quat = Quaternion(axis=(0.0, 0.0, 1.0), radians=bboxes[i, 6])
        corners = (quat.rotation_matrix @ corners.T).T
        corners += bboxes[i:i+1, :3]
        corners *= scale_factor
        corners += viz_offset
        if isinstance(name, str):
            box_label = name + " %s" % i if name else ""
        elif isinstance(name, list):
            box_label = name[i]
        else:
            name = ""
        wandb_box = {'corners': corners.tolist(),
                     "label": box_label,
                     "color": [int(colors[i,0]), int(colors[i,1]), int(colors[i, 2])]}
        wandb_boxes.append(wandb_box)

    return np.array(wandb_boxes)

def get_boxdir_points(bboxes, colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)), length_scale=1.0):
    all_dir_points = np.empty((0, 3))
    all_dir_point_colors = np.empty((0, 3))

    for i in range(bboxes.shape[0]):
        quat = Quaternion(axis=(0.0, 0.0, 1.0), radians=bboxes[i, 6])
        dir_points = np.array([[1, 0, 0]]).repeat(NUM_DIR_PNTS, axis=0) * np.arange(NUM_DIR_PNTS).reshape(-1, 1) / NUM_DIR_PNTS
        dir_points *= bboxes[i, 3] / 2 * length_scale  # scale by length of object
        dir_points = (quat.rotation_matrix @ dir_points.T).T + bboxes[i:i+1, :3]
        dir_points = dir_points * scale_factor + viz_offset
        all_dir_points = np.concatenate([all_dir_points, dir_points], axis=0)
        all_dir_point_colors = np.concatenate([all_dir_point_colors, np.ones((NUM_DIR_PNTS, 3))*colors[i:i+1, :]], axis=0)

    return all_dir_points, all_dir_point_colors

def viz_bboxes_and_nocs(input_seq,
                        input_boxes,
                        input_confs,
                        pred_nocs,
                        pred_seg,
                        pred_bboxes,
                        pred_confidences,
                        gt_nocs,
                        gt_seg,
                        gt_bboxes,
                        kabsch_centers,
                        kabsch_orientations,
                        kabsch_parametrics,
                        haslabel_mask,
                        point2frameidx,
                        prediction_name="normal",
                        media_outdir=''):
    """
    Visualizes bounding boxes and TNOCS canonicalization from Caspr network. Everything is in NuScenes reference frame.
    Args:
        input_seq: tensor size TN x 4
        pred_nocs: tensor size TN x 4
        pred_seg: tensor size TN
        pred_bboxes: tensor size T x 7
        gt_nocs: tensor size TN x 4
        gt_seg: tensor size TN
        gt_bboxes: tensor size T x 7
        procrustes_transforms: list of (center, rot_mat), size T
        haslabel_mask: tensor size T; indicates which frames labels, ie valid GT values
        point2frameidx: tensor size TN

    """
    # prepare inputs
    input_seq, input_boxes, input_confs, pred_nocs, pred_seg, pred_bboxes, pred_confidences = torch_to_numpy([input_seq, input_boxes, input_confs, pred_nocs, pred_seg, pred_bboxes, pred_confidences])
    gt_nocs, gt_seg, gt_bboxes, haslabel_mask, point2frameidx  = torch_to_numpy([gt_nocs, gt_seg, gt_bboxes, haslabel_mask, point2frameidx])
    # gt_bboxes = gt_bboxes[haslabel_mask, :] if gt_nocs is not None else None
    # gt_seg = gt_seg[haslabel_mask, :] if gt_seg is not None else None
    # gt_nocs = gt_nocs[haslabel_mask, :, :] if gt_nocs is not None else None
    point_haslabel = haslabel_mask[point2frameidx] if haslabel_mask is not None else None

    # get gt wandb bboxes
    if gt_bboxes is not None and gt_bboxes.shape[0] > 0:
        gt_box_colors = np.array([[0, 255, 0]]).repeat(gt_bboxes.shape[0], axis=0)
        gt_wandb_boxes = get_wandb_boxes(gt_bboxes, gt_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET, name="gt")
        gt_wandb_boxdir_points, gt_wandb_boxdir_clrs = get_boxdir_points(gt_bboxes, gt_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET)
    else:
        gt_wandb_boxes, gt_wandb_boxdir_points, gt_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # get predicted wandb bboxes
    if pred_bboxes is not None:
        pred_bboxes_withlabels = pred_bboxes[:, :]
        pred_box_colors = np.array([[255, 0, 255]]).repeat(pred_bboxes.shape[0], axis=0)
        pred_wandb_boxes = get_wandb_boxes(pred_bboxes_withlabels, pred_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET, name="pred")
        pred_wandb_boxdir_points, pred_wandb_boxdir_clrs = get_boxdir_points(pred_bboxes_withlabels, pred_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET)
    else:
        pred_wandb_boxes, pred_wandb_boxdir_points, pred_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # get input wandb bboxes
    if input_boxes is not None:
        input_bboxes_withlabels = input_boxes[:, :]
        input_box_colors = np.array([[0, 0, 255]]).repeat(input_boxes.shape[0], axis=0)
        input_wandb_boxes = get_wandb_boxes(input_bboxes_withlabels, input_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET, name="input")
        input_wandb_boxdir_points, input_wandb_boxdir_clrs = get_boxdir_points(input_bboxes_withlabels, input_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET)
    else:
        input_wandb_boxes, input_wandb_boxdir_points, input_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # unify all boxes and box-dir points
    wandb_boxes = np.array(gt_wandb_boxes.tolist() + pred_wandb_boxes.tolist() + input_wandb_boxes.tolist() + WANDB_NOCS_BOXES.tolist())
    wandb_boxdir_points = np.concatenate([gt_wandb_boxdir_points, pred_wandb_boxdir_points, input_wandb_boxdir_points], axis=0)
    wandb_boxdir_clrs = np.concatenate([gt_wandb_boxdir_clrs, pred_wandb_boxdir_clrs, input_wandb_boxdir_clrs], axis=0)

    # get input-sequence points
    viz_input_seq = input_seq[:, :3].reshape(-1, 3) * LIDAR_VIZ_DOWNSCALE + INPUT_POINTS_OFFSET
    input_seq_cmap = plt.get_cmap("viridis")
    input_seq_clrs = (input_seq[:, 3]).flatten() / np.max(input_seq[:, 3])  # todo: move to max-seq limit to know how long the sequence is...
    input_seq_clrs = input_seq_cmap(input_seq_clrs)[:, :3] * 255

    # get gt nocs-sequence points
    withgt = gt_nocs.shape[0] > 0 if gt_nocs is not None else False
    if withgt:
        if gt_bboxes is not None and gt_bboxes.shape[0] > 0:
            viz_gt_nocs = gt_nocs[:, :3] / np.max(gt_bboxes[:, 3:6])
        elif pred_bboxes is not None:
            viz_gt_nocs = gt_nocs[:, :3] / np.max(pred_bboxes[:, 3:6])
        else:
            viz_gt_nocs = gt_nocs[:, :3]
        gt_nocs_clrs = np.floor((viz_gt_nocs[:, :3] * 0.6 * 255 + 0.4 * 255)).reshape(-1, 3)
        viz_gt_nocs = viz_gt_nocs.reshape(-1, 3) + GT_NOCS_OFFSET + NOCS_OFFSET
        gt_nocs_clrs *= gt_seg.reshape((-1, 1))
    else:
        viz_gt_nocs, gt_nocs_clrs = np.zeros((0, 3)), np.zeros((0, 3))

    # get pred nocs-sequence points
    if pred_nocs is not None:
        withgtnocs = gt_nocs.shape[0] > 0 if gt_nocs is not None else False
        if withgtnocs:
            pred_nocs_labeled = pred_nocs[point_haslabel, :3]
            pred_seg_labeled = pred_seg[point_haslabel].reshape(-1, 1)
            if gt_bboxes is not None:
                viz_pred_nocs = pred_nocs_labeled[:, :3] / np.max(gt_bboxes[:, 3:6])
            elif pred_bboxes is not None:
                viz_pred_nocs = pred_nocs_labeled[:, :3] / np.max(pred_bboxes[:, 3:6])
            else:
                viz_pred_nocs = pred_nocs_labeled[:, :3]
            viz_pred_nocs = viz_pred_nocs.reshape(-1, 3)
            viz_pred_nocs = viz_pred_nocs + PRED_NOCS_OFFSET + NOCS_OFFSET
            if gt_nocs is not None and gt_nocs.shape[0] > 0:
                pred_nocs_clrs = l1_error_colors(viz_pred_nocs - PRED_NOCS_OFFSET, viz_gt_nocs - GT_NOCS_OFFSET , pred_seg_labeled, error_threshold=L1_ERROR_COLOR_THRESH)
            else:
                pred_nocs_clrs = np.floor((pred_nocs[point_haslabel, :3] * 0.6 * 255 + 0.4 * 255)).reshape(-1, 3)
                pred_nocs_clrs *= pred_seg[point_haslabel].reshape((-1, 1)) > 0.0  # because is logit form
        else:
            viz_pred_nocs, pred_nocs_clrs = np.zeros((0, 3)), np.zeros((0, 3))

        # get aggregate predicted nocs-sequence
        withgtboxes = gt_bboxes.shape[0] > 0 if gt_bboxes is not None else False
        if withgtboxes:
            viz_pred_agg_nocs = pred_nocs[:, :3] / np.max(gt_bboxes[:, 3:6])
        elif pred_bboxes is not None:
            viz_pred_agg_nocs = pred_nocs[:, :3] / np.max(pred_bboxes[:, 3:6])
        else:
            viz_pred_agg_nocs = pred_nocs[:, :3]
        viz_pred_agg_nocs = viz_pred_agg_nocs.reshape(-1, 3)
        pred_agg_nocs_clrs = np.floor((viz_pred_agg_nocs[:, :3] * 0.6*255 + 0.4*255)).reshape(-1, 3)
        viz_pred_agg_nocs = viz_pred_agg_nocs + PRED_AGG_NOCS_OFFSET + NOCS_OFFSET
        pred_agg_nocs_clrs *= pred_seg.reshape((-1, 1)) > 0.0
    else:
        viz_pred_nocs, pred_nocs_clrs = np.zeros((0, 3)), np.zeros((0, 3))
        viz_pred_agg_nocs, pred_agg_nocs_clrs = np.zeros((0, 3)), np.zeros((0, 3))

    # get coordinate frame
    coord_frame_pts, coord_frame_clrs = get_coord_frame()
    coord_frame_pts_input = (coord_frame_pts / 3.0) + INPUT_POINTS_OFFSET
    coord_frame_pts += COORD_FRAME_OFFSET

    # combine everything
    viz_pts = np.vstack([viz_input_seq, viz_gt_nocs, viz_pred_nocs, viz_pred_agg_nocs, wandb_boxdir_points, coord_frame_pts, coord_frame_pts_input])
    viz_clrs = np.vstack([input_seq_clrs, gt_nocs_clrs, pred_nocs_clrs, pred_agg_nocs_clrs, wandb_boxdir_clrs, coord_frame_clrs, coord_frame_clrs])
    wandb_pc = np.hstack([viz_pts, viz_clrs])

    # log to wandb api
    if prediction_name == "normal":
        wandb.log({"Predictions": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})
        wandb.log({"Inputs": wandb.Object3D({"type": "lidar/beta", "points": np.hstack([viz_input_seq, input_seq_clrs]), "boxes": np.array([])})})
    else:
        wandb.log({"Tracking-Predictions": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})
        wandb.log({"Tracking-Inputs": wandb.Object3D({"type": "lidar/beta", "points": np.hstack([viz_input_seq, input_seq_clrs]), "boxes": np.array([])})})

    # save locally
    if media_outdir:
        wandb_step = wandb.run._step
        prediction_name = "Predictions" if prediction_name == "normal" else "Tracking-Predictions"
        local_filename = os.path.join(media_outdir, prediction_name + "_wandbstep_%s.npz" % wandb_step)
        np.savez(local_filename, points=input_seq, point2frame=point2frameidx, predicted_bboxes=pred_bboxes, predicted_confidences=pred_confidences,
                 predicted_segmentation=pred_seg, original_bboxes=input_boxes, original_confidences=input_confs, haslabel_mask=haslabel_mask,
                 gt_bboxes=gt_bboxes, gt_segmentation=gt_seg)

    # ============== Visualize Procrustes =============
    if kabsch_centers is None or pred_seg is None or pred_nocs is None:
        return

    # get procrustes info
    proc_transforms = [(kabsch_centers[i].to('cpu').data.numpy(), kabsch_orientations[i].to('cpu').data.numpy()) for i in range(kabsch_centers.size(0))]
    proc_points, proc_bboxes, proc_timestamps = procrustes_transform_points(proc_transforms, pred_nocs, pred_bboxes, haslabel_mask, pred_seg, point2frameidx)
    # if proc_points.shape[0] != input_seq[:, :, :3].reshape(-1, 3).shape[0]:
    #     return

    # proc_points_clrs = l1_error_colors(proc_points, input_seq[haslabel_mask, :, :3].reshape(-1, 3), pred_seg_labeled, error_threshold=L1_ERROR_COLOR_THRESH)
    proc_points_clrs = np.array([[255, 100, 255]]) * (0.35 + proc_timestamps).reshape(-1, 1)

    proc_points = proc_points * LIDAR_VIZ_DOWNSCALE + PROC_POINTS_OFFSET
    proc_input_points = input_seq[:, :3].reshape(-1, 3) * LIDAR_VIZ_DOWNSCALE + PROC_POINTS_OFFSET
    proc_input_points_clrs = np.array([[100, 255, 100]]) * input_seq[:, 3].reshape(-1, 1)

    # get procrustes boxes
    proc_box_colors = np.array([[70, 130, 255]]).repeat(pred_bboxes.shape[0], axis=0)
    proc_wandb_boxes = get_wandb_boxes(proc_bboxes, proc_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET, name="pred")
    proc_wandb_boxdir_points, proc_wandb_boxdir_clrs = get_boxdir_points(proc_bboxes, proc_box_colors, scale_factor=LIDAR_VIZ_DOWNSCALE, viz_offset=INPUT_POINTS_OFFSET)

    # combine 2 procrustes visualizations
    viz_proc = np.vstack([proc_points, proc_input_points, viz_input_seq, proc_wandb_boxdir_points])
    proc_clrs = np.vstack([proc_points_clrs, proc_input_points_clrs, input_seq_clrs, proc_wandb_boxdir_clrs])
    wandb_proc = np.hstack([viz_proc, proc_clrs])
    wandb_proc_boxes = np.array(gt_wandb_boxes.tolist() + proc_wandb_boxes.tolist())

    # log procrustes to wandb api
    wandb.log({"Procrustes": wandb.Object3D({"type": "lidar/beta", "points": wandb_proc, "boxes": wandb_proc_boxes})})


def sample_and_viz_bboxes_and_nocs(tpointnet_encoder,
                                   viz_loader,
                                   device,
                                   max_pnts_per_batch,
                                   nsamples=5,
                                   parallel=False,
                                   outdir=''):
    if outdir:
        media_outdir = os.path.join(outdir, "media")
        if not os.path.exists(media_outdir):
            os.makedirs(media_outdir)
    else:
        media_outdir = ''

    tpointnet_encoder.eval()
    i, loader_state = 0, "success"
    while loader_state == "success":
        try:
            data = list(viz_loader)
        except RuntimeError:  # due to possible pytorch multiprocesssing deadlock
            dataset = viz_loader.dataset
            num_workers = viz_loader.num_workers
            dataset.next()
            del viz_loader
            gc.collect()
            viz_loader = DataListLoader(dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers>0)
            continue

        # check if we've run through the whole dataset
        if len(data) == 0:
            print("Out of visualizing samples...")
            break

        sample_frameidx = np.random.randint(low=0, high=len(data), size=1).item()
        # Load and format data
        inputs = [data[j][0].to(device) for j in range(len(data))]
        gt = [inputs[j].labels for j in range(len(inputs))]
        local2scene = [inputs[j].reference_frame for j in range(len(inputs))]
        for j in range(len(inputs)):
            del inputs[j].labels
            del inputs[j].reference_frame

        inputs_split = split_inputs(inputs, max_pnts_per_batch)
        predictions = {}
        for inputs_subset in inputs_split:
            if parallel:
                predictions_subset = tpointnet_encoder(inputs_subset)  # caspr_to_scene_dict
                predictions_subset['bbox_regression'] = LidarBoundingBoxes(None, predictions_subset['bbox_regression'], size_anchors=tpointnet_encoder.module.size_clusters.to(device).float())
            else:
                batch_inputs = Batch.from_data_list(inputs_subset)
                predictions_subset = tpointnet_encoder(batch_inputs)
            update_dict(predictions, predictions_subset)

        # format data for visualization
        inputs, point2frameidx, gt, local2scene = format_pytorchgeo_data(inputs, gt, local2scene)
        frame2batchidx, _ = group_first_k_values(inputs.batch, batch=point2frameidx, k=1)
        frame2batchidx = frame2batchidx.flatten()

        # extract relevant info from predictions
        point2seqidx = frame2batchidx[point2frameidx]
        pred_tnocs = predictions['bbox_regression'].box_canonicalize(inputs.x[:, :3], batch=point2frameidx, scale=False)
        pred_tnocs = pred_tnocs[point2seqidx == sample_frameidx]
        pred_seg = predictions['segmentation'][point2seqidx == sample_frameidx]
        pred_bboxes = predictions['bbox_regression'].box[frame2batchidx == sample_frameidx]
        pred_confidences = F.sigmoid(predictions['classification']).flatten()
        _, seq_point2frameidx = torch.unique(point2frameidx[point2seqidx == sample_frameidx], return_inverse=True)

        # extract relevant info from gt
        gt_point2seqidx = gt['haslabel_mask'][point2frameidx]
        gt_point2seqidx = point2seqidx[gt_point2seqidx]
        gt_frame2batchidx = frame2batchidx[gt['haslabel_mask']]
        gt_tnocs = gt.tnocs[gt_point2seqidx == sample_frameidx]
        gt_seg = gt.segmentation[gt_point2seqidx == sample_frameidx]
        gt_boxes = gt.boxes.box[gt_frame2batchidx == sample_frameidx]
        frames_mask = gt.haslabel_mask[frame2batchidx == sample_frameidx]

        # format input seq
        input_seq = inputs.x[point2seqidx == sample_frameidx]

        input_bboxes = LidarBoundingBoxes.concatenate(inputs.boxes).to(inputs.x)
        input_bboxes = input_bboxes.box[frame2batchidx == sample_frameidx]
        input_confidences = inputs.confidences[frame2batchidx == sample_frameidx]

        # extract procrustes info
        # kabsch_centers = predictions['kabsch_centers'][frame2batchidx == sample_frameidx]
        # kabsch_orientations = predictions['kabsch_orientations'][frame2batchidx == sample_frameidx]
        # kabsch_parametrics = predictions['parametrics'][sample_frameidx]
        kabsch_centers = None
        kabsch_orientations = None
        kabsch_parametrics = None

        # visualize
        viz_bboxes_and_nocs(input_seq, input_bboxes, input_confidences, pred_tnocs, pred_seg, pred_bboxes, pred_confidences, gt_tnocs, gt_seg, gt_boxes,
                            kabsch_centers, kabsch_orientations, kabsch_parametrics, frames_mask, seq_point2frameidx, media_outdir=media_outdir)

        if i >= nsamples:
            break

        # update loader
        loader_state = viz_loader.dataset.next()
        i += 1


def sample_and_viz_bboxes_and_nocs_cherrypick(tpointnet_encoder,
                                   viz_loader,
                                   device,
                                   max_pnts_per_batch,
                                   nsamples=5,
                                   parallel=False,
                                   outdir=''):
    if outdir:
        media_outdir = os.path.join(outdir, "media")
        if not os.path.exists(media_outdir):
            os.makedirs(media_outdir)
    else:
        media_outdir = ''

    tpointnet_encoder.eval()
    i, loader_state = 0, "success"
    while loader_state == "success":
        try:
            data = list(viz_loader)
        except RuntimeError:  # due to possible pytorch multiprocesssing deadlock
            dataset = viz_loader.dataset
            num_workers = viz_loader.num_workers
            dataset.next()
            del viz_loader
            gc.collect()
            viz_loader = DataListLoader(dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers>0)
            continue

        # check if we've run through the whole dataset
        if len(data) == 0:
            print("Out of visualizing samples...")
            break

        # Load and format data
        inputs = [data[j][0].to(device) for j in range(len(data))]
        gt = [inputs[j].labels for j in range(len(inputs))]
        local2scene = [inputs[j].reference_frame for j in range(len(inputs))]
        for j in range(len(inputs)):
            del inputs[j].labels
            del inputs[j].reference_frame

        inputs_split = split_inputs(inputs, max_pnts_per_batch)
        predictions = {}
        for inputs_subset in inputs_split:
            if parallel:
                predictions_subset = tpointnet_encoder(inputs_subset)  # caspr_to_scene_dict
                predictions_subset['bbox_regression'] = LidarBoundingBoxes(None, predictions_subset['bbox_regression'], size_anchors=tpointnet_encoder.module.size_clusters.to(device).float())
            else:
                batch_inputs = Batch.from_data_list(inputs_subset)
                predictions_subset = tpointnet_encoder(batch_inputs)
            update_dict(predictions, predictions_subset)

        # format data for visualization
        inputs, point2frameidx, gt, local2scene = format_pytorchgeo_data(inputs, gt, local2scene)
        frame2batchidx, _ = group_first_k_values(inputs.batch, batch=point2frameidx, k=1)
        frame2batchidx = frame2batchidx.flatten()

        # get losses
        tpointnet_lossfunc = tpointnet_encoder.loss if not parallel else tpointnet_encoder.module.loss
        tnet_losses, tnet_metrics, success = tpointnet_lossfunc(predictions, gt, point2frameidx)
        origcrop_metrics = compute_origcrop_metrics(tpointnet_encoder, parallel, gt, local2scene, point2frameidx)

        # extract relevant info from predictions
        # sample_frameidx = np.random.randint(low=0, high=len(data), size=1).item()
        for sample_frameidx in range(len(data)):
            # check GT errors
            gt_frame2batchidx = frame2batchidx[gt['haslabel_mask']]
            our_box_center_err, our_box_yaw_err = tnet_metrics["center-l1"][gt_frame2batchidx==sample_frameidx], tnet_metrics["yaw-l1"][gt_frame2batchidx==sample_frameidx]
            orig_box_center_err, orig_box_yaw_err = origcrop_metrics["inputcrop-center-l1"][gt_frame2batchidx==sample_frameidx], origcrop_metrics["inputcrop-yaw-l1"][gt_frame2batchidx==sample_frameidx]
            if our_box_center_err.size(0) <= 1:
                continue

            our_box_center_err, our_box_yaw_err = torch.mean(our_box_center_err), torch.mean(our_box_yaw_err)
            orig_box_center_err, orig_box_yaw_err = torch.mean(orig_box_center_err), torch.mean(orig_box_yaw_err)

            if our_box_center_err > 0.4 * orig_box_center_err:
                continue
            # elif our_box_yaw_err > 0.35 * orig_box_yaw_err:
            #     continue

            # "inputcrop-yaw-l1": yaw_err,
            #                          "inputcrop-scale-iou": scale_iou_err, "inputcrop-center-l1": center_err,

            point2seqidx = frame2batchidx[point2frameidx]
            pred_tnocs = predictions['bbox_regression'].box_canonicalize(inputs.x[:, :3], batch=point2frameidx, scale=False)
            pred_tnocs = pred_tnocs[point2seqidx == sample_frameidx]
            pred_seg = predictions['segmentation'][point2seqidx == sample_frameidx]
            pred_bboxes = predictions['bbox_regression'].box[frame2batchidx == sample_frameidx]
            # if torch.mean(pred_bboxes[:, 7:9]) < 0.15:
            #     continue
            pred_confidences = F.sigmoid(predictions['classification']).flatten()
            _, seq_point2frameidx = torch.unique(point2frameidx[point2seqidx == sample_frameidx], return_inverse=True)

            # extract relevant info from gt
            gt_point2seqidx = gt['haslabel_mask'][point2frameidx]
            gt_point2seqidx = point2seqidx[gt_point2seqidx]
            gt_frame2batchidx = frame2batchidx[gt['haslabel_mask']]
            gt_tnocs = gt.tnocs[gt_point2seqidx == sample_frameidx]
            gt_seg = gt.segmentation[gt_point2seqidx == sample_frameidx]
            gt_boxes = gt.boxes.box[gt_frame2batchidx == sample_frameidx]
            frames_mask = gt.haslabel_mask[frame2batchidx == sample_frameidx]

            # format input seq
            input_seq = inputs.x[point2seqidx == sample_frameidx]

            input_bboxes = LidarBoundingBoxes.concatenate(inputs.boxes).to(inputs.x)
            input_bboxes = input_bboxes.box[frame2batchidx == sample_frameidx]
            input_confidences = inputs.confidences[frame2batchidx == sample_frameidx]

            # extract procrustes info
            # kabsch_centers = predictions['kabsch_centers'][frame2batchidx == sample_frameidx]
            # kabsch_orientations = predictions['kabsch_orientations'][frame2batchidx == sample_frameidx]
            # kabsch_parametrics = predictions['parametrics'][sample_frameidx]
            kabsch_centers = None
            kabsch_orientations = None
            kabsch_parametrics = None

            # visualize
            viz_bboxes_and_nocs(input_seq, input_bboxes, input_confidences, pred_tnocs, pred_seg, pred_bboxes, pred_confidences, gt_tnocs, gt_seg, gt_boxes,
                                kabsch_centers, kabsch_orientations, kabsch_parametrics, frames_mask, seq_point2frameidx, media_outdir=media_outdir)

        if i >= nsamples:
            break

        # update loader
        loader_state = viz_loader.dataset.next()
        i += 1



def viz_bboxes_and_nocs_from_raw_preds(predictions,
                                       inputs,
                                       point2frameidx,
                                       num_samples,
                                       media_outdir):

    for sample_frameidx in range(num_samples):
        if (sample_frameidx % 4) != 0:
            continue
        # format data for visualization
        frame2batchidx, _ = group_first_k_values(inputs.batch, batch=point2frameidx, k=1)
        frame2batchidx = frame2batchidx.flatten()

        # if torch.sum(frame2batchidx == sample_frameidx) < 25:
        #     continue

        # extract relevant info from predictions
        point2seqidx = frame2batchidx[point2frameidx]
        pred_tnocs = predictions['bbox_regression'].box_canonicalize(inputs.x[:, :3], batch=point2frameidx, scale=False)
        pred_tnocs = pred_tnocs[point2seqidx == sample_frameidx]
        pred_seg = predictions['segmentation'][point2seqidx == sample_frameidx]
        pred_bboxes = predictions['bbox_regression'].box[frame2batchidx == sample_frameidx]
        pred_confidences = F.sigmoid(predictions['classification']).flatten()

        _, seq_point2frameidx = torch.unique(point2frameidx[point2seqidx == sample_frameidx], return_inverse=True)

        # format input seq
        input_seq = inputs.x[point2seqidx == sample_frameidx]
        input_bboxes = inputs.boxes.box[frame2batchidx == sample_frameidx]
        input_confidences = inputs.confidences[frame2batchidx == sample_frameidx]

        # extract procrustes info
        kabsch_centers = None
        kabsch_orientations = None
        kabsch_parametrics = None

        # visualize
        viz_bboxes_and_nocs(input_seq, input_bboxes, input_confidences, pred_tnocs, pred_seg, pred_bboxes, pred_confidences, None, None, None,
                            kabsch_centers, kabsch_orientations, kabsch_parametrics, None, seq_point2frameidx,
                            prediction_name="tracking", media_outdir=media_outdir)




def viz_pred_boxes(pred_bboxes, gt_boxes=None, name="Predicted Boxes", pred_colors=None, gt_colors=None):
    """
    Visualizes bounding boxes and TNOCS canonicalization from Caspr network. Everything is in NuScenes reference frame.
    Args:
        pred_bboxes: tensor size M x 7

    """
    # get predicted wandb bboxes
    if pred_colors is None:
        pred_box_colors = np.array([[255, 0, 255]]).repeat(pred_bboxes.shape[0], axis=0)
    else:
        pred_box_colors = pred_colors
    pred_wandb_boxes = get_wandb_boxes(pred_bboxes, pred_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
    pred_wandb_boxdir_points, pred_wandb_boxdir_clrs = get_boxdir_points(pred_bboxes, pred_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))

    if gt_boxes is not None:
        if gt_colors is None:
            gt_box_colors = np.array([[0, 255, 0]]).repeat(gt_boxes.shape[0], axis=0)
        else:
            gt_box_colors = gt_colors
        gt_wandb_boxes = get_wandb_boxes(gt_boxes, gt_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        gt_wandb_boxdir_points, gt_wandb_boxdir_clrs = get_boxdir_points(gt_boxes, gt_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
    else:
        gt_wandb_boxes, gt_wandb_boxdir_points, gt_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # unify all boxes and box-dir points
    wandb_boxes = np.array(gt_wandb_boxes.tolist() + pred_wandb_boxes.tolist())
    wandb_boxdir_points = np.concatenate([gt_wandb_boxdir_points, pred_wandb_boxdir_points], axis=0)
    wandb_boxdir_clrs = np.concatenate([gt_wandb_boxdir_clrs, pred_wandb_boxdir_clrs], axis=0)

    # get coordinate frame
    # coord_frame_pts, coord_frame_clrs = get_coord_frame()
    # coord_frame_pts_input = (coord_frame_pts / 3.0)

    # combine everything
    viz_pts = np.vstack([wandb_boxdir_points])
    viz_clrs = np.vstack([wandb_boxdir_clrs])
    wandb_pc = np.hstack([viz_pts, viz_clrs])

    # log to wandb api
    wandb.log({name: wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})


def viz_st_detections(detections, sample_timestep, trajectory_params):
    if len(detections.enhanced_objects) > 0 and not isinstance(detections.enhanced_objects, EmptySTObjectRepresentations):
        enhanced_viz, enhanced_boxes = viz_st_detections_helper(detections.enhanced_objects, sample_timestep, trajectory_params)
    else:
        enhanced_viz, enhanced_boxes = np.zeros((0, 6)), np.array([])

    if len(detections.base_objects) > 0 and not isinstance(detections.base_objects, EmptySTObjectRepresentations):
        base_viz, base_boxes = viz_st_detections_helper(detections.base_objects, sample_timestep, trajectory_params)
    else:
        base_viz, base_boxes = np.zeros((0, 6)), np.array([])

    wandb_pc = np.vstack([enhanced_viz, base_viz])
    wandb_boxes = np.array(enhanced_boxes.tolist() + base_boxes.tolist())
    wandb.log({"Current Tracks": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})


def viz_st_detections_helper(detections, sample_timestep, trajectory_params):
    """
    Args:
        detections (MergedEnhancedAndBaseObjectRepresentations):
    Returns:

    """

    # add processed st-detections
    device = detections.device
    points = detections.points[:, :3]
    color_pallete = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255], [0, 0, 0]]
    clrs = torch.tensor(COLOR_PALLETE).to(device)[detections.point2batchidx % len(COLOR_PALLETE)]
    # if isinstance(detections, ProcessedSTObjectRepresentations):
    #     clrs = clrs * (detections.segmentation.view(-1, 1) > 0)

    # compute boxes and parametrics
    sample_timestep_tnsr = torch.tensor([sample_timestep], dtype=torch.double).to(device)
    est_boxes, parametrics = detections.estimate_boxes(sample_timestep_tnsr, **trajectory_params)
    est_boxes, parametrics = est_boxes[0], parametrics[0]

    # convert everything to numpy
    points, clrs, est_boxes, parametrics = torch_to_numpy([points, clrs, est_boxes.box, parametrics])

    # set up boxes
    box_colors = np.array(COLOR_PALLETE)[np.arange(len(detections)) % len(COLOR_PALLETE)]
    det_wandb_boxes = get_wandb_boxes(est_boxes, box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
    det_wandb_boxdir_points, det_wandb_boxdir_clrs = get_boxdir_points(est_boxes, box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))

    # set up parametrics points
    num_trajectory_samples = 200
    order = trajectory_params["interpolation_order"]
    sample_times = np.arange(num_trajectory_samples) / num_trajectory_samples
    sample_times = np.stack([sample_times**i for i in range(order+1)])
    parametric_points = (parametrics @ sample_times).transpose((0, 2, 1)).reshape(-1, 4)[:, :3]  # M x 200 x 4
    parametric_clrs = np.array(COLOR_PALLETE)[np.arange(len(detections)) % len(COLOR_PALLETE)]
    parametric_clrs = parametric_clrs.repeat(num_trajectory_samples, axis=1).reshape(-1, 3)

    # combine all boxes
    wandb_points = np.concatenate([points, det_wandb_boxdir_points, parametric_points])
    wandb_clrs = np.concatenate([clrs, det_wandb_boxdir_clrs, parametric_clrs])
    wandb_pc = np.concatenate([wandb_points, wandb_clrs], axis=1)
    return wandb_pc, det_wandb_boxes


def viz_detections_and_tracks(global_detections, tracks, assignments):
    """

    Args:
        detections (torch.tensor): global-ref-frame detection points of size (M,T,N,4)
        detection_boxes (torch.tensor): global-ref-frame detection boxes of size (M, T, 7)
        tracks (list): List of ObjectTrack; each ObjectTrack has the global-ref-frame stored points of size (T',N,4)
        assignments (dict): Dictionary mapping detection-idx to track-idx
    Returns:

    """

    color_idx = 0
    default_det_clr = [200, 200, 200]
    default_track_clr = [75, 75, 75]
    viz_pnts = []
    viz_clrs = []

    # add matched detection-tracks
    for det_idx, track_idx in assignments.items():
        if global_detections['withpoints_mask'][det_idx]:
            points_idx = torch.sum(global_detections['withpoints_mask'][:det_idx]).long()  # say this is #6
            det_pnts = global_detections['points'][points_idx].view(-1, 4)[:, :3].to('cpu').data.numpy()
            det_clrs = np.array(COLOR_PALLETE[color_idx]).reshape(1, 3).repeat(det_pnts.shape[0], axis=0)
            viz_pnts.append(det_pnts)
            viz_clrs.append(det_clrs)
        if tracks[track_idx].frames_since_points > 0:
            track_pnts = tracks[assignments[det_idx]].track_points.view(-1, 4)[:, :3].to('cpu').data.numpy()
            track_clrs = np.array(COLOR_PALLETE[color_idx]).reshape(1, 3).repeat(track_pnts.shape[0], axis=0) * 1.2
            viz_pnts.append(track_pnts)
            viz_clrs.append(track_clrs)
        color_idx = (color_idx + 1) % len(COLOR_PALLETE)

    # add unmatched detections
    for det_idx in range(global_detections['boxes'].size(0)):
        if det_idx in assignments.keys() or not global_detections['withpoints_mask'][det_idx]:
            continue
        points_idx = torch.sum(global_detections['withpoints_mask'][:det_idx]).long()  # say this is #6
        det_pnts = global_detections['points'][points_idx].view(-1, 4)[:, :3].to('cpu').data.numpy()
        det_clrs = np.array(default_det_clr).reshape(1, 3).repeat(det_pnts.shape[0], axis=0)
        viz_pnts.append(det_pnts)
        viz_clrs.append(det_clrs)

    # add unmatched tracks
    for track_idx, obj_track in enumerate(tracks):
        if track_idx in assignments.values() or obj_track.frames_since_points > 0:
            continue
        track_pnts = obj_track.track_points.view(-1, 4)[:, :3].to('cpu').data.numpy()
        track_clrs = np.array(default_track_clr).reshape(1, 3).repeat(track_pnts.shape[0], axis=0)
        viz_pnts.append(track_pnts)
        viz_clrs.append(track_clrs)

    # compute detection boxes
    DET_MATCHED_COLOR = [0, 255, 0]
    TRACK_MATCHED_COLOR = [0, 0, 255]
    DET_UNMATCHED_COLOR = [255, 255, 0]
    TRACK_UNMATCHED_COLOR = [255, 0, 255]
    NOPOINTS_COLOR = [0, 0, 0]
    NOPOINTS_MATCHED_COLOR = [255, 255, 255]

    detection_mintime_idxs = torch.argmin(global_detections['timesteps'], dim=1)
    mintime_detection_boxes = batched_index_select(global_detections['boxes'], dim=1, index=detection_mintime_idxs.view(-1, 1))
    viz_detection_boxes = mintime_detection_boxes.view(-1, 7).to('cpu').data.numpy()
    if viz_detection_boxes.shape[0] > 0:
        detection_box_colors = []
        for det_idx in range(viz_detection_boxes.shape[0]):
            if det_idx in assignments.keys() and global_detections['withpoints_mask'][det_idx]:
                detection_box_colors.append(DET_MATCHED_COLOR)
            elif det_idx in assignments.keys() and not global_detections['withpoints_mask'][det_idx]:
                detection_box_colors.append(NOPOINTS_MATCHED_COLOR)
            elif det_idx not in assignments.keys() and global_detections['withpoints_mask'][det_idx]:
                detection_box_colors.append(DET_UNMATCHED_COLOR)
            else:
                detection_box_colors.append(NOPOINTS_COLOR)
        detection_box_colors = np.array(detection_box_colors)
        det_wandb_boxes = get_wandb_boxes(viz_detection_boxes, detection_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        det_wandb_boxdir_points, det_wandb_boxdir_clrs = get_boxdir_points(viz_detection_boxes, detection_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
    else:
        det_wandb_boxes, det_wandb_boxdir_points, det_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # compute tracking boxes
    if len(tracks) > 0:
        viz_track_boxes, track_box_colors = [], []
        for track_idx, track in enumerate(tracks):
            box = track.boxes[track.current_idx]
            viz_track_boxes.append(box.to('cpu').data.numpy().tolist())
            if track_idx in assignments.values() and track.with_current_points:
                track_box_colors.append(TRACK_MATCHED_COLOR)
            elif track_idx in assignments.values() and not track.with_current_points:
                track_box_colors.append(NOPOINTS_MATCHED_COLOR)
            elif track_idx not in assignments.values() and track.with_current_points:
                track_box_colors.append(TRACK_UNMATCHED_COLOR)
            else:
                track_box_colors.append(NOPOINTS_COLOR)
        viz_track_boxes = np.array(viz_track_boxes)
        track_box_colors = np.array(track_box_colors)
        track_wandb_boxes = get_wandb_boxes(viz_track_boxes, track_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        track_wandb_boxdir_points, track_wandb_boxdir_clrs = get_boxdir_points(viz_track_boxes, track_box_colors, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
    else:
        track_wandb_boxes, track_wandb_boxdir_points, track_wandb_boxdir_clrs = np.array([]), np.zeros((0, 3)), np.zeros((0, 3))

    # combine all boxes
    wandb_boxes = np.array(det_wandb_boxes.tolist() + track_wandb_boxes.tolist())
    wandb_boxdir_points = np.concatenate([det_wandb_boxdir_points, track_wandb_boxdir_points], axis=0)
    wandb_boxdir_clrs = np.concatenate([det_wandb_boxdir_clrs, track_wandb_boxdir_clrs], axis=0)
    viz_pnts.append(wandb_boxdir_points)
    viz_clrs.append(wandb_boxdir_clrs)

    # put everything together
    viz_pnts = np.concatenate(viz_pnts, axis=0)
    viz_clrs = np.concatenate(viz_clrs, axis=0)
    wandb_pc = np.concatenate([viz_pnts, viz_clrs], axis=1)
    wandb.log({"Current Tracks": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})



def l1_error_colors(cloud1, cloud2, segmentation_mask, error_threshold):
    dists = np.linalg.norm(cloud1 - cloud2, axis=1)
    blues = np.floor((error_threshold - dists) * (1 / error_threshold) * 255) * (segmentation_mask.flatten() > 0)
    reds = np.floor((dists) * (1 / error_threshold) * 255) * (segmentation_mask.flatten() > 0)
    colors = np.stack([reds, np.zeros(reds.shape), blues], axis=1)  # Now is T x N x 3
    colors = np.where(colors > 255, 255, colors)
    return colors


def get_coord_frame():
    x_frame = np.array([[1,0,0]]).repeat(40, axis=0) * np.arange(40).reshape(40, 1) / 40
    x_color = np.array([[1,0,0]]).repeat(40, axis=0) * 255
    y_frame = np.array([[0,1,0]]).repeat(40, axis=0) * np.arange(40).reshape(40, 1) / 40
    y_color = np.array([[0,1,0]]).repeat(40, axis=0) * 255
    z_frame = np.array([[0,0,1]]).repeat(40, axis=0) * np.arange(40).reshape(40, 1) / 40
    z_color = np.array([[0,0,1]]).repeat(40, axis=0) * 255
    coord_frame = np.vstack((x_frame, y_frame, z_frame))
    coord_colors = np.vstack((x_color, y_color, z_color))
    return coord_frame, coord_colors


def compute_origcrop_metrics(tpointnet_encoder, is_parallel, gt, local2scene, point2frameidx):
    # get proposal-boxes info
    proposed_boxes = local2scene.proposal_boxes
    proposed_boxes.all_size_residuals = proposed_boxes.compute_size_residuals()
    orig_nocs_points = local2scene.proposal_tnocs

    # compute bounding box yaw and center error
    bbox_regressor = tpointnet_encoder.bbox_regressor if not is_parallel else tpointnet_encoder.module.bbox_regressor
    _, _, yaw_err, scale_iou_err, center_err, velocity_err = bbox_regressor.loss(proposed_boxes, gt.boxes, gt.haslabel_mask)

    # Compute NOCS L1 Error
    nocs_loss_calculator = tpointnet_encoder.classification_and_segmentation.nocs_loss if not is_parallel else tpointnet_encoder.module.classification_and_segmentation.nocs_loss
    _, per_seq_loss_metric = nocs_loss_calculator(orig_nocs_points, gt.tnocs, gt.haslabel_mask, gt.segmentation, point2frameidx)

    # Compute precision - recall of classification
    class_loss_calculator = tpointnet_encoder.classification_and_segmentation.classification_loss if not is_parallel else tpointnet_encoder.module.classification_and_segmentation.classification_loss
    _, precision, recall = class_loss_calculator(torch.logit(local2scene.confidences.unsqueeze(-1)), gt.classification, center_err, gt.haslabel_mask, gt.keyframe_mask)

    orig_crop_metrics = {"inputcrop-tnocs-l1": per_seq_loss_metric, "inputcrop-yaw-l1": yaw_err,
                         "inputcrop-scale-iou": scale_iou_err, "inputcrop-center-l1": center_err,
                         "inputcrop-velocity-l1": velocity_err,
                         "inputcrop-precision": precision, "inputcrop-recall": recall}
    return orig_crop_metrics