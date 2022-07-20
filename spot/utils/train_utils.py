import time

import matplotlib
matplotlib.use('Agg')

import numpy as np

import torch
import gc
from spot.data.box3d import LidarBoundingBoxes
from torch_geometric.data import Batch
from spot.io.run_utils import split_inputs, update_dict
from torch_geometric.data.data import Data as GeometricData

from spot.io.logging_utils import log_batch_stats, log_epoch_stats, log
GC_FLAG = False
from torch_geometric.nn.glob.glob import global_add_pool
from spot.ops.pc_util import group_first_k_values, merge_point2frame_across_batches
from torch_geometric.loader import DataListLoader

MAX_AUTOREGRESS_PROB = 0.6
MAX_ROLLOUT = 4


def run_one_epoch(tpointnet_encoder,
                  data_loader,
                  device,
                  optimizer,
                  epoch,
                  model_config,
                  log_out,
                  mode='train',
                  max_pnts_per_batch=10000,
                  print_stats_every=10,
                  autoregressive_training=False,
                  dataset=None,
                  num_workers=8):
    '''
    Runs through the given dataset once to train or test the model depending on the mode given.
    '''

    if mode not in ['train', 'val', 'test']:
        print('Most must be train or test!')
        exit()

    is_parallel = isinstance(tpointnet_encoder, torch.nn.DataParallel)
    batch_losses, batch_metrics, epoch_losses, epoch_metrics = {}, {}, {}, {}

    if mode == 'train':
        tpointnet_encoder = tpointnet_encoder.train()
    else:
        tpointnet_encoder = tpointnet_encoder.eval()

    t_init = time.time()

    num_seqs_with_labels = 0
    num_seqs_total = 0
    size_bin_freqs = torch.tensor([0, 0, 0]).to(device)
    i, loader_state = 0, "success"
    dataloader_iter = dataset.cur_idx
    while True:
        if (i % 400) == 1:
            del data_loader
            gc.collect()
            data_loader = DataListLoader(dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers>0)

        # Load and format data
        try:
            data = list(data_loader)
        except RuntimeError as e:
            print("60 second timeout occured in dataloader. Skipping this batch.")
            failure_exit(mode, e)
            del data_loader
            gc.collect()
            dataset.cur_idx = dataloader_iter + 1
            data_loader = DataListLoader(dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers>0)
            if mode == "val": continue
            if 'total' not in epoch_losses: epoch_losses['total'] = np.inf
            return False, i, np.mean(epoch_losses['total']), epoch_metrics


        # keep track of main-thread state if appropriate
        if num_workers > 0:
            dataset.next()

        # if we've reached the end of the dataset, then exit
        if len(data) == 0:
            break

        # format data appropriately
        inputs = [data[j][0].to(device) for j in range(len(data))]
        gt = [inputs[j].labels for j in range(len(inputs))]
        local2scene = [inputs[j].reference_frame for j in range(len(inputs))]
        inputs_orig = [inputs[j].clone() for j in range(len(inputs))]
        for j in range(len(inputs)):
            del inputs[j].labels
            del inputs[j].reference_frame

        # Configure iteration params
        if mode == 'train':
            optimizer.zero_grad()

        # todo: make this more robust, but for now, ignore batches with all FP
        gt_test = Batch.from_data_list(gt)
        gt_test.boxes = LidarBoundingBoxes.concatenate(gt_test.boxes)
        if gt_test.tnocs.size(0) == 0 or gt_test.boxes.size(0) == 0:
            print("Skipping batch of all false positives...")
            continue

        # allow auto-regressive estimates
        if autoregressive_training:
            autoregress_prob = min(max(0, (epoch)) / 8, MAX_AUTOREGRESS_PROB)
        else:
            autoregress_prob = 0

        # Forward TPointNet
        inputs_split = split_inputs(inputs, max_pnts_per_batch)
        if len(inputs_split) > 1:
            log(log_out, "Note. In this batch, our point-threshold caused us to run on %s splits." % len(inputs_split))
        predictions = {}
        for inputs_subset in inputs_split:

            try:
                # we may auto-regressively train on our predicted bounding boxes
                num_autoregressions = 0
                cum_center_offsets = torch.zeros(len(inputs_subset), 3).to(device)
                while num_autoregressions < MAX_ROLLOUT:
                    inputs_subset_cp = [GeometricData(x=one_input.x.clone(), boxes=one_input.boxes.clone(),
                                                     point2frameidx=one_input.point2frameidx.clone(),
                                                     confidences=one_input.confidences.clone()) for one_input in inputs_subset]
                    if is_parallel:
                        predictions_subset = tpointnet_encoder(inputs_subset_cp)  # caspr_to_scene_dict
                        all_size_residuals = predictions_subset['all_size_residuals']
                        predictions_subset['bbox_regression'] = LidarBoundingBoxes(None, predictions_subset['bbox_regression'], size_anchors=tpointnet_encoder.module.size_clusters.to(device).float(), all_size_residuals=all_size_residuals)
                    else:
                        batch_inputs = Batch.from_data_list(inputs_subset_cp)
                        predictions_subset = tpointnet_encoder(batch_inputs)

                    # we will need this info for auto-regressive computation
                    batch_inputs = Batch.from_data_list(inputs_subset)
                    point2frameidx = merge_point2frame_across_batches(batch_inputs.point2frameidx, batch_inputs.batch)
                    frame2batchidx, _ = group_first_k_values(batch_inputs.batch, batch=point2frameidx, k=1)
                    frame2batchidx = frame2batchidx.flatten()

                    # we may want to autoregressively update the bounding boxes
                    if torch.rand(1) < autoregress_prob:
                        box_autoregressive_update = predictions_subset['bbox_regression'].detach().clone()
                        for j, sample in enumerate(inputs_subset):
                            sample.boxes = box_autoregressive_update[frame2batchidx == j].clone()
                            center_offset = sample.boxes.center[0:1].clone()  # todo: this is inf??
                            sample.boxes = sample.boxes.translate(-center_offset, torch.zeros(sample.boxes.size(0), dtype=torch.long).to(device))
                            if torch.rand(1) < 0.5:  # half the time, re-center reference frame to first predicted box.
                                sample.x[:, :3] -= center_offset
                                cum_center_offsets[j:j + 1] += center_offset
                            # note: we don't autoregressively update confidences!
                            if torch.any(torch.isnan(sample.boxes.box)) or torch.any(torch.isnan(sample.boxes.box_enc)):
                                print("NAN in box autoregressive update!")
                                print(predictions_subset['bbox_regression'].box[frame2batchidx == j])
                                print(j)
                                print(num_autoregressions)
                                print(torch.sum(frame2batchidx == j))
                                print(center_offset)
                                print(">>")
                                print(frame2batchidx)
                                print("...")
                                print(sample.boxes.box)
                                print("////")
                                print(sample.boxes.box_enc)

                        num_autoregressions += 1

                    # end autoregressive predictions
                    else:
                        # return points and boxes to original reference frame (aligned with GT)
                        predictions_subset['bbox_regression'] = predictions_subset['bbox_regression'].translate(cum_center_offsets, batch=frame2batchidx)
                        break
                # finally, record these predictions
                update_dict(predictions, predictions_subset)

            except (RuntimeError, ValueError, IndexError) as e:
                failure_exit(mode, e)
                if 'total' not in epoch_losses: epoch_losses['total'] = np.inf
                return False, i, np.mean(epoch_losses['total']), epoch_metrics

        # Compute Losses and Metrics
        inputs, point2frameidx, gt, local2scene = format_pytorchgeo_data(inputs_orig, gt, local2scene)
        tpointnet_lossfunc = tpointnet_encoder.loss if not is_parallel else tpointnet_encoder.module.loss
        tnet_losses, tnet_metrics, success = tpointnet_lossfunc(predictions, gt, point2frameidx)
        origcrop_metrics = compute_origcrop_metrics(tpointnet_encoder, is_parallel, gt, local2scene, point2frameidx)

        # DEBUGGING:
        frame2batchidx, _ = group_first_k_values(inputs.batch, batch=point2frameidx, k=1)
        frame2batchidx = frame2batchidx.flatten()
        seqs_with_labels = global_add_pool(gt.haslabel_mask.float(), batch=frame2batchidx) > 0
        num_seqs_total += seqs_with_labels.size(0)
        num_seqs_with_labels += torch.sum(seqs_with_labels)
        unique_idxs, gt_box_count = torch.unique(gt.boxes.box_size_anchor_idxs, return_counts=True)
        size_bin_freqs[unique_idxs] += gt_box_count

        # Accumulate TPointNet Losses
        loss = torch.zeros(1, requires_grad=True).to(device)
        for loss_type, loss_term in tnet_losses.items():
            loss += torch.mean(loss_term)

        # Backprop if applicable
        if mode == 'train':
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                print("Optimizer Backward Error.")
                failure_exit(mode, e)
                if 'total' not in epoch_losses: epoch_losses['total'] = np.inf
                optimizer.zero_grad()
                return False, i, np.mean(epoch_losses['total']), epoch_metrics

        # Keep track of loss values
        losses = {**tnet_losses, **{"total": loss}}
        metrics = {**tnet_metrics, **origcrop_metrics}
        minibatch_idx = i % print_stats_every
        batch_losses = update_stats(batch_losses, losses)
        batch_metrics = update_stats(batch_metrics, metrics)
        epoch_losses = update_stats(epoch_losses, losses)
        epoch_metrics = update_stats(epoch_metrics, metrics)

        # log stats and print update
        if minibatch_idx == (print_stats_every - 1):
            time_elapsed = time.time() - t_init
            t_init = time.time()
            print("Time Elapsed: %s" % time_elapsed)
            log_batch_stats(batch_losses, batch_metrics, epoch, dataloader_iter, len(data_loader.dataset) // data_loader.dataset.batch_size, mode, log_out)
            batch_losses, batch_metrics = {}, {}

        # increment batch index
        i += 1
        dataloader_iter += 1

    print("@@@@@@@@@@@@@@@@@@@@@@@")
    print("Num Seqs total")
    print(num_seqs_total)
    print("Num Seqs with labels:")
    print(num_seqs_with_labels)
    print("Size-Bin Frequencies:")
    print(size_bin_freqs)


    log_epoch_stats(epoch_losses, epoch_metrics, epoch, mode, log_out)
    torch.cuda.empty_cache()
    if 'total' not in epoch_losses:
        epoch_losses['total'] = np.inf
    return True, 0, np.mean(epoch_losses['total']), epoch_metrics

def failure_exit(mode, error_msg):
    print("==================")
    print("Error occurred. In Mode: ")
    print(mode)
    print(error_msg)
    print("==================")
    torch.cuda.empty_cache()
    gc.collect()

def update_stats(recurring_stats, cur_stats):
    cur_stats = {k: val.to('cpu').data.numpy() for k, val in cur_stats.items()}
    for k, val in cur_stats.items():
        if k not in recurring_stats:
            recurring_stats[k] = [val]
        else:
            recurring_stats[k].append(val)
    return recurring_stats


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

def format_pytorchgeo_data(inputs, gt, local2scene):
    inputs = Batch.from_data_list(inputs)
    point2frameidx = merge_point2frame_across_batches(inputs.point2frameidx, inputs.batch)
    gt = Batch.from_data_list(gt)
    gt.boxes = LidarBoundingBoxes.concatenate(gt.boxes).to(inputs.x)
    local2scene = Batch.from_data_list(local2scene)
    local2scene.proposal_boxes = LidarBoundingBoxes.concatenate(local2scene.proposal_boxes).to(inputs.x)

    return inputs, point2frameidx, gt, local2scene