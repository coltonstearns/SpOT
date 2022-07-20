import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.glob.glob import global_max_pool, global_mean_pool, global_add_pool
import math


class ClassificationAndSegmentation(nn.Module):

    def __init__(self, feat_dim, group_norm):
        super(ClassificationAndSegmentation, self).__init__()
        self.feat_dim = feat_dim

        # extra processing
        self.conv1 = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.bn1 = nn.GroupNorm(16, self.feat_dim) if group_norm else nn.BatchNorm1d(self.feat_dim)
        self.bn2 = nn.GroupNorm(16, self.feat_dim) if group_norm else nn.BatchNorm1d(self.feat_dim)

        # predict binary object mask afterward
        self.segment_pred = torch.nn.Conv1d(self.feat_dim, 1, 1)

        # CNF feature
        self.cnf_pred = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)

        # Classification
        self.class_layer_1 = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.class_layer_2 = torch.nn.Linear(self.feat_dim, 1)

        # loss functions
        self.tnocs_loss_func = torch.nn.L1Loss(reduce=False)
        self.seg_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.class_loss_func = torch.nn.BCEWithLogitsLoss(reduce=False)

    def forward(self, x, point2batchidx, point2frameidx):
        """
        x: Computed per-point features of the full input sequence; size (feat_dim, BTN)
        """
        # process to get latent features output
        nfeats, BTN = x.size()
        x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))  # size 1, feat_dim, B*T*N

        # predict instance segmentation
        segmentation_raw = self.segment_pred(F.relu(x))  # no sigmoid because keep in logit form
        segmentation = segmentation_raw.transpose(2, 1).view(BTN, 1)

        # process into CNF feature
        global_feat = self.cnf_pred(x)  # feat_size x BTN
        global_feat *= F.sigmoid(segmentation_raw)
        global_feat = global_feat.squeeze(0).transpose(1, 0).contiguous()
        global_feat = global_max_pool(global_feat, batch=point2batchidx)  # B x feat_size

        # perform per-frame binary classification
        classification_feat = self.class_layer_1(x)
        classification_feat *= F.sigmoid(segmentation_raw)
        classification_feat = classification_feat.squeeze(0).transpose(1, 0).contiguous()
        classification_feat = global_max_pool(classification_feat, batch=point2frameidx)
        classification = self.class_layer_2(classification_feat)

        return segmentation, global_feat, classification

    def segmentation_loss(self, preds, gt, haslabel_indicator, point2frameidx):
        # if we're just performing regression, simply loss on all points
        labeled_preds, point2frameidx = self._filter2labeled(preds, haslabel_indicator, point2frameidx)

        # compute segmentation loss
        loss = self.seg_loss_func(labeled_preds, gt.float())
        loss = global_mean_pool(loss, batch=point2frameidx).mean()

        # compute precision-recall metrics
        predicted_segmentation = labeled_preds > 0.0  # because is logit form
        true_positives = global_add_pool(torch.logical_and(predicted_segmentation, gt).float(), batch=point2frameidx)
        union_positives = global_add_pool(torch.logical_or(predicted_segmentation, gt).float(), batch=point2frameidx)
        IoUs = (true_positives / union_positives)  # size BT

        # remove frames with 0 GT points
        valid_mask = torch.isfinite(IoUs).flatten()
        IoUs = IoUs[valid_mask]

        # Get average IOUs
        frame_mIoU_at_70 = (IoUs > 0.7).float()

        return loss, IoUs, frame_mIoU_at_70

    def classification_loss(self, preds, gt, center_errs, haslabel_mask, is_keyframe_mask, temperature=0.7, worst_ratio=0.5):
        # compute loss
        losses = self._classification_loss(preds.flatten(), gt.float(), center_errs, haslabel_mask, is_keyframe_mask, alpha=temperature, worst_ratio=worst_ratio)
        avg_prec, avg_rec = self._classification_err(preds, gt)
        return losses, avg_prec, avg_rec

    def _classification_loss(self, preds, gt, center_errs, haslabel_mask, is_keyframe_mask, alpha=0.7, worst_ratio=0.5):
        # for true-positives, estimate continuous confidence
        tp_target_confs = 2**(-alpha * center_errs)
        tp_loss = self.class_loss_func(preds[haslabel_mask], tp_target_confs)

        # for false-positives, attempts to predict confidence of 0
        fp_preds = preds[~haslabel_mask & is_keyframe_mask]
        # assert torch.all(gt[~haslabel_mask & is_keyframe_mask] == 0)
        fp_loss = self.class_loss_func(fp_preds, gt[~haslabel_mask & is_keyframe_mask])

        # format losses in order
        worst_tp_idxs = torch.argsort(tp_loss, descending=True)[:math.ceil(tp_loss.size(0) * worst_ratio)]
        worst_fp_idxs = torch.argsort(fp_loss, descending=True)[:math.ceil(fp_loss.size(0) * worst_ratio)]
        worst_losses = torch.cat([tp_loss[worst_tp_idxs], fp_loss[worst_fp_idxs]])
        return worst_losses.view(-1, 1)

    def _classification_err(self, preds, gt):
        # Calculate rough precision and recall.
        num_positives = torch.sum(gt)
        thresholds = torch.linspace(0, 1.0, 200).view(1, -1).to(preds)  # 10 thresholds
        positive_preds = F.sigmoid(preds).view(-1, 1) > thresholds  # num-preds x 10 array
        tp = torch.sum(positive_preds & gt.bool().view(-1, 1), dim=0)
        fp = torch.sum(positive_preds & (~gt.bool().view(-1, 1)), dim=0)

        # get indices with recalls closes to
        threshold_recalls = tp / num_positives  # length 200 array sortd in decending order
        sampled_recalls = torch.linspace(0.5, 1.0, 20).view(1, -1).to(preds)
        rec_thresh_idxs = torch.argmin(torch.abs(threshold_recalls.view(-1, 1) - sampled_recalls), dim=0)
        bad_idxs = (threshold_recalls[rec_thresh_idxs] < 0.1) | ~(torch.isfinite(threshold_recalls[rec_thresh_idxs]))
        rec_thresh_idxs = rec_thresh_idxs[~bad_idxs]
        if rec_thresh_idxs.size(0) == 0:
            avg_prec, avg_rec = torch.zeros(1).to(preds), torch.zeros(1).to(preds)
        else:
            # compute precisions
            precisions = tp / (tp + fp)
            precisions = precisions[rec_thresh_idxs]
            avg_prec = torch.mean(precisions)
            avg_rec = torch.mean(threshold_recalls[rec_thresh_idxs])

        return avg_prec, avg_rec

    def nocs_loss(self, preds, gt, haslabel_indicator, gt_foreground_mask, point2frameidx):
        '''
        Computes the loss for TNOCS regression given the outputs of the network compared to GT
        TNOCS values. Returns unreduces loss values (per-point)
        preds: BTN x 4 dimensional tensor of predicted NOCS
        haslabel_indicator: BT mask indicating if a frame is labeled
        gt: BTN' x 4 dimensional tensor of GT NOCS **for frames with labels!
        gt_foreground_mask: BTN' x 4 of GT segmentation

        '''
        # filter predictions to only those with GT label
        labeled_preds, point2frameidx = self._filter2labeled(preds, haslabel_indicator, point2frameidx)

        # compute per-point NOCS loss
        per_point_loss = self.tnocs_loss_func(labeled_preds, gt)
        foreground_numpnts = global_add_pool(gt_foreground_mask.float(), batch=point2frameidx)
        frame_withpnt_mask = foreground_numpnts.flatten() > 0
        foreground_numpnts[foreground_numpnts == 0] = 1

        # compute foreground loss
        per_point_loss_foreground = per_point_loss * gt_foreground_mask
        frame_loss_foreground = global_add_pool(per_point_loss_foreground, batch=point2frameidx)
        frame_loss_foreground /= foreground_numpnts
        frame_loss_all = 0.1 * global_mean_pool(per_point_loss, batch=point2frameidx)
        tnocs_loss = frame_loss_foreground + frame_loss_all  # BT x 4

        # get L1 metric for logging
        point_l1_metric = (per_point_loss * gt_foreground_mask.view(-1, 1))  # BT x 4
        frame_l1_metric = global_add_pool(point_l1_metric, batch=point2frameidx)
        frame_l1_metric = frame_l1_metric[frame_withpnt_mask]  # remove frames with 0 foreground points
        frame_l1_metric /= foreground_numpnts[frame_withpnt_mask]

        return tnocs_loss, frame_l1_metric


    def _filter2labeled(self, preds, haslabel_indicator, point2frameidx):
        perpoint_haslabel = haslabel_indicator[point2frameidx]
        labeled_preds = preds[perpoint_haslabel]  # should be btn_labeled now
        point2frameidx = point2frameidx[perpoint_haslabel]
        _, point2frameidx = torch.unique(point2frameidx, return_inverse=True)
        return labeled_preds, point2frameidx