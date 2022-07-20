import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.glob.glob import global_max_pool, global_add_pool
from spot.data.box3d import LidarBoundingBoxes


class BoundingBoxRegression(nn.Module):
    def __init__(self, feat_dim, size_bins, batchnorm, network_type='vote', size_pct_thresh=1.0):
        """

        Args:
            feat_dim:
            size_bins: array of possible size bins
            batchnorm: If True, use batchnorm, otherwise groupnorm
            network_type: One of ['vote', 'ode'], indicating the type of predictions to do.
        """
        super(BoundingBoxRegression, self).__init__()
        self.network_type = network_type

        self.feat_dim = feat_dim
        self.size_bins = size_bins
        self.num_size_bins = size_bins.shape[0]
        self.size_pct_thresh = size_pct_thresh

        # further per-point processing
        self.conv1 = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.bn1 = nn.GroupNorm(16, self.feat_dim) if not batchnorm else nn.BatchNorm1d(self.feat_dim)
        self.bn2 = nn.GroupNorm(16, self.feat_dim) if not batchnorm else nn.BatchNorm1d(self.feat_dim)

        if self.network_type == 'vote':
            self.prediction_backbone = VoteBoundingBoxRegression(feat_dim, self.num_size_bins, batchnorm)
        elif self.network_type == "refinement":
            self.prediction_backbone = RefinementBoundingBoxRegression(feat_dim, self.num_size_bins, batchnorm)
        else:
            raise RuntimeError("Box Regression Network Type is not specified: %s" % self.network_type)

        self.regression_loss = torch.nn.L1Loss(reduction='none')
        self.binning_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, point_cloud, bboxes_in=None, frame2batchidx=None, point2frameidx=None):
        """
        x is the point-wise features, ie (feat_size, BTN)
        raw_points is (BTN, 4)
        bboxes_in is (BT, 4)
        """
        # Further pre-point processing
        BTN, _ = point_cloud.size()
        x = F.relu(self.bn1(self.conv1(x.unsqueeze(0))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, BTN).transpose(0, 1).contiguous()

        # format raw points
        if self.network_type == "vote":
            centers, velocities, yaw_sincos, size_residual, size_bin = self.prediction_backbone(x, point_cloud[:, :4], point2frameidx, frame2batchidx)
        else:  # self.network_type == "refinement":
            centers, velocities, yaw_sincos, size_residual, size_bin = self.prediction_backbone(x, bboxes_in, point_cloud[:, :4], point2frameidx, frame2batchidx)

        # format predictions into 7dof box tensor
        size_residuals, size_bins = size_residual[frame2batchidx], size_bin[frame2batchidx]
        BT = frame2batchidx.size(0)
        size_residuals = size_residuals.view(BT, self.num_size_bins, 3)
        winning_size_bins = torch.argmax(size_bins, dim=1)  # size (BT,) tensor
        anchor_size_res = size_residuals[torch.arange(BT), winning_size_bins]  # BT, 3
        accelerations = torch.zeros(centers.size()).to(centers)
        enc_pred_boxes = torch.cat([centers, anchor_size_res, yaw_sincos, velocities, accelerations, size_bins], dim=1)  # size BT x 8
        pred_boxes = LidarBoundingBoxes(box=None, box_enc=enc_pred_boxes, size_anchors=torch.tensor(self.size_bins).to(enc_pred_boxes), all_size_residuals=size_residuals)
        return pred_boxes

    def loss(self, pred_boxes, gt_box, haslabel_indicator):
        """

        Args:
            pred_boxes (LidarBoundingBoxes): bounding box object containing our predictions
            gt_box (LidarBoundingBoxes): bounding box object containing corresponding GT
            haslabel_indicator (torch.tensor): Indicator stating if a a bounding box is labeled or not

        Returns:

        """
        # filter predicted boxes to only those labeled
        labeled_pred_box = pred_boxes[haslabel_indicator]
        num_labels = labeled_pred_box.size(0)

        # compute per-frame metrics
        yaw_err = torch.norm((labeled_pred_box.yaw - gt_box.yaw + np.pi) % (2*np.pi) - np.pi, dim=1)
        center_err = torch.norm(labeled_pred_box.center[:, :2] - gt_box.center[:, :2], dim=1)  # only xy center!
        velocity_err = torch.norm((labeled_pred_box.velocity - gt_box.velocity), dim=1)
        scale_iou_err = LidarBoundingBoxes.scale_iou(labeled_pred_box, gt_box)

        # compute losses
        size_bin_loss = self.binning_loss(labeled_pred_box.box_size_anchor_likelihoods, gt_box.box_size_anchor_idxs).unsqueeze(-1)
        center_loss = self.regression_loss(labeled_pred_box.box_encoding[:, :3], gt_box.box_encoding[:, :3])  # B, T, 7
        vel_and_acc_loss =  self.regression_loss(labeled_pred_box.box_encoding[:, 6:11], gt_box.box_encoding[:, 6:11])  # B, T, 7

        # compute size loss
        pred_assigned_residuals = labeled_pred_box.all_size_residuals[torch.arange(num_labels), gt_box.box_size_anchor_idxs]  # becomes BN' x n-bins x 3 --> BN' x 3
        size_loss = self.regression_loss(pred_assigned_residuals, gt_box.box_encoding[:, 3:6])

        # only take worst-performing size losses
        if self.size_pct_thresh < 1.0:
            size_bin_loss_idxs = torch.argsort(size_bin_loss.flatten(), descending=True)[:int(np.ceil(size_bin_loss.size(0) * self.size_pct_thresh))]
            size_bin_loss = size_bin_loss[size_bin_loss_idxs]
            size_loss_idxs = torch.argsort(torch.norm(size_loss, dim=1), descending=True)[:int(np.ceil(size_bin_loss.size(0) * self.size_pct_thresh))]
            size_loss *= (1/self.size_pct_thresh)
            size_loss[size_loss_idxs] = 0

        # put together into residual loss
        residuals_loss = torch.cat([center_loss, size_loss, vel_and_acc_loss], dim=1)
        return residuals_loss, size_bin_loss, yaw_err, scale_iou_err, center_err, velocity_err


    def spline_consistency_loss(self, pred_boxes, gt_instance_ids, haslabel_indicator):
        pass


class VoteBoundingBoxRegression(nn.Module):
    def __init__(self, featsize, num_size_bins, batchnorm):
        super().__init__()
        self.featsize = featsize

        # center vote processing
        self.vote_weighting = torch.nn.Linear(self.featsize, 1)
        self.voting = torch.nn.Linear(self.featsize, 3)

        # frame-wise yaw prediction and sequence-wise size prediction
        self.yaw_resid_layer = torch.nn.Linear(self.featsize, 2)  # [centers, yaw_residuals]
        self.vel_resid_layer = torch.nn.Linear(self.featsize, 3)

        self.size_binning = torch.nn.Linear(self.featsize, num_size_bins)
        self.size_resid_layer = torch.nn.Linear(self.featsize, num_size_bins*3)  # [wlh_residual]

        if not batchnorm:
            self.bn1 = nn.GroupNorm(16, self.featsize)
        else:
            self.bn1 = nn.BatchNorm1d(self.featsize)

    def forward(self, x, raw_xyz, point2frameidx, frame2batchidx):
        """

        Args:
            x: Per-Point features size (BTN x num_feats)
            raw_xyz: size (BTN, 3)
            point2frameidx: Point to frame assignments (BTN,)

        Returns:

        """

        # max pool per-point features into per-frame features
        frame_pooled_feats = global_max_pool(x, batch=point2frameidx)
        seq_pooled_feats = global_max_pool(frame_pooled_feats, batch=frame2batchidx)

        # predict center offsets
        vote_weights = torch.clip(F.sigmoid(self.vote_weighting(x)), 1e-5)  # BTN x 1
        vote_offsets = self.voting(x)  # BT x 2
        center_votes = (raw_xyz[:, :3] + vote_offsets) * vote_weights
        centers = global_add_pool(center_votes, point2frameidx) / global_add_pool(vote_weights, point2frameidx)

        # predict per-frame velocities --> try frame pairs
        velocities = self.vel_resid_layer(frame_pooled_feats)  # BT x 3

        # predict yaw
        yaw_sincos = self.yaw_resid_layer(frame_pooled_feats)  # BT x 2

        # predict size
        size_residual = self.size_resid_layer(seq_pooled_feats)  # B x (num-bins * 3)
        size_bin = F.softmax(self.size_binning(seq_pooled_feats), dim=1)  # B x size-bins

        return centers, velocities, yaw_sincos, size_residual, size_bin


class RefinementBoundingBoxRegression(nn.Module):
    def __init__(self, featsize, num_size_bins, batchnorm):
        super().__init__()
        self.featsize = featsize
        self.center_layer = torch.nn.Linear(self.featsize, 3)
        self.yaw_resid_layer = torch.nn.Linear(self.featsize, 2)  # [centers, yaw_residuals]
        self.vel_resid_layer = torch.nn.Linear(self.featsize, 3)
        self.size_binning = torch.nn.Linear(self.featsize, num_size_bins)
        self.size_resid_layer = torch.nn.Linear(self.featsize, num_size_bins*3)  # [wlh_residual]

        if not batchnorm:
            self.bn1 = nn.GroupNorm(16, self.featsize)
        else:
            self.bn1 = nn.BatchNorm1d(self.featsize)

    def forward(self, x, bboxes_in, raw_xyz, point2frameidx, frame2batchidx):
        """

        Args:
            x: Per-Point features size (BTN, point_featsize)
            point2frameidx: Point to frame assignments (BTN,)
        Returns:

        """
        # max pool per-point features into per-frame features
        frame_pooled_feats = global_max_pool(x, batch=point2frameidx)
        seq_pooled_feats = global_max_pool(frame_pooled_feats, batch=frame2batchidx)

        # predict center offsets
        centers = self.center_layer(frame_pooled_feats) # BT x 2

        # predict per-frame velocities --> try frame pairs
        velocities = self.vel_resid_layer(frame_pooled_feats)  # BT x 3
        # predict yaw
        yaw_sincos = self.yaw_resid_layer(frame_pooled_feats)  # BT x 2

        # predict size
        size_residual = self.size_resid_layer(seq_pooled_feats)  # B x 3
        size_bin = F.softmax(self.size_binning(seq_pooled_feats), dim=1)  # B x size-bins

        # make our predictions a refinement of existing boxes
        if bboxes_in is not None:
            centers = bboxes_in.center + centers
            rot_mats = torch.stack([bboxes_in.cos_yaw, bboxes_in.sin_yaw, -bboxes_in.sin_yaw, bboxes_in.cos_yaw], dim=1).view(-1, 2, 2)  # BT x 2 x 2
            yaw_sincos = (rot_mats @ yaw_sincos.unsqueeze(2)).squeeze(2)  # BT x 2

        return centers, velocities, yaw_sincos, size_residual, size_bin


