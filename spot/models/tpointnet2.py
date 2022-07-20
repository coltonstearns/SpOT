import torch
import torch.nn as nn
from spot.data.box3d import LidarBoundingBoxes

from .pointnet import PointNet4D
from .bbox_regression import BoundingBoxRegression
from .seg_nocs_regression import ClassificationAndSegmentation
from spot.ops.pc_util import group_first_k_values, merge_point2frame_across_batches
from torch_geometric.nn.glob.glob import global_mean_pool
from .transformer.detr4d import build_4detr
import torch.nn.functional as F

XYZ_SIZE, XYZ_CENTERED_SIZE, T_SIZE, CONF_SIZE = 3, 3, 1, 1

class TPointNet2(nn.Module):
    '''
    TPointNet++
    Extracts an initial z0 feature based on the given sequence and regresses TNOCS points.
    '''
    def __init__(self, cfg, loss_params, size_clusters, object_class, parallel):

        super(TPointNet2, self).__init__()
        self.parallel = parallel
        self.size_clusters = torch.tensor(size_clusters)
        self.object_class = object_class

        # input feature sizes
        self.aug_quad, self.aug_pairs = cfg['in-out-features']['augment-quad'], cfg['in-out-features']['augment-pairs']
        input_box_size = 8 if cfg['use-input-box-size'] else 5
        self.pointnet_input_size = XYZ_SIZE + T_SIZE + XYZ_CENTERED_SIZE + 3*self.aug_pairs + 3*self.aug_quad + CONF_SIZE + input_box_size
        self.transformer_input_size = XYZ_CENTERED_SIZE + 3*self.aug_pairs + 3*self.aug_quad + CONF_SIZE + input_box_size
        self.enc_feat_size = 0

        # proposal bounding-box fusion
        self.input_bbox_use_size = cfg['use-input-box-size']

        # set up architecture
        self.architecture = cfg['architecture']
        assert self.architecture in ["pointnet-4detr", "4detr-only", "pointnet-only"]
        if "4detr" in self.architecture:
            transormer_feat_size = cfg['in-out-features']['transformer-feat-size']
            self.extrinsic_st_attention = build_4detr(cfg['transformer-args'], self.transformer_input_size, transormer_feat_size, object_class)
            self.enc_feat_size += transormer_feat_size
        if "pointnet" in self.architecture:
            self.pointnet_feat_size = cfg['in-out-features']['pointnet-feat-size']
            self.extr_pointnet4d = PointNet4D(input_dim=self.pointnet_input_size, feat_size=self.pointnet_feat_size, batchnorm=True)
            self.enc_feat_size += self.pointnet_feat_size

        # Output networks
        self.classification_and_segmentation = ClassificationAndSegmentation(self.enc_feat_size, group_norm=False)
        self.bbox_regressor = BoundingBoxRegression(feat_dim=self.enc_feat_size, size_bins=size_clusters, batchnorm=True,
                                                    network_type=cfg["bbox-backbone"], size_pct_thresh=cfg["size_pct_thresh"])

        # loss terms
        self.loss_params = loss_params

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Input data in torch_geometric format
        Returns:
        """
        # Format out of torch_geometric
        point2frameidx = merge_point2frame_across_batches(data.point2frameidx, data.batch)
        frame2batchidx, _ = group_first_k_values(data.batch, batch=point2frameidx, k=1)
        frame2batchidx = frame2batchidx.flatten()
        bboxes_in = LidarBoundingBoxes.concatenate(data.boxes).to(data.x)
        x = data.x
        confidences = data.confidences

        # process input points; append box features
        infeats = self._compute_additional_input_feats(x, confidences, bboxes_in, point2frameidx)
        box_feats = bboxes_in.box_encoding[point2frameidx][:, [0, 1, 2, 6, 7]]

        # Extrinsic processing
        extrinsic_feats = self._perpoint_encoding(x, infeats, box_feats, frame2batchidx, point2frameidx, self.extr_pointnet4d, self.extrinsic_st_attention)

        # Extrinsic Predictions
        extr_boxes = self.bbox_regressor(extrinsic_feats, x, bboxes_in, frame2batchidx, point2frameidx)
        segmentation_out, feats, classification = self.classification_and_segmentation(extrinsic_feats, frame2batchidx[point2frameidx], point2frameidx)
        all_size_residuals = extr_boxes.all_size_residuals
        if self.parallel: extr_boxes = extr_boxes.box_enc

        # Segmentation Predictions
        out_dict = {"latent_encoding": feats, "segmentation": segmentation_out,
                    "bbox_regression": extr_boxes, "extr_perpoint_feats": extrinsic_feats.transpose(0, 1).contiguous(),
                    "classification": classification, "frame2batchidx": frame2batchidx, "all_size_residuals": all_size_residuals}

        return out_dict

    def _compute_additional_input_feats(self, x, confidences, bboxes_in, point2frameidx=None):
        """
            point_assignments: Boolean assignment matrix of size (BTN, BT), indicating which point belongs to which frame
        """
        frame_means = global_mean_pool(x, batch=point2frameidx)
        per_point_means = frame_means[point2frameidx]  # size BTN x 4

        local_in = x[:, :3] - per_point_means[:, :3]  # mean center each pc in the sequence
        if self.aug_quad:
            quad_terms = local_in[:, 0:3] * local_in[:, 0:3]
            local_in = torch.cat([local_in, quad_terms], dim=1)
        if self.aug_quad:
            xz = local_in[:, 0:1] * local_in[:, 2:3]
            xy = local_in[:, 0:1] * local_in[:, 1:2]
            yz = local_in[:, 2:3] * local_in[:, 1:2]
            local_in = torch.cat([local_in, xz, xy, yz], dim=1)
        local_in = torch.cat([local_in, confidences.unsqueeze(-1)[point2frameidx]], dim=1)

        if self.input_bbox_use_size:
            local_in = torch.cat([local_in, bboxes_in.wlh[point2frameidx]], dim=1)

        return local_in

    def _perpoint_encoding(self, x, infeats, box_feats, frame2batchidx, point2frameidx, pointnet4d, detr4d):
        # Run 4D PointNet
        BTN, _ = x.size()
        B = torch.max(frame2batchidx).item() + 1
        point2batchidx = frame2batchidx[point2frameidx]

        if "pointnet" in self.architecture:
            pn4d_in = torch.cat([x, infeats, box_feats], dim=1) if box_feats is not None else torch.cat([x, infeats], dim=1)
            pn4d_in = pn4d_in.view(1, BTN, -1).transpose(2, 1).contiguous()
            pn4d_out, _ = pointnet4d(pn4d_in, frame2batchidx[point2frameidx])  # feats, BTN

        # Run 4D Transformer
        if "4detr" in self.architecture:
            detr_feats, _ = detr4d(x, infeats, box_feats, frame2batchidx, point2frameidx)

        # Combine all per-point features
        if self.architecture == "pointnet-4detr":
            out_feats = torch.stack([pn4d_out.squeeze(0), detr_feats], dim=0)  # 2, feats//2, BTN
            out_feats = F.layer_norm(out_feats.transpose(1, 2).contiguous(), normalized_shape=(self.pointnet_feat_size,)).transpose(1, 2).contiguous()  # normalizes point feature magnitudes
            if self.training:
                out_feats = self._layernorm_training_dropout(out_feats, point2batchidx, B)
            out_feats = out_feats.view(-1, BTN)
        elif self.architecture == "pointnet-only":
            out_feats = pn4d_out.squeeze(0)
        else:
            out_feats = detr_feats

        return out_feats

    def _layernorm_training_dropout(self, out_feats, point2batchidx, batch_size):
        manual_dropout_thresh = torch.rand(batch_size).to(out_feats)

        # apply pointnet-dropout
        pointnet_dropout = (manual_dropout_thresh < 0.25)[point2batchidx]
        out_feats[0, :, pointnet_dropout] *= 0
        out_feats[1, :, pointnet_dropout] *= 2

        # apply transformer dropout
        transformer_dropout = ((manual_dropout_thresh >= 0.25) & (manual_dropout_thresh < 0.5))[point2batchidx]
        out_feats[0, :, transformer_dropout] *= 2
        out_feats[1, :, transformer_dropout] *= 0

        return out_feats

    def loss(self, predictions, gt, point2frameidx=None):
        success = check_success(predictions)
        if not success:
            return {}, {}, False

        # losses for per-point segmentation
        segmentation_loss, mIoU, IoU_above_70 = self.classification_and_segmentation.segmentation_loss(predictions['segmentation'], gt.segmentation, gt.haslabel_mask, point2frameidx)
        segmentation_loss *= self.loss_params['usage']['segmentation-loss'] * self.loss_params['weight']['segmentation-loss']

        # losses for bounding box regression
        regressed_boxes = predictions['bbox_regression']
        l1_regression_loss, size_bin_loss, yaw_err, scale_iou_err, center_err, velocity_err = self.bbox_regressor.loss(regressed_boxes, gt.boxes, gt.haslabel_mask)
        l1_regression_loss[:, :2] *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-center-xy-residuals']
        l1_regression_loss[:, 2:3] *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-center-height-residuals']
        l1_regression_loss[:, 3:6] *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-size-residuals']
        l1_regression_loss[:, 6:8] *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-yaw-residuals']
        l1_regression_loss[:, 8:11] *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-velocity-residuals']

        center_res_loss, size_res_loss, yaw_res_loss, vel_res_loss = l1_regression_loss[:, :3], l1_regression_loss[:, 3:6], l1_regression_loss[:, 6:8], l1_regression_loss[:, 8:11]
        size_bin_loss *= self.loss_params['usage']['bbox-regression'] * self.loss_params['weight']['bbox-regress-size-bin']

        # compute classification loss
        classification_temp = self.loss_params['other']['classification-temperature']
        if isinstance(classification_temp, dict):
            classification_temp = classification_temp[self.object_class]
        classification_dropout = self.loss_params['other']['classification-worst-ratio']
        if isinstance(classification_dropout, dict):
            classification_dropout = classification_dropout[self.object_class]
        classification_loss, prec, rec = self.classification_and_segmentation.\
            classification_loss(predictions['classification'], gt.classification, center_err, gt.haslabel_mask, gt.keyframe_mask,
                                temperature=classification_temp, worst_ratio=classification_dropout)
        classification_loss *= self.loss_params['usage']['classification'] * self.loss_params['weight']['classification']

        # put everything together
        success = True
        loss_dict = {"segmentation": segmentation_loss, "bbox-velocity-residuals": vel_res_loss,
                     "bbox-center-residuals": center_res_loss, "bbox-size-residuals": size_res_loss,
                     "bbox-yaw-residuals": yaw_res_loss, "size-bin": size_bin_loss, "classification": classification_loss}
        metrics_dict = { "segmentation_mIoU": mIoU, "yaw-l1": yaw_err, "scale-iou": scale_iou_err, "center-l1": center_err,
                         "velocity-l1": velocity_err, "class-precision": prec, "class-recall": rec}

        return loss_dict, metrics_dict, success


def check_success(predictions):
    failed = lambda x: torch.any(x.isnan())
    failure = False
    for k, pred in predictions.items():
        if torch.is_tensor(pred) or isinstance(pred, LidarBoundingBoxes):
            failure = failure or failed(pred)
        else:
            for val in pred:
                failure = failure or failed(val)
    return not failure
