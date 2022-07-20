import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS
from .partial_bin_based_bbox_coder import PartialBinBasedBBoxCoder
from spot.ops.data_utils import batched_index_select


@BBOX_CODERS.register_module()
class CasprBBoxCoder(PartialBinBasedBBoxCoder):
    """Anchor free bbox coder for 3D boxes.

    Args:
        num_dir_bins (int): Number of bins to encode direction angle.
        num_sizes (int): Number of size clusters.
        size_clusters (list[list[list[float]]]): Inner list of lists represents 2D array of sizes for the object. Outer list
                                              represents per-class size-clusters.
        with_rot (bool): Whether the bbox is with rotation.
    """

    def __init__(self, num_dir_bins, size_clusters, with_rot=True):
        super(CasprBBoxCoder, self).__init__(
            num_dir_bins, 0, [], with_rot=with_rot)
        self.size_clusters = size_clusters

    def encode(self, gt_bboxes_3d, gt_label):
        """Encode ground truth to prediction targets. Only works for one class-type at a time!

        Args:
            gt_bboxes_3d (BaseInstance3DBoxes): Ground truth bboxes with shape (n, 7) for ONE specific class.
            gt_labels_3d (torch.Tensor): Ground truth class.

        Returns:
            tuple: Targets of center, size and direction.
        """
        # generate center target
        center_target = gt_bboxes_3d.gravity_center

        n_boxes = gt_bboxes_3d.tensor.size(0)
        device = gt_bboxes_3d.tensor.device

        # generate bbox size target
        size_clusters = gt_bboxes_3d.tensor.new_tensor(self.size_clusters[gt_label])
        num_clusters = size_clusters.size(0)
        bbox_dims = self.convert_mmdet2nusc_dims(gt_bboxes_3d.dims) #
        size_diffs = bbox_dims.repeat(1, num_clusters).view(n_boxes, num_clusters, 3) - size_clusters.view(1, num_clusters, 3)
        size_diff_norms = torch.norm(size_diffs, dim=2)
        anchor_idxs = torch.min(size_diff_norms, dim=1)[1]
        intermediate_idxs = [i*n_boxes+i for i in range(n_boxes)]
        residuals = size_diffs[:, anchor_idxs, :].view(n_boxes*n_boxes, 3)[intermediate_idxs, :]
        size_bin_target = anchor_idxs.type(torch.FloatTensor).to(device).view(-1, 1)
        size_res_target = residuals

        # generate dir target
        if self.with_rot:
            (dir_bin_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d.yaw)
        else:
            dir_bin_target = gt_bboxes_3d.tensor.new_zeros(n_boxes)
            dir_res_target = gt_bboxes_3d.tensor.new_zeros(n_boxes)

        return (center_target, size_bin_target, size_res_target,
                dir_bin_target.view(-1, 1), dir_res_target.view(-1, 1))

    def convert_mmdet2nusc_dims(self, dims):
        return dims[:, [1, 0, 2]]

    def convert_nusc2mmdet_dims(self, dims):
        return dims[:, [1, 0, 2]]

    def decode(self, bbox_out, gt_label=None, suffix='', from_caspr=False):
        """Decode predicted parts to bbox3d.

        Args:
            bbox_out (dict): Predictions from model, should contain keys below.

                - center: predicted center of bboxes (true middle of box, not bottom).
                - dir_class: predicted bbox direction class.
                - dir_res: predicted bbox direction residual.
                - size_class: predicted bbox size class.
                - size_res: predicted bbox size residual.
            suffix (str): Decode predictions with specific suffix.

        Returns:
            torch.Tensor: Decoded bbox3d with shape (num_proposals, 7).
        """
        center = bbox_out['center' + suffix]
        num_proposal = center.shape[0]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out['dir_class' + suffix], dim=1).view(num_proposal, 1)
            dir_res = bbox_out['dir_res' + suffix]
            dir_angle = self.class2angle(dir_class, dir_res, from_caspr=from_caspr).reshape(num_proposal, 1)
        else:
            dir_angle = center.new_zeros(num_proposal, 1)

        # decode bbox size
        size_clusters = center.new_tensor(self.size_clusters[gt_label])  # (num_sizes, 3)
        pred_cluster_idxs = torch.argmax(bbox_out['size_class' + suffix], dim=1, keepdim=True) # size (num_props, 1)
        anchor_sizes = batched_index_select(size_clusters.repeat(num_proposal, 1).view(num_proposal, -1, 3), dim=1, index=pred_cluster_idxs).view(num_proposal, 3)

        # anchor_sizes = size_clusters.repeat(num_class_proposals, 1).view(-1, num_class_proposals, 3)[pred_cluster_idxs, pred_cluster_idxs, :]
        bbox_size = anchor_sizes + bbox_out['size_res' + suffix][:, [1, 0, 2]]  # (n,3) + (n,3)  # todo: unsure why we swap the residual here...
        bbox_size = self.convert_nusc2mmdet_dims(bbox_size)

        center[:, 2] -= bbox_size[:, 2]/2  # put at "bottom" of box

        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def angle2class(self, angle):
        """Convert continuous angle to a discrete bin and a residual.

        Convert continuous angle to a discrete class and a small
        regression number from class center angle to current angle.

        Args:
            angle (torch.Tensor): Angle is from 0-2pi (or -pi~pi),
                class center at 0+(pi/N), 1*(2pi/N)+(pi/N), 2*(2pi/N)+(pi/N) ...  (N-1)*(2pi/N)+(pi/N).

        Returns:
            tuple: Encoded discrete class and residual.
        """
        angle = angle % (2 * np.pi)
        angle_per_bin = 2 * np.pi / self.num_dir_bins
        angle_cls = torch.floor(angle / angle_per_bin)
        angle_res = (angle % angle_per_bin) - (angle_per_bin/2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True, from_caspr=False):
        """Inverse function to angle2class.

        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].

        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        if from_caspr:
            angle_res *= (np.pi / 180)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res + angle_per_class/2
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi

        return angle
