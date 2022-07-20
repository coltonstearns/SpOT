import torch
from spot.ops.data_utils import batched_index_select
import numpy as np
import torch.nn.functional as F

class LidarBoundingBoxes:

    def __init__(self, box, box_enc, size_anchors, all_size_residuals=None):
        """

        Args:
            box (torch.tensor): Size (K, 9). Array of 3D bounding boxes with parameters [center, wlh, yaw, vel]
            size_anchors (torch.tensor): Size num_anchors x 3 indicating the wlh of the anchor
        """
        assert not (box is None and box_enc is None)

        # set size anchors
        assert size_anchors is not None
        self.size_anchors = size_anchors
        self.num_size_anchors = size_anchors.size(0)

        # load boxes
        if box is not None and box_enc is None:
            self.box = box
            self.box_enc = self._encode_box()
        elif box_enc is not None and box is None:
            self.box_enc = box_enc
            self.box = self._decode_box()
        else:
            self.box = box
            self.box_enc = box_enc
        self.all_size_residuals = all_size_residuals
        self.device = self.box.device
        assert self.box.dtype == torch.float

    @property
    def center(self):
        return self.box[:, :3]

    @property
    def yaw(self):
        return self.box[:, 6:7]

    @property
    def wlh(self):
        return self.box[:, 3:6]

    @property
    def velocity(self):
        return self.box[:, 7:10]

    @property
    def acceleration(self):
        return self.box[:, 10:13]

    @property
    def box_encoding(self):
        return self.box_enc

    @property
    def box_size_anchor_idxs(self):
        anchor_idxs = torch.argmax(self.box_enc[:, -self.num_size_anchors:], dim=1)
        return anchor_idxs

    @property
    def box_size_anchor_likelihoods(self):
        return self.box_enc[:, -self.num_size_anchors:]

    @property
    def cos_yaw(self):
        return self.box_enc[:, 7]

    @property
    def sin_yaw(self):
        return self.box_enc[:, 6]

    def _decode_box(self):
        """
        Takes in the internal network representation and returns the corresponding 7-dof box.
        box_encodings: (M, 7+size_bins) tensor
        """
        # centers fall out directly
        centers = self.box_enc[:, :3]
        vels = self.box_enc[:, 8:11]
        accs = self.box_enc[:, 11:14]

        # get yaws
        yaws_sincos_unsized = self.box_enc[:, 6:8]
        yaws_sincos = yaws_sincos_unsized / torch.norm(yaws_sincos_unsized, dim=1, keepdim=True)
        yaws = torch.atan2(yaws_sincos[:, 0], yaws_sincos[:, 1]).view(-1, 1)

        # get sizes
        m = self.box_enc.size()[0]
        size_anchors = self.size_anchors.clone().view(1, -1, 3).transpose(1, 2).repeat(m, 1, 1).to(self.box_enc.device)  # m x num_bins x 3
        size_bin_idxs = torch.argmax(self.box_enc[:, -self.num_size_anchors:], dim=1)  # size (num_bins,) of indices
        sizes = batched_index_select(size_anchors, dim=2, index=size_bin_idxs.view(-1, 1)).squeeze(2)
        sizes += self.box_enc[:, 3:6]
        return torch.cat([centers, sizes, yaws, vels, accs], dim=1)

    def _encode_box(self):
        """ Encode box to network input/output """
        # generate center target
        center = self.box[:, :3]
        vels = self.box[:, 7:10]
        accs = self.box[:, 10:13]

        # generate bbox size target
        size_bin_target, size_res_target, _ = self.size2sizebin(self.box[:, 3:6], self.size_anchors)
        size_bin_target = F.one_hot(size_bin_target.flatten().type(torch.int64), num_classes=self.num_size_anchors).type(torch.FloatTensor).to(self.box.device)

        # generate dir target
        yaws = self.box[:, 6] % (2 * np.pi)
        yaw_sin, yaw_cos = torch.sin(yaws), torch.cos(yaws)
        yaw_sincos = torch.stack([yaw_sin, yaw_cos], dim=1)

        # convert bin targets into one-hot
        enc_box = torch.cat([center, size_res_target, yaw_sincos, vels, accs, size_bin_target], dim=1)
        return enc_box

    def compute_size_residuals(self):
        _, _, all_size_residuals = self.size2sizebin(self.box[:, 3:6], self.size_anchors)  # N x size_bins x 3
        return all_size_residuals

    def coordinate_transfer(self, affine, batch, inverse=False):
        """
        Returns a LidarBoundingBoxes object in the transformed reference frame.
        Args:
            affine (torch.tensor): size (B, 4, 4) sequence of affine matrix transformations
            batch (torch.tensor): size (BT,) tensor indicating which box belongs to which batch

        Returns:
        """
        boxes = self.box.clone()
        if inverse:
            affine = affine.inverse()
        affines = affine[batch]  # BT, 4, 4

        # apply rotation + translation to bounding box center
        boxes[:, :3] = (affines[:, :3, :3].double() @ boxes[:, :3].double().unsqueeze(-1)).squeeze(-1)
        boxes[:, :3] += affines[:, :3, 3]

        # apply rotation to velocity and acceleration
        boxes[:, 7:10] = (affines[:, :3, :3].double() @ boxes[:, 7:10].double().unsqueeze(-1)).squeeze(-1)
        boxes[:, 10:13] = (affines[:, :3, :3].double() @ boxes[:, 10:13].double().unsqueeze(-1)).squeeze(-1)

        # apply rotation to yaws
        v = affines[:, :3, :3].float() @ torch.tensor([1, 0, 0]).view(3, 1).to(boxes)
        affine_yaws = torch.atan2(v[:, 1], v[:, 0])
        yaws = boxes[:, 6:7] + affine_yaws
        yaws = yaws % (2*np.pi)
        yaws[yaws > np.pi] -= 2 * np.pi
        boxes[:, 6:7] = yaws
        return LidarBoundingBoxes(box=boxes, box_enc=None, size_anchors=self.size_anchors.clone(), all_size_residuals=self.all_size_residuals)

    def translate(self, translation, batch):
        """
        Returns a LidarBoundingBoxes object translated by the batch-assigned vector.
        Args:
            translation (torch.tensor): size (B, 3) sequence of center translations
            batch (torch.tensor): size (BT,) tensor indicating which box belongs to which batch

        Returns:
        """
        boxes = self.box.clone()
        box_encs = self.box_enc.clone()
        translations = translation.clone()[batch.clone()]  # BT, 3
        boxes[:, :3] += translations
        box_encs[:, :3] += translations
        size_anchors = self.size_anchors.clone()
        return LidarBoundingBoxes(box=boxes, box_enc=box_encs, size_anchors=size_anchors, all_size_residuals=self.all_size_residuals)

    def rotate(self, angle, batch, points=None, point2batch=None):
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        zero_pad = torch.zeros(angle.size(0)).to(angle)
        rot_mat = torch.stack([rot_cos, -rot_sin, zero_pad,
                                 rot_sin, rot_cos, zero_pad,
                                 zero_pad, zero_pad, zero_pad+1], dim=1).view(angle.size(0), 3, 3)  # B x 3 x 3

        center = (rot_mat[batch] @ self.center.unsqueeze(-1)).squeeze(-1)
        velocity = (rot_mat[batch] @ self.velocity.unsqueeze(-1)).squeeze(-1)
        acceleration = (rot_mat[batch] @ self.acceleration.unsqueeze(-1)).squeeze(-1)

        # update yaw
        yaw = (self.yaw + angle[batch].unsqueeze(-1)) % (2*np.pi)
        yaw[yaw > np.pi] -= 2*np.pi

        if self.box.shape[1] == 9:
            pass  # in future, handle velocity vector as well

        out = LidarBoundingBoxes(box=torch.cat([center, self.wlh, yaw, velocity, acceleration], dim=1), box_enc=None, size_anchors=self.size_anchors,  all_size_residuals=self.all_size_residuals)
        if points is None:
            return out

        else:
            if isinstance(points, torch.Tensor):
                points_rotated = points.clone()
                points_rotated[:, :3] = (rot_mat[point2batch] @ points_rotated[:, :3].unsqueeze(-1)).squeeze(-1)
            elif isinstance(points, list):
                points_rotated = []
                rot_mats = rot_mat[batch]
                for i, point_batch in enumerate(points):
                    point_batch_rotated = point_batch.clone()
                    point_batch_rotated[:, :3] = point_batch_rotated[:, :3] @ rot_mats[i].transpose(0, 1)
                    points_rotated.append(point_batch_rotated)
            else:
                raise ValueError

            return out, points_rotated

    def flip(self, bev_direction='horizontal', points=None):
        assert bev_direction in ('horizontal', 'vertical')
        box = self.box.clone()
        if bev_direction == 'horizontal':
            box[:, 1] = -box[:, 1]  # todo: modify for velocity as well...
            box[:, 8] = -box[:, 8]
            box[:, 11] = -box[:, 11]
            box[:, 6] = -box[:, 6]
        elif bev_direction == 'vertical':
            box[:, 0] = -box[:, 0]
            box[:, 7] = -box[:, 7]
            box[:, 10] = -box[:, 10]
            box[:, 6] = (-box[:, 6] + np.pi) % (2*np.pi)

        out = LidarBoundingBoxes(box=box, box_enc=None, size_anchors=self.size_anchors)
        if points is None:
            return out

        else:
            if isinstance(points, torch.Tensor):
                points_flipped = points.clone()
                if bev_direction == 'horizontal':
                    points_flipped[:, 1] *= -1
                elif bev_direction == 'vertical':
                    points_flipped[:, 0] *= -1

            elif isinstance(points, list):
                points_flipped = []
                for i, point_batch in enumerate(points):
                    point_batch_flipped = point_batch.clone()
                    if bev_direction == 'horizontal':
                        point_batch_flipped[:, 1] *= -1
                    elif bev_direction == 'vertical':
                        point_batch_flipped[:, 0] *= -1

                    points_flipped.append(point_batch_flipped)

            else:
                raise ValueError

            return out, points_flipped

    def scale(self, scale, batch):
        boxes = self.box.clone()
        scales = scale.clone()[batch.clone()]  # BT, 3
        boxes[:, :6] *= scales.view(boxes.size(0), -1)
        boxes[:, 7:] *= scales.view(boxes.size(0), -1)
        size_anchors = self.size_anchors.clone()
        return LidarBoundingBoxes(box=boxes, box_enc=None, size_anchors=size_anchors)

    def box_canonicalize(self, points, batch, scale=False):
        # center points
        canonical_points = points.clone()
        canonical_points[:, :3] -= self.center[batch]

        # rotate points appropriate box angles
        angles, dummy_frame_idxs = -self.yaw.flatten().clone(), torch.zeros(self.size(0), dtype=torch.long).to(self.device)
        _, canonical_points[:, :3] = self.rotate(angles, dummy_frame_idxs.long(), canonical_points[:, :3], batch)

        if scale:
            scales = torch.norm(self.wlh, dim=1, p=2)
            scales = scales[batch].view(-1, 1)
            canonical_points[:, :3] /= scales

        return canonical_points

    def __getitem__(self, idx):
        indexed_box = self.box[idx]
        indexed_box_enc = self.box_enc[idx]
        if torch.is_tensor(self.all_size_residuals):
            all_size_residuals = self.all_size_residuals[idx]
        else:
            all_size_residuals = None
        return LidarBoundingBoxes(box=indexed_box, box_enc=indexed_box_enc, size_anchors=self.size_anchors, all_size_residuals=all_size_residuals)

    def __setitem__(self, idx, value):
        self.box[idx] = value.box
        self.box_enc[idx] = value.box_enc
        if torch.is_tensor(self.all_size_residuals):
            self.all_size_residuals[idx] = value.all_size_residuals

    def __delitem__(self, idx):
        del self.box[idx]
        del self.box_enc[idx]
        if torch.is_tensor(self.all_size_residuals):
            del self.all_size_residuals[idx]

    def clone(self):
        """Clone the Boxes.

        Returns:
            :obj:`BaseInstance3DBoxes`: Box object with the same properties \
                as self.
        """
        box = self.box.clone()
        box_enc = self.box_enc.clone()
        size_anchors = self.size_anchors.clone()
        if torch.is_tensor(self.all_size_residuals):
            all_size_residuals = self.all_size_residuals.clone()
        else:
            all_size_residuals = None
        return LidarBoundingBoxes(box=box, box_enc=box_enc, size_anchors=size_anchors, all_size_residuals=all_size_residuals)

    def detach(self):
        self.box = self.box.detach()
        self.box_enc = self.box_enc.detach()
        self.size_anchors = self.size_anchors.detach()
        if torch.is_tensor(self.all_size_residuals):
            self.all_size_residuals.detach()
        return self

    def contiguous(self):
        self.box = self.box.contiguous()
        self.box_enc = self.box_enc.contiguous()
        self.size_anchors = self.size_anchors.contiguous()
        if torch.is_tensor(self.all_size_residuals):
            self.all_size_residuals.contiguous()
        return self

    def to(self, device):
        self.device = device
        self.box = self.box.to(device)
        self.box_enc = self.box_enc.to(device)
        self.size_anchors = self.size_anchors.to(device)
        if torch.is_tensor(self.all_size_residuals):
            self.all_size_residuals = self.all_size_residuals.to(device)
        return self

    def size(self, dim=None):
        if dim is None:
            return self.box.size()
        else:
            return self.box.size(dim)

    def isnan(self):
        return self.box.isnan().sum(dim=1) | self.box_enc.isnan().sum(dim=1)

    def new_box(self, box=None, box_enc=None):
        size_anchors = self.size_anchors.clone()
        return LidarBoundingBoxes(box=box, box_enc=box_enc, size_anchors=size_anchors)

    @staticmethod
    def size2sizebin(wlhs, size_clusters):
        # generate bbox size target
        N = wlhs.size(0)
        device = wlhs.device
        nbins = size_clusters.size(0)

        # get all possible residuals
        all_resids = wlhs.repeat(1, nbins).view(N, nbins, 3) - size_clusters.view(1, nbins, 3)
        anchor_idxs = torch.min(torch.norm(all_resids, dim=2), dim=1)[1]
        box_idxs = torch.arange(N).to(wlhs.device).long()
        residuals = all_resids[box_idxs, anchor_idxs]  # N x 3

        # format outputs
        size_bin_target = anchor_idxs.type(torch.FloatTensor).to(device).view(-1, 1)
        size_res_target = residuals
        return size_bin_target, size_res_target, all_resids

    @staticmethod
    def concatenate(lidar_bboxes, dim=0):
        """

        Args:
            lidar_bboxes (list<LidarBoundingBoxes>): list of LidarBoundingBox Object that we intend to merge together.

        Returns: LidarBoundingBoxes with all boxes merged.

        """
        assert len(lidar_bboxes) > 0
        size_anchors = lidar_bboxes[0].size_anchors.clone()
        box = torch.cat([box.box for box in lidar_bboxes], dim=dim)
        box_enc = torch.cat([box.box_enc for box in lidar_bboxes], dim=dim)
        all_with_all_size_residuals = all([torch.is_tensor(lidar_bboxes[i].all_size_residuals) for i in range(len(lidar_bboxes))])
        if all_with_all_size_residuals:
            all_size_residuals = torch.cat([box.all_size_residuals for box in lidar_bboxes], dim=dim)
        else:
            all_size_residuals = None

        return LidarBoundingBoxes(box=box, box_enc=box_enc, size_anchors=size_anchors, all_size_residuals=all_size_residuals)

    @staticmethod
    def scale_iou(boxes1, boxes2):
        """

        Args:
            boxes1 (LidarBoundingBoxes): first set of bounding boxes
            boxes2: second set of bounding boxes

        Returns:
        """
        stacked_sizes = torch.stack((boxes1.wlh, boxes2.wlh), dim=2)  # size (M, 3, 2)
        stacked_sizes = torch.clamp(stacked_sizes, min=0)
        intersecting_volumes = torch.min(stacked_sizes[:, 0, :], dim=1)[0] * \
                               torch.min(stacked_sizes[:, 1, :], dim=1)[0] * \
                               torch.min(stacked_sizes[:, 2, :], dim=1)[0]
        box1_volumes = torch.prod(boxes1.wlh, dim=1)
        box2_volumes = torch.prod(boxes2.wlh, dim=1)
        union_volumes = box1_volumes + box2_volumes - intersecting_volumes
        scale_iou = intersecting_volumes / union_volumes
        return scale_iou





