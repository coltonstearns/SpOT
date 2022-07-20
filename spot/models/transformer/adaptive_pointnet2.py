import third_party.pointnet2.pointnet2_utils as pointnet2_utils
import third_party.pointnet2.pytorch_utils as pt_utils
import torch_cluster
import numpy as np
import torch
import torch.nn as nn
from typing import List
from third_party.pointnet2.pointnet2_utils import three_interpolate, grouping_operation
NUM_GROUPS = 16 # for group norm
import torch.nn.functional as F
from spot.ops.pc_util import group_first_k_values
from spot.ops.torch_utils import torch_sets_unique

class AdaptiveBatchPointnetSAModule(nn.Module):
    ''' Modified based on _PointnetSAModuleBase to efficiently handle varying-size point clouds. '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            fps_sample_ratio: float = 1.0,
            fps_lower_thresh: int = 3
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.fps_sample_ratio = fps_sample_ratio
        self.fps_lower_thresh = fps_lower_thresh

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
        self.mlp_outdim = mlp_spec[-1]


    def forward(self, xyz: torch.Tensor,
                times: torch.Tensor = None,
                features: torch.Tensor = None,
                boxes: torch.Tensor = None,
                point2frameidx = None,
                frame2batchidx = None,
                use_boxes = False):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (BN, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (BN, C) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        # ================== Run FPS ==================
        torchcluster_ratio = 0.8
        inds = torch_cluster.fps(xyz, point2frameidx, ratio=torchcluster_ratio, random_start=True)

        # ============= Filter to Fixed Number of Queries Per Frame ===========
        points_per_frame = torch.unique(point2frameidx, return_counts=True)[1]
        frame_query_limits = torch.ceil(points_per_frame * self.fps_sample_ratio)
        frame_query_limits = torch.clip(frame_query_limits, min=self.fps_lower_thresh, max=self.npoint).long()
        num_fps_available_limits = torch.ceil(points_per_frame * torchcluster_ratio).long()  # this becomes min
        frame_query_limits = torch.where(frame_query_limits > num_fps_available_limits, num_fps_available_limits, frame_query_limits)

        # Get global query idxs
        dense_query2frameidx = point2frameidx[inds]  # frame idx corresponding to each fps sample
        inds_grouped, inds_mask = group_first_k_values(inds, dense_query2frameidx, k=frame_query_limits)
        inds = inds_grouped.flatten()[inds_mask.flatten()]

        # filter based on indices
        query_xyz = xyz[inds].unsqueeze(0)  # 1 x nquery x 3
        query_times = times[inds].unsqueeze(0)
        query_boxes = boxes[inds].unsqueeze(0) if boxes is not None else None
        query_frame_idxs = point2frameidx[inds]  # one frame idx per query point
        query_batch_idxs = frame2batchidx[query_frame_idxs]  # one batch idx per query point
        # this outputs 1D array of FPS indices
        # ===========================================

        # ================= Compute Per-Point FPS Assignment(s) ==================
        num_queries = inds.size(0)
        query_assignments = torch_cluster.radius(x=xyz, y=xyz[inds], r=self.radius, batch_x=point2frameidx, batch_y=query_frame_idxs, max_num_neighbors=self.nsample)  # outputs 2D size (num_queries, total_assigned)
        # messy bug patch, because torch_cluster randomly drops query points
        bug_idx = 0
        while torch.unique(query_assignments[0, :]).size(0) < num_queries:
            failed_qidxs = torch_sets_unique(torch.unique(query_assignments[0, :]), torch.arange(num_queries).to(query_assignments))
            for failed_qidx in failed_qidxs:
                frame_pnts = xyz[point2frameidx == query_frame_idxs[failed_qidx]]
                query_pnt = xyz[inds][failed_qidx:failed_qidx+1]
                qdists = torch.norm(frame_pnts - query_pnt, dim=1)  # todo: is this using the whole bbox as a feature --> causes large distance?
                assign_idxs = torch.arange(xyz.size(0)).to(xyz.device)[point2frameidx == query_frame_idxs[failed_qidx]][qdists < self.radius][:self.nsample]
                fix_qidx = torch.stack([torch.ones(assign_idxs.size(0)).to(failed_qidxs)*failed_qidx, assign_idxs])
                query_assignments = torch.cat([query_assignments, fix_qidx], dim=1)
            if bug_idx > 0:
                print("Bug Patch is requiring more than 1 iteration?? On Iter %s" % bug_idx)
                print("Dumping Variables:")
                print(torch.unique(query_assignments[0, :]).size(0))
                print(num_queries)
                print(failed_qidxs)
                print("Error in query assignments. Putting in identity assignment only")
                for failed_qidx in failed_qidxs:
                    query_idx_in_full_batch = inds[failed_qidx].item()
                    fix_qidx = torch.stack([torch.ones(1).to(failed_qidxs)*failed_qidx, torch.ones(1).to(failed_qidxs)*query_idx_in_full_batch])
                    query_assignments = torch.cat([query_assignments, fix_qidx], dim=1)
                break
            bug_idx += 1

        pointgroups_idxs, pointgroups_mask = group_first_k_values(query_assignments[1, :], query_assignments[0, :], k=self.nsample)
        pointgroups_idxs = pointgroups_idxs.unsqueeze(0).int()
        # point_group_idxs is (1, nquery, nsample) of grouped fps points

        # ====================== Get grouped points and features =======================
        xyz_flipped = xyz.unsqueeze(0).transpose(1, 2).contiguous()
        if features is not None:
            features_flipped = features.unsqueeze(0).transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_flipped, pointgroups_idxs.int())  # returns (1, 3, nquery, nsample)
        grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features_flipped, pointgroups_idxs)
            if self.use_xyz:
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (1, C + 3, nquery, nsample)
            else:
                grouped_features = grouped_features
        else:
            assert (self.use_xyz), "Cannot have not features and not use xyz as a feature!"
            grouped_features = grouped_xyz

        if boxes is not None and use_boxes:
            boxes_flipped = boxes.unsqueeze(0).transpose(1, 2).contiguous()
            grouped_boxes = grouping_operation(boxes_flipped, pointgroups_idxs)
            grouped_boxes[:, 0:3, :, :] -= query_xyz.transpose(1, 2).unsqueeze(-1)
            if self.normalize_xyz:
                grouped_boxes[:, 0:3, :, :] /= self.radius
            grouped_features = torch.cat([grouped_features, grouped_boxes], dim=1)
        # ========================================================================

        # ======================== Run MLP on Grouped Points =====================
        new_features = self.mlp_module(grouped_features)  # (1, mlp[-1], nquery, nsample)
        # =======================================================================

        # ======================= Pool Features Appropriately ====================
        expand_point_mask = pointgroups_mask.expand(self.mlp_outdim, -1, -1).unsqueeze(0)
        if self.pooling == 'max':
            new_features = new_features.masked_fill(~expand_point_mask, -np.inf)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'avg':
            new_features = new_features.masked_fill(~expand_point_mask, 0.0)
            new_features = torch.sum(new_features, dim=3, keepdim=True)
            new_features /= torch.sum(pointgroups_mask, dim=1).view(1, 1, num_queries, 1) # (B, mlp[-1], npoint, 1)
        elif self.pooling == 'rbf':
            # Use radial basis function kernel for weighted sum of features (normalized by nsample and sigma)
            # Ref: https://en.wikipedia.org/wiki/Radial_basis_function_kernel
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1,keepdim=False) / (self.sigma**2) / 2) # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True)
            new_features = new_features.masked_fill(~expand_point_mask, 0.0)
            new_features = torch.sum(new_features) / torch.sum(pointgroups_mask, dim=1).view(1, 1, num_queries, 1) # (B, mlp[-1], npoint, 1)
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        # =======================================================================

        return query_xyz, query_times, new_features, query_boxes, query_batch_idxs, query_frame_idxs


class AdaptivePointNet2FeaturePropagator(nn.Module):
    """A single feature-propagation layer for the PointNet++ architecture.

    Used for segmentation.

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }

    Args:
        num_features (int): The number of features in the current layer.
            Note: this is the number of output features of the corresponding
            set abstraction layer.

        num_features_prev (int): The number of features from the previous
            feature propagation layer (corresponding to the next layer during
            feature extraction).
            Note: this is the number of output features of the previous feature
            propagation layer (or the number of output features of the final set
            abstraction layer, if this is the very first feature propagation
            layer)

        layer_dims (List[int]): Sizes of the MLP layer.
            Note: the first (input) dimension SHOULD NOT be included in the list,
            while the last (output) dimension SHOULD be included in the list.

        batchnorm (bool): Whether or not to use batch normalization.
    """

    def __init__(self, num_features, num_features_prev, layer_dims, batchnorm=True):
        super(AdaptivePointNet2FeaturePropagator, self).__init__()

        self.layer_dims = layer_dims

        unit_pointnets = []
        in_features = num_features + num_features_prev
        for out_features in layer_dims:
            unit_pointnets.append(
                nn.Conv1d(in_features, out_features, 1))

            if batchnorm:
                unit_pointnets.append(nn.BatchNorm1d(out_features))
            else:
                unit_pointnets.append(nn.GroupNorm(NUM_GROUPS, out_features))

            unit_pointnets.append(nn.ReLU())
            in_features = out_features

        self.unit_pointnet = nn.Sequential(*unit_pointnets)

    def forward(self, xyz, xyz_prev, features=None, features_prev=None, point2frameidx=None, query2frameidx=None):
        """
        Args:
            xyz (torch.Tensor): shape = (num_points, 3)
                The 3D coordinates of each point at current layer,
                computed during feature extraction (i.e. set abstraction).

            xyz_prev (torch.Tensor|None): shape = (num_points_prev, 3)
                The 3D coordinates of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
                This value can be None (i.e. for the very first propagator layer).

            features (torch.Tensor|None): shape = (num_features, num_points)
                The features of each point at current layer,
                computed during feature extraction (i.e. set abstraction).

            features_prev (torch.Tensor|None): shape = (num_features_prev, num_points_prev)
                The features of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).

        Returns:
            (torch.Tensor): shape = (batch_size, num_features_out, num_points)
        """
        # Run KNN to get neighbors
        device = xyz.device
        idx = torch_cluster.knn(xyz_prev, xyz, k=3, batch_x=query2frameidx, batch_y=point2frameidx)
        # pad num neighbors to always be 3
        sorted_idx, sort_mapping = torch.sort(idx[0, :], stable=True)
        first_occurances = torch.where(torch.cat([torch.ones(1).to(device), sorted_idx[1:] - sorted_idx[: -1]]))[0]
        first_occurances = sort_mapping[first_occurances]  # for each query, gives the index of first-occurance of that query-idx
        for i in range(2):
            _, num_neighbors = torch.unique(idx[0, :], return_counts=True)
            idx_needs_more = num_neighbors < 3
            idx = torch.cat([idx, idx[:, first_occurances[idx_needs_more]].clone()], dim=1)
        _, num_neighbors = torch.unique(idx[0, :], return_counts=True)
        idx = idx[:, torch.argsort(idx[0, :])]
        assert torch.all(num_neighbors == 3)

        # compute distance and indices
        dist = torch.norm(xyz[idx[0, :], :] - xyz_prev[idx[1, :], :], dim=1).view(-1, 3)
        point2query_idxs = idx[1, :].view(-1, 3)
        # Compute interpolations; requires (batch_size, num_points, 3)
        inverse_dist = 1.0 / (dist + 1e-8)
        total_inverse_dist = torch.sum(inverse_dist, dim=1, keepdim=True)
        weights = inverse_dist / total_inverse_dist
        idx = idx[1].view(-1, 3)
        new_features = three_interpolate(features_prev.transpose(0,1).contiguous().unsqueeze(0), idx.int().unsqueeze(0), weights.unsqueeze(0)).squeeze(0)
        # new_features is size (num_features_prev, num_points)

        if features is not None:
            new_features = torch.cat([new_features, features], dim=0)

        return self.unit_pointnet(new_features.unsqueeze(0)).squeeze(0).contiguous()

    def get_num_features_out(self):
        return self.layer_dims[-1]