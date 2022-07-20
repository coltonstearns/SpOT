'''
This PointNet implementation was adapted from:
https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py

'''


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.glob.glob import global_max_pool


class PointNetBase(nn.Module):
    '''
    Simple PointNet that extracts point-wise feature by concatenating local and global features.
    Uses group norm instead of batch norm.
    '''
    def __init__(self, input_dim=3, feat_size=1024, layer_sizes=[64, 128], batchnorm=False):
        super(PointNetBase, self).__init__()
        self.output_size = feat_size
        self.input_dim = input_dim
        self.conv1 = torch.nn.Conv1d(self.input_dim, layer_sizes[0], 1)
        self.conv2 = torch.nn.Conv1d(layer_sizes[0], layer_sizes[1], 1)
        self.conv3 = torch.nn.Conv1d(layer_sizes[1], self.output_size, 1)

        # concatenation features
        self.compress_glob = torch.nn.Conv1d(self.output_size, 3*self.output_size//4, 1)
        self.compress_perpnt = torch.nn.Conv1d(layer_sizes[1], self.output_size//4, 1)

        if not batchnorm:
            self.bn1 = nn.GroupNorm(16, layer_sizes[0])
            self.bn2 = nn.GroupNorm(16, layer_sizes[1])
            self.bn3 = nn.GroupNorm(16, self.output_size)
        else:
            self.bn1 = nn.BatchNorm1d(layer_sizes[0])
            self.bn2 = nn.BatchNorm1d(layer_sizes[1])
            self.bn3 = nn.BatchNorm1d(self.output_size)

    def forward(self, x, point2batchidx):
        print("Calling Abstract Class Method! Instead, must use child of PointNetBase.")


class PointNet4D(PointNetBase):
    def __init__(self,  input_dim=3, feat_size=1024, layer_sizes=[64, 128], batchnorm=False):
        super(PointNet4D, self).__init__(input_dim, feat_size, layer_sizes, batchnorm)

    def forward(self, x, point2batchidx):
        B, _, n_pts = x.size()

        # Process 4D point cloud
        x = F.relu(self.bn1(self.conv1(x)))
        temporal_pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        intermediate_feats = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(B, -1, n_pts)
        temporal_pointfeat = temporal_pointfeat.view(B, -1, n_pts)

        # get per-point segmentation features
        per_pnt_feats = self.compress_perpnt(intermediate_feats)
        global_feat = global_max_pool(x.squeeze(0).transpose(0, 1).contiguous(), point2batchidx)
        global_feat_repeated = global_feat[point2batchidx].transpose(0, 1).contiguous().unsqueeze(0)  # B x -1 x n_pts
        global_feat_repeated = self.compress_glob(global_feat_repeated)
        x = torch.cat([per_pnt_feats, global_feat_repeated], dim=1)

        return x, temporal_pointfeat



class PointNetFrame3D(PointNetBase):
    def __init__(self, input_dim=3, out_size=512, layer_sizes=[64, 128], batchnorm=False):
        super(PointNetFrame3D, self).__init__(input_dim, out_size, layer_sizes, batchnorm)

    def forward(self, x):
        """
        x: size (B, feats, T*N)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x


class PointAttentionLayer(nn.Module):
    def __init__(self, in_feats=1024, num_heads=4):
        super(PointAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attend_layer = torch.nn.Conv1d(in_feats*2, self.num_heads, 1)
        self.sigmoid_layer = torch.nn.Sigmoid()

    def forward(self, x):
        B, _, n_pts = x.size()

        pooled_feat = torch.max(x, 2, keepdim=True)[0].repeat(1, 1, n_pts)
        attention_feats = torch.cat([pooled_feat, x], dim=1)
        mh_attention = self.sigmoid_layer(self.attend_layer(attention_feats))
        x = x.repeat(1, self.num_heads, 1).view(B, -1, self.num_heads, n_pts) * mh_attention.view(B, 1, self.num_heads, n_pts)
        return x, mh_attention
