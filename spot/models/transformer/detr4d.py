import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from typing import List
from spot.models.transformer.helpers import GenericMLP
from spot.models.transformer.transformer import (TransformerEncoder, TransformerEncoderLayer)
NUM_GROUPS = 16 # for group norm
import seaborn
from matplotlib import pyplot as plt
import wandb
from third_party.pointnet2.pointnet2_utils import three_interpolate, grouping_operation
import spot.models.transformer.adaptive_pointnet2 as adaptive_pointnet2
from spot.models.transformer.global_preencoder import QueryGlobalPositionalEncoding
from spot.ops.pc_util import group_first_k_values


class Model4DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-model_weights
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    """

    def __init__(self, pre_encoder, encoder, object_type, encoder_dim=256, out_dim=256, position_embedding="fourier",
                 time_window=1.5, posenc_style="add-t-concat-xyz", use_attention=True, debug_mode=False):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.encoder_dim = encoder
        self.out_dim = out_dim
        self.object_type = object_type
        self.time_window = time_window
        self.posenc_style = posenc_style
        self.use_attention = use_attention
        self.debug_mode = debug_mode
        hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=encoder_dim,
            hidden_dims=hidden_dims,
            output_dim=out_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )

        self.set_interpolator = adaptive_pointnet2.AdaptivePointNet2FeaturePropagator(num_features=0,
                                                           num_features_prev=out_dim,
                                                           layer_dims=[out_dim, out_dim],
                                                           batchnorm=True)

        self.pos_encoder = QueryGlobalPositionalEncoding(object_type, encoder_dim, position_embedding, time_window,
                                                         posenc_style, posenc_use_boxfeats=True)


    def run_encoder(self, xyzt, features, boxes, frame2batchidx, point2frameidx):
        """
        Args:
            point_clouds: Tensor size (BTN, 4)
            boxes: Tensor size (BTN, 5) for xyzsincos encoding of per-point bounding box assignments

        Returns:

        """
        BTN, BT, B = point2frameidx.size(0), frame2batchidx.size(0), torch.max(frame2batchidx).item() + 1
        xyz, times = xyzt[:, 0:3], xyzt[:, 3:4]
        query_xyz, query_times, query_features, query_boxes, query2batchidx, query2frameidx = self.pre_encoder(xyz, times, features, boxes, point2frameidx, frame2batchidx, use_boxes=True)
        # pre_enc_xyz is 1 x BTQ x 3; query_features is 1 x mlp[-1] x BTQ

        # self.debug_mode = True
        if self.debug_mode:
            self.viz_detr_inputs(xyz, times, query_xyz.squeeze(0).clone(), frame2batchidx, point2frameidx, query2batchidx, query2frameidx)

        # compute positional encoding
        if self.use_attention:
            glob_pos_encoding, glob_cat_encoding = self.pos_encoder(query_xyz, query_times, query_boxes)
            query_features = torch.cat([query_features, glob_cat_encoding], dim=1)
            query_features += glob_pos_encoding

        # Reformat into per-sequence processing to B x mlp[-1] x max-query
        _, counts = torch.unique(query2batchidx, return_counts=True)
        query_idxs = torch.arange(query_xyz.size(1)).to(xyzt.device)
        batch_query_feature_idxs, batch_query_mask = group_first_k_values(query_idxs, query2batchidx, k=torch.max(counts).item())

        batch_query_feature_idxs = batch_query_feature_idxs.unsqueeze(0).int().to(query_features.device)
        batch_query_features = grouping_operation(query_features, batch_query_feature_idxs).squeeze(0)  # returns (mlp[-1], B, max-query)
        orig_query_idxs = batch_query_feature_idxs.squeeze(0)[batch_query_mask].long()
        assert torch.all(torch.sort(orig_query_idxs)[0] == torch.arange(orig_query_idxs.size(0)).to(orig_query_idxs))
        # visualize our transformer input

        # Format input for Transformer: nquery x batch x channel features
        if self.use_attention:
            batch_query_features = batch_query_features.permute(2, 1, 0)
            _, batch_enc_features, _ = self.encoder(batch_query_features, src_key_padding_mask=~batch_query_mask)  # is (max-query, B, mlp[-1])
            batch_enc_features = batch_enc_features.permute(2, 1, 0).contiguous() # returns (mlp[-1], B, max-query)
        else:
            batch_enc_features = batch_query_features

        # Format encoded features back into queries
        enc_features = batch_enc_features[:, batch_query_mask]  # size (mlp[-1] , BTQ)

        # Re-order encoded features to match original query order
        ordered_enc_feats = torch.zeros(enc_features.size()).to(enc_features)
        ordered_enc_feats[:, orig_query_idxs] = enc_features
        ordered_enc_feats = ordered_enc_feats.contiguous()

        return query_xyz, ordered_enc_feats, query_times, query2batchidx, query2frameidx

    def forward(self, xyzt, point_features, box_features, frame2batchidx, point2frameidx):
        """

        Args:
            point_clouds: Size BTNx4 tensor

        Returns:

        """
        # get encoded features
        query_xyz, enc_features, query_times, query2batchidx, query2frameidx = self.run_encoder(xyzt, point_features, box_features, frame2batchidx, point2frameidx)
        # query_xyz is 1 x BTQ x 3; enc_features is mlp[-1] x BTQ; query_times is 1 x BTQ x 1; query2batchidx is (BTQ,) \in [0, B-1]

        # project back into output-space size
        enc_features = self.encoder_to_decoder_projection(enc_features.unsqueeze(0))  # (1, mlp[-1], BTQ) --> (1, out_feat, BTQ)

        # per-point-feats is size (num_features_out, BTN)
        per_point_feats = self.set_interpolator(xyz=xyzt[:, :3],
                                                xyz_prev=query_xyz.squeeze(0),
                                                features=None,
                                                features_prev=enc_features.squeeze(0).transpose(0, 1).contiguous(),  # BTQ x out_feat
                                                point2frameidx=point2frameidx,
                                                query2frameidx=query2frameidx)
        return per_point_feats, enc_features

    def viz_detr_inputs(self, point_clouds, times, query_xyz, frame2batchidx, point2frameidx, query2batchidx, query2frameidx):
        """

        Args:
            point_clouds: (B*T*N, d)
            query_xyz: (B*T*N', 3)
            frame_assigments: (B*T*N, 1)
            frame2batchidx: (B*T, N)
            point_assignments
            point2frameidx

        Returns:

        """
        BTN, BT, B = point2frameidx.size(0), frame2batchidx.size(0), torch.max(frame2batchidx).item() + 1

        for i in range(B):
            # get queries belonging to this sequence
            seq_queries = query_xyz[query2batchidx == i]
            seq_query2frameidx = query2frameidx[query2batchidx == i].clone()
            _, seq_query2frameidx = torch.unique(seq_query2frameidx, return_inverse=True)
            _, query_counts = torch.unique(query2frameidx, return_counts=True)
            max_queries = torch.max(query_counts).item()

            # get points belonging to this sequence
            frameidxs = torch.where(frame2batchidx == i)[0]
            pointidxs = torch.zeros(point_clouds.size(0), dtype=torch.bool).to(point2frameidx.device)
            for frameidx in frameidxs:
                pointidxs = pointidxs | (point2frameidx == frameidx)
            seq_pc_orig = point_clouds[pointidxs, :3]
            times_orig = times[pointidxs]
            _, seq_point2frameidx = torch.unique(point2frameidx[pointidxs], return_inverse=True)

            # Add displacement based on frame idx
            seq_queries = seq_queries.double() + torch.tensor([[5, 0, 0]]).to(seq_queries) * seq_query2frameidx.view(-1, 1)
            seq_pc = seq_pc_orig.double() + torch.tensor([[5, 0, 0]]).to(seq_pc_orig) * seq_point2frameidx.view(-1, 1)

            # Get point-to-query assignments
            point2query_dists = torch.cdist(seq_pc.double(), seq_queries.double())  # 10 x N x N'
            toquery_dists, query_idxs = torch.min(point2query_dists, dim=1)  # 10 x N
            query_idxs = query_idxs
            query_idxs[toquery_dists > self.pre_encoder.radius] = -1
            viz_pnts, pnt_clrs = viz_pnts_query_clrs(seq_pc, query_idxs, max_queries)

            # also visualize actual sequence
            viz_seq = seq_pc_orig[:, :3].to('cpu').data.numpy().reshape(-1, 3)
            seq_cmap = plt.get_cmap("GnBu")
            seq_clrs = times_orig.flatten() / torch.max(times_orig).item()
            seq_clrs = seq_clrs.to('cpu').data.numpy()
            seq_clrs = seq_cmap(seq_clrs)[:, :3] * 255
            viz_seq += np.array([[-5, 0, 0]])

            # visualize query points a full balls
            # generate colored detection centers
            gauss_ball = torch.randn((1, 300, 3))
            gauss_ball = gauss_ball / torch.norm(gauss_ball, dim=2, keepdim=True) / 10
            query_centers = (seq_queries.view(-1, 1, 3) + gauss_ball.to(seq_queries)).view(-1, 3)
            qcenter_assigns = torch.arange(seq_queries.size(0)).repeat_interleave(300)
            viz_queries, query_clrs = viz_pnts_query_clrs(query_centers, qcenter_assigns, num_clrs=1)

            # combine into wandb viz
            wandb_pc = np.hstack([np.vstack([viz_pnts, viz_seq, viz_queries]), np.vstack([pnt_clrs, seq_clrs, query_clrs])])
            wandb.log({"Transformer Debug Inputs": wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": np.array([])})})


def viz_pnts_query_clrs(point_sequence, query_idxs, num_clrs):
    query_idxs = query_idxs.to('cpu').data.numpy()
    clrs = seaborn.color_palette(palette='husl', n_colors=num_clrs)
    clrs.append((0, 0, 0))
    clrs = np.array([list(clr) for clr in clrs])
    query_idxs[query_idxs != -1] = query_idxs[query_idxs != -1] % num_clrs
    per_point_clrs = clrs[query_idxs.flatten()] * 255
    viz_pnts = point_sequence[:, :3].to('cpu').data.numpy()
    viz_pnts = viz_pnts.reshape(-1, 3)
    return viz_pnts, per_point_clrs


def build_preencoder(transformer_args, num_feats, object_class):
    encoder_dim = transformer_args['enc_dim']
    if transformer_args["posenc_style"] in ["add-t-concat-xyz", "concat-xyzt"] and transformer_args["use_attention"]:
        encoder_dim = encoder_dim // 2
    mlp_dims = [num_feats, 64, 128, encoder_dim]
    preencoder = adaptive_pointnet2.AdaptiveBatchPointnetSAModule(
        radius=transformer_args['radius'][object_class],  # ball query radius
        nsample=64,  # max number of points per ball-query
        npoint=transformer_args['fps_max_points'],  # number of points in FPS
        mlp=mlp_dims,
        normalize_xyz=True,
        fps_sample_ratio=transformer_args['fps_sample_ratio'],
        fps_lower_thresh=transformer_args['fps_min_points']
    )

    return preencoder


def build_encoder(transformer_args):
    encoder_layer = TransformerEncoderLayer(
        d_model=transformer_args["enc_dim"],
        nhead=transformer_args["enc_nhead"],
        dim_feedforward=transformer_args["enc_ffn_dim"],
        dropout=transformer_args["enc_dropout"],
        activation=transformer_args["enc_activation"],
    )
    encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=transformer_args["enc_nlayers"]
    )
    return encoder


def build_4detr(transformer_args, preencode_in_dim, out_dim, object_class, debug_mode=False):
    pre_encoder = build_preencoder(transformer_args, preencode_in_dim, object_class)
    encoder = build_encoder(transformer_args)
    model = Model4DETR(pre_encoder,
                       encoder,
                       encoder_dim=transformer_args["enc_dim"],
                       out_dim=out_dim,
                       position_embedding="fourier",
                       object_type=object_class,
                       posenc_style=transformer_args["posenc_style"],
                       use_attention=transformer_args["use_attention"],
                       debug_mode=debug_mode)
    return model
