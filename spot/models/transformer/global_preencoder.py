
import torch.nn as nn
from spot.models.transformer.helpers import GenericMLP
import torch
from spot.models.transformer.position_embedding import PositionEmbeddingCoordsSine
from spot.io.globals import TRANSFORMER_POSITIONAL_ENCODING_SCALES


class QueryGlobalPositionalEncoding(nn.Module):

    def __init__(self, object_type, encoder_dim=256, position_embedding="fourier", time_window=1.5,
                 posenc_style="add-t-concat-xyz", posenc_use_boxfeats=False):
        super().__init__()
        self.object_type = object_type
        self.time_window = time_window
        self.posenc_style = posenc_style
        self.posenc_use_boxfeats = posenc_use_boxfeats

        # set up positional encoding for XYZT
        if self.posenc_style == "add-xyzt":
            self.d_in = 9 if self.posenc_use_boxfeats else 4
            self.pos_embedding_xyzt = PositionEmbeddingCoordsSine(d_pos=encoder_dim, pos_type=position_embedding, normalize=True, d_in=self.d_in)

        elif self.posenc_style == "add-xyz-and-t" and self.posenc_use_boxfeats:
            self.d_in = 8 if self.posenc_use_boxfeats else 3
            self.pos_embedding_xyz = PositionEmbeddingCoordsSine(d_pos=encoder_dim//2, pos_type=position_embedding, normalize=True, d_in=self.d_in)
            self.pos_embedding_t = PositionEmbeddingCoordsSine(d_pos=encoder_dim//2, pos_type=position_embedding, normalize=True, d_in=1)

        elif self.posenc_style == "add-t-concat-xyz":
            self.pos_embedding_t = PositionEmbeddingCoordsSine(d_pos=encoder_dim, pos_type=position_embedding, normalize=True, d_in=1)
            self.d_in = 8 if self.posenc_use_boxfeats else 3
            self.pos_embedding_xyz = GenericMLP(
                input_dim=self.d_in,
                hidden_dims=[encoder_dim // 2, encoder_dim // 2],
                output_dim=encoder_dim // 2,
                norm_fn_name="ln",
                activation="relu",
                use_conv=True)

        elif self.posenc_style == "concat-xyzt":
            self.d_in = 9 if self.posenc_use_boxfeats else 4
            self.pos_embedding_xyzt = GenericMLP(
                input_dim=self.d_in,
                hidden_dims=[encoder_dim // 2, encoder_dim // 2],
                output_dim=encoder_dim // 2,
                norm_fn_name="ln",
                activation="relu",
                use_conv=True)
        else:
            raise RuntimeError("Invalid posenc_style %s. Must be one of ['add-xyzt', 'add-xyz-and-t', 'add-t-concat-xyz', 'concat-xyzt']")

    def forward(self, query_xyz, query_times, query_boxes):
        if query_boxes is None:
            assert not self.posenc_use_boxfeats

        # add positional embeddings: takes bsize, npoints, 4  --> batch x channel x npoint
        xyz_rescale = TRANSFORMER_POSITIONAL_ENCODING_SCALES[self.object_type]
        input_rescale = torch.tensor([self.time_window, xyz_rescale, xyz_rescale, xyz_rescale, xyz_rescale, xyz_rescale, xyz_rescale, 1.0, 1.0]).to(query_xyz)

        if self.posenc_style == "add-xyzt":
            query_xyzt = torch.cat([query_times, query_xyz], dim=2)
            if self.posenc_use_boxfeats:
                query_xyzt = torch.cat([query_xyzt, query_boxes], dim=2)
            pos_encoding = self.pos_embedding_xyzt(query_xyzt, input_rescale=input_rescale[:self.d_in])
            cat_encoding = torch.zeros((pos_encoding.size(0), 0, pos_encoding.size(2))).to(pos_encoding)  # B x num_feats x num_queries

        elif self.posenc_style == "add-xyz-and-t" and self.posenc_use_boxfeats:
            if self.posenc_use_boxfeats:
                query_xyz = torch.cat([query_xyz, query_boxes], dim=2)
            xyz_pos_encoding = self.pos_embedding_xyz(query_xyz, input_rescale=input_rescale[1:self.d_in+1])
            t_pos_encoding = self.pos_embedding_t(query_times, input_rescale=input_rescale[:1])
            pos_encoding = torch.cat([xyz_pos_encoding, t_pos_encoding], dim=1)
            cat_encoding = torch.zeros((pos_encoding.size(0), 0, pos_encoding.size(2))).to(pos_encoding)  # B x num_feats x num_queries

        elif self.posenc_style == "add-t-concat-xyz":
            if self.posenc_use_boxfeats:
                query_xyz = torch.cat([query_xyz, query_boxes], dim=2)
            cat_encoding = self.pos_embedding_xyz(query_xyz.transpose(1, 2).contiguous())
            pos_encoding = self.pos_embedding_t(query_times, input_rescale=input_rescale[:1])

        else:  # self.posenc_style == "concat-xyzt":
            query_xyzt = torch.cat([query_times, query_xyz], dim=2)
            if self.posenc_use_boxfeats:
                query_xyzt = torch.cat([query_xyzt, query_boxes], dim=2)
            cat_encoding = self.pos_embedding_xyzt(query_xyzt.transpose(1, 2).contiguous())
            pos_encoding = torch.zeros((cat_encoding.size(0), 1, cat_encoding.size(2))).to(cat_encoding)

        return pos_encoding, cat_encoding

