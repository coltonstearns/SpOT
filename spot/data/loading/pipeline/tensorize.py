import torch

class TensorizePoints:

    def __init__(self, seq_len, num_pts, device):
        self.seq_len = seq_len
        self.num_pts = num_pts
        self.device = device

    def points2tensor(self, proposals, points):
        # create dense array of sequence points
        tensor_points, tensor_gtseg, point2frameidx = self._tensorize_points(points['points'], points['gt_segmentation'], proposals['hasframe_mask'], proposals['haslabel_mask'])
        tensor_timesteps = self._tensorize_timesteps(proposals['timesteps'], point2frameidx.long())
        tensor_points = torch.cat([tensor_points, tensor_timesteps.unsqueeze(-1)], dim=1).float()

        # replace old point info with tensor point info
        points['points'] = tensor_points
        points['gt_segmentation'] = tensor_gtseg
        points['point2frameidx'] = point2frameidx.long()
        return points

    def _tensorize_timesteps(self, timesteps, point2frameidx):
        tensor_timesteps = timesteps[point2frameidx].to(self.device)
        return tensor_timesteps

    def _tensorize_points(self, points, gt_segmentation, hasframe_mask, haslabel_mask):
        # sample and populate points for existing frames
        if len(points) == 0:
            return torch.zeros(0, 3).to(self.device), torch.tensor([]).to(self.device), torch.tensor([]).to(self.device, dtype=torch.long)
        points, point_idxs = random_downsample(points, self.num_pts)
        tensor_pnts = torch.cat(points, dim=0).to(self.device)
        point2frameidx = torch.cat([torch.ones(points[i].size(0)) * i for i in range(len(points))]).to(self.device)

        # expand label-mask to entire sequence
        haslabel_mask_expanded = torch.zeros(hasframe_mask.size(0), dtype=torch.bool).to(self.device)
        haslabel_mask_expanded[hasframe_mask] = haslabel_mask

        # format GT segmentation
        gtseg_samplingidxs = [point_idxs[j] for j in range(len(point_idxs)) if haslabel_mask_expanded[j]]
        if len(gt_segmentation) > 0:
            valid_gtseg, _ = random_downsample(gt_segmentation, self.num_pts, indices=gtseg_samplingidxs)
            tensor_gtseg = torch.cat(valid_gtseg, dim=0).to(self.device)
        else:
            tensor_gtseg = torch.tensor([]).to(self.device)

        return tensor_pnts, tensor_gtseg, point2frameidx


class PointCanonicalizer:

    def __init__(self, seq_len, num_pts, num_size_groupings, device):
        self.seq_len = seq_len
        self.num_pts = num_pts
        self.num_size_groupings = num_size_groupings
        self.device = device

    def canonicalize_points(self, proposals, points):
        # First, run on full sequence
        input_nocs = self._compute_nocs(proposals['input_boxes'], points['points'], points['point2frameidx'], proposals['hasframe_mask'])
        proposal_nocs = self._compute_nocs(proposals['proposal_boxes'], points['points'], points['point2frameidx'], proposals['hasframe_mask'])

        # Compute GT boxes and points (which uses a different masking)
        gt_pointmask = proposals['haslabel_mask'][points['point2frameidx']]
        gt_points = points['points'][gt_pointmask]
        _, gt_point2frameidx = torch.unique(points['point2frameidx'][gt_pointmask], return_inverse=True)
        gt_nocs = self._compute_nocs(proposals['gt_boxes'], gt_points, gt_point2frameidx, proposals['haslabel_mask'])

        # create tensorized output (doesn't overwrite anything)
        canonical_points = {"gt_nocs": gt_nocs, "input_nocs": input_nocs, "proposed_nocs": proposal_nocs}
        points = {**points, **canonical_points}
        return points

    def _compute_nocs(self, boxes, tensor_points, point2frameidx, box_frame_mask):
        # remove invalid boxes
        if torch.sum(box_frame_mask) == 0 or tensor_points.size(0) == 0:
            return torch.zeros(0, 4).to(self.device)

        # update centers to match number of point-frames
        nocs_pts = boxes.box_canonicalize(tensor_points, point2frameidx)
        return nocs_pts


def sample_to_fixed_num_points(tensors, target_length, indices=None):
    """
    Each tensor must be 2-dimensional, with target_length referring to dim=0
    Args:
        tensors: list of tensors to fill to target_length in dim=0. Cannot be an empty list!
        target_length:
        indices: list of tensors with idx to use

    Returns: stacked tensor of size (len(tensors), target_length, input_dim1)

    """
    if len(tensors) == 0:
        raise RuntimeError("Input list of tensors is empty!")

    sampled_tensors = []
    idxs = []
    for i, tensor in enumerate(tensors):
        npoints = tensor.size()[0]
        if npoints < target_length:
            if indices is not None:
                sampled_tensor, sample_idxs = buffer_tensor(tensor, target_length, indices[i])
            else:
                sampled_tensor, sample_idxs = buffer_tensor(tensor, target_length, None)
        elif npoints > target_length:
            if indices is not None:
                sample_idxs = indices[i]
            else:
                perm = torch.randperm(npoints)
                sample_idxs = perm[:target_length]
            sampled_tensor = tensor[sample_idxs, :]
        else:
            sample_idxs = torch.arange(npoints).to(tensor.device)
            sampled_tensor = tensor

        sampled_tensors.append(sampled_tensor)
        idxs.append(sample_idxs)

    sampled_tensors = torch.stack(sampled_tensors, dim=0)
    return sampled_tensors, idxs

def random_downsample(tensors, target_length, indices=None):
    if len(tensors) == 0:
        raise RuntimeError("Input list of tensors is empty!")

    sampled_tensors = []
    idxs = []
    for i, tensor in enumerate(tensors):
        npoints = tensor.size()[0]
        if npoints > target_length:
            if indices is not None:
                sample_idxs = indices[i]
            else:
                perm = torch.randperm(npoints)
                sample_idxs = perm[:target_length]
            sampled_tensor = tensor[sample_idxs, :]
            sampled_tensors.append(sampled_tensor)
            idxs.append(sample_idxs)
        else:
            sampled_tensors.append(tensor)
            idxs.append(None)
    return sampled_tensors, idxs



def buffer_tensor(tensor, target_length, indices=None):
    num_repeats = target_length // tensor.size(0)
    buffered_tensor = tensor.repeat(num_repeats, 1)
    if indices is not None and indices.size(0) > 0:
        sample_idxs = indices
    else:
        perm = torch.randperm(tensor.size(0))
        sample_idxs = perm[:target_length % tensor.size(0)]

    buffered_tensor = torch.cat((buffered_tensor, tensor[sample_idxs, :]), dim=0)
    return buffered_tensor, sample_idxs
