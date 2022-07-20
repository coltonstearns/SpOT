import numpy as np
import torch
from torch_geometric.nn.glob.glob import global_mean_pool, global_max_pool
from torch_scatter.scatter import scatter_min, scatter_max
from spot.ops.pc_util import group_first_k_values
from spot.data.box3d import LidarBoundingBoxes
from third_party.mmdet3d.ops.iou3d.iou3d_utils import nms_gpu, boxes_iou_bev
from spot.ops.transform_utils import lidar2bev_box
from torch_geometric.data.data import Data as GeometricData
from spot.ops.torch_utils import is_sorted


class BaseSTObjectRepresentations:

    OBJECT_ATTRS = ['global2locals', 'global2currentego']
    FRAME_ATTRS = ['boxes', 'confidences', 'timesteps', 'instance_ids', 'gt_boxes', 'with_gt_mask', 'gt_classification']
    POINT_ATTRS = ['points', 'gt_segmentation']

    def __init__(self, object_data, frame_data, point_data, point2frameidx, frame2objectidx, ref_frame="local"):
        """
        DOES NOT ASSUME THAT ALL BOXES HAVE AT LEAST ONE POINT-OBSERVATION
        WITHIN THEM.
        Args:
            affines (torch.tensor): specifies
        """
        required_keys = ['global2locals', 'global2currentego', 'boxes', 'confidences', 'instance_ids', 'timesteps', 'gt_boxes', 'with_gt_mask',
                         'points', 'gt_segmentation', 'gt_classification']
        for required_key in required_keys:
            assert required_key in object_data or required_key in frame_data or required_key in point_data
        assert isinstance(frame_data['boxes'], LidarBoundingBoxes)
        assert frame_data['boxes'].size(0) == frame2objectidx.size(0) and frame_data['timesteps'].size(0) == frame2objectidx.size(0) and frame_data['confidences'].size(0) == frame2objectidx.size(0)
        assert object_data['global2locals'].dtype == torch.double
        assert object_data['global2currentego'].dtype == torch.double
        assert frame_data['timesteps'].dtype == torch.double

        # object-level info
        self.object_data = object_data

        # frame-level info
        self.frame_data = frame_data
        self.frame2batchidx = frame2objectidx

        # point-level info
        self.point_data = point_data
        self.point2frameidx = point2frameidx
        self.point2batchidx = frame2objectidx[point2frameidx]

        # other info
        self.device = self.point_data['points'].device
        self.size_clusters = self.frame_data['boxes'].size_anchors
        self.reference_frame = ref_frame

    @property
    def global2locals(self):
        return self.object_data['global2locals']

    @property
    def global2currentego(self):
        return self.object_data['global2currentego']

    @property
    def boxes(self):
        return self.frame_data['boxes']

    @property
    def confidences(self):
        return self.frame_data['confidences']

    @property
    def instance_ids(self):
        return self.frame_data['instance_ids']

    @property
    def timesteps(self):
        return self.frame_data['timesteps']

    @property
    def gt_boxes(self):
        return self.frame_data['gt_boxes']

    @property
    def with_gt_mask(self):
        return self.frame_data['with_gt_mask']

    @property
    def points(self):
        return self.point_data['points']

    @property
    def gt_segmentation(self):
        return self.point_data['gt_segmentation']

    def change_basis_local2global(self):
        assert self.reference_frame == "local"
        out = self.coordinate_transfer(self.global2locals, inverse=True)
        out.reference_frame = "global"
        return out

    def change_basis_global2local(self):
        assert self.reference_frame == "global"
        out = self.coordinate_transfer(self.global2locals, inverse=False)
        out.reference_frame = "local"
        return out

    def change_basis_currentego2global(self):
        assert self.reference_frame == "ego"
        out = self.coordinate_transfer(self.global2currentego, inverse=True)
        out.reference_frame = "global"
        return out

    def change_basis_global2currentego(self):
        assert self.reference_frame == "global"
        out = self.coordinate_transfer(self.global2currentego, inverse=False)
        out.reference_frame = "ego"
        return out

    def coordinate_transfer(self, affines, pre_translations=None, time_offset=None, inverse=False):
        """
        Transform this object representation into a new reference frame.
        Args:
            affines (torch.tensor): Affine transformation of size (B, 4, 4)
            pre_translations (torch.tensor): Optional transformation to apply BEFORE the affine transformation.
            inverse (bool): whether to transform by the original or the inverse matrix
            time_offset (double): change of bases in timeline
        Returns: a new STObjectRepresentation

        """
        point_data, frame_data = self._coordinate_transfer(affines, pre_translations, time_offset, inverse)

        return type(self)(self.object_data, frame_data, point_data, self.point2frameidx, self.frame2batchidx, self.reference_frame)

    def _coordinate_transfer(self, affines, pre_translations, time_offset, inverse):
        # transform points
        points = self.affine_transform_points(self.point_data['points'], self.point2batchidx, affines, pre_translations, inverse)
        point_data = {**self.point_data.copy(), **{'points': points}}

        # transform timesteps
        timesteps = self.frame_data['timesteps'].clone()
        if time_offset is not None:
            timesteps += time_offset.double()
        transformed = {"timesteps": timesteps}

        # transform boxes
        transform_box_keys = ['boxes'] if "gt_boxes" not in self.frame_data else ["boxes", "gt_boxes"]
        for k in transform_box_keys:
            boxes = self.frame_data[k].clone()
            if pre_translations is not None:
                boxes = boxes.translate(pre_translations, batch=self.frame2batchidx)
            boxes = boxes.coordinate_transfer(affines, batch=self.frame2batchidx, inverse=inverse)
            transformed[k] = boxes

        # create frame data
        frame_data = {**self.frame_data.copy(), **transformed}

        return point_data, frame_data

    def to_base_rep(self):
        return self

    def estimate_boxes(self, eval_times, relative_time_context=0.4, time_context=1.5):
        """

        Args:
            eval_times (torch.DoubleTensor): a tensor of timesteps for which to compute our best estimate of each object's bounding box
            time_context (float): the time-window to look over for any interpolation

        Returns:

        """
        assert eval_times.dtype == torch.double

        # convert boxes + times to padded-batch representation
        batched_boxes, with_value_mask = self.to_batch_padded(self.frame_data['boxes'].box, self.frame2batchidx)
        batched_timesteps, _ = self.to_batch_padded(self.frame_data['timesteps'], self.frame2batchidx)

        B, Tmax, _ = batched_boxes.size()
        Tevals = eval_times.size(0)
        device = batched_boxes.device

        # format boxes + timesteps
        batched_boxes = batched_boxes.repeat(1, Tevals, 1).view(B * Tevals, Tmax, -1)

        # Set up preliminary variables for mask computation
        batched_boxes = batched_boxes.view(B*Tevals, Tmax, -1)
        time_diffs = torch.abs(batched_timesteps.view(B, 1, Tmax) - eval_times.view(1, -1, 1))  # size (B, Tevals, Tmax)
        time_diffs[~with_value_mask.unsqueeze(1).repeat(1, Tevals, 1)] = np.inf
        closest_timeidxs = torch.argmin(time_diffs.view(B*Tevals, Tmax), dim=1)
        anchor_ts = batched_timesteps[torch.arange(B).repeat_interleave(Tevals), closest_timeidxs]
        anchor_time_diffs = torch.abs(batched_timesteps.view(B, 1, Tmax) - anchor_ts.view(B, Tevals, 1))  # size (B, Tevals, Tmax)

        # 1. set up mask for closest observations
        anchor_box_mask = torch.zeros(B*Tevals, Tmax).bool().to(device)
        anchor_box_mask[torch.arange(B * Tevals), closest_timeidxs] = True

        # 2. set up mask for observations close enough to desired eval timestep
        absolute_proximity_mask = (time_diffs <= time_context).view(B * Tevals, Tmax)

        # 3. set up mask for observations close enough to anchor box (for smoothened motion estimate)
        relative_proximity_mask = (anchor_time_diffs <= relative_time_context).view(B * Tevals, Tmax)

        # 4. set up mask if eval-times are observed vs need interpolation: True if needed, False otherwise
        need_interpolation_mask = torch.ones(B*Tevals, Tmax).bool().to(device)
        need_interpolation_mask *= time_diffs.view(B * Tevals, Tmax)[torch.arange(B * Tevals), closest_timeidxs].view(-1, 1) > 0.02

        # Finally, combine all masks to get a list of bounding boxes for interpolation
        mask = (absolute_proximity_mask & relative_proximity_mask & need_interpolation_mask) | anchor_box_mask

        # mean-center sequences (for numerical stability)
        centers = batched_boxes[:, :, :3].clone() * mask.unsqueeze(-1)
        center_means = torch.sum(centers, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1)
        centers = centers - center_means
        velocities = batched_boxes[:, :, 7:10]

        # normalize bounding-box times (for numerical stability)
        batched_timesteps = batched_timesteps.repeat(1, Tevals).view(B*Tevals, Tmax)
        batched_eval_times = eval_times.repeat(B).view(B*Tevals, 1)
        ts = (batched_timesteps - batched_eval_times).float()
        t_outs = torch.zeros(B*Tevals, 1).to(ts)

        # run regression
        eval_centers, eval_vels, center_parametrics = self.regression_1st_order(ts, centers, velocities, t_outs, ~mask)
        eval_centers[:, :3] += center_means.view(B*Tevals, 3)
        center_parametrics[:, :3, 0] += center_means.view(B*Tevals, 3)  # (B, 3, 2)

        # get final boxes
        eval_boxes = batched_boxes[torch.arange(B * Tevals), closest_timeidxs]
        eval_boxes[:, :3] = eval_centers
        eval_boxes[:, 7:10] = eval_vels
        eval_boxes = eval_boxes.view(B, Tevals, -1)

        # format parametrics
        yaw_parametrics = torch.stack([eval_boxes.view(B*Tevals, -1)[:, 6:7], torch.zeros(B*Tevals, 1).to(eval_boxes)], dim=2) # (B, 1, 2)
        parametrics = torch.cat([center_parametrics, yaw_parametrics], dim=1)  # (B, 4, 2)
        parametrics = parametrics.view(B, Tevals, 4, 2)

        # Format into boxes
        eval_boxes_list, parametrics_list = [], []
        for i in range(Tevals):
            eval_box = LidarBoundingBoxes(box=eval_boxes[:, i, :], box_enc=None, size_anchors=self.size_clusters)
            eval_boxes_list.append(eval_box)
            parametrics_list.append(parametrics[:, i, :, :])


        return eval_boxes_list, parametrics_list

    def get_object_confidences(self, which):
        # if float input is given
        if isinstance(which, float):
            time_diffs = torch.abs(self.frame_data['timesteps'] - which)  # size (BT,)
            size = int(self.frame2batchidx.max().item() + 1)
            closest_times, closest_idxs = scatter_min(time_diffs, self.frame2batchidx, dim=0, dim_size=size)
            confidences = self.frame_data['confidences'][closest_idxs]

        # if string input is given
        elif isinstance(which, str):
            if which == "mean":
                confidences = global_mean_pool(self.frame_data['confidences'], self.frame2batchidx)
            elif which == "max":
                confidences = global_max_pool(self.frame_data['confidences'], self.frame2batchidx)
            else:
                raise RuntimeError("Passing in a string to this method requires it be ['mean', 'max'].")
        else:
            raise RuntimeError("This method takes in a float or string object.")

        return confidences

    def get_object_timesteps(self, which="all", return_idx=False):
        if which == "all":
            return self.frame_data['timesteps']
        elif which == "min":
            size = int(self.frame2batchidx.max().item() + 1)
            min_times, idxs = scatter_min(self.frame_data['timesteps'], self.frame2batchidx, dim=0, dim_size=size)
            ret = min_times if not return_idx else (min_times, idxs)
            return ret
        elif which == "max":
            size = int(self.frame2batchidx.max().item() + 1)
            max_times, idxs = scatter_max(self.frame_data['timesteps'], self.frame2batchidx, dim=0, dim_size=size)
            ret = max_times if not return_idx else (max_times, idxs)
            return ret
        else:
            raise RuntimeError("Object timesteps 'which' must be one of ['all', 'min', 'max'].")

    def get_withpoints_mask(self):
        withpoints_mask = torch.zeros(len(self), dtype=torch.bool).to(self.device)
        object_idxs_wpoints = torch.unique(self.point2batchidx)
        withpoints_mask[object_idxs_wpoints] = True
        return withpoints_mask

    def remove_empty_frames(self):
        frame_idxs_with_points, point2frameidx = torch.unique(self.point2frameidx, return_inverse=True)
        frame_data = self.frame_data.copy()
        for k in frame_data.keys():
            frame_data[k] = self.frame_data[k][frame_idxs_with_points]
        frame2batchidx = self.frame2batchidx[frame_idxs_with_points]
        update = type(self)(self.object_data, frame_data, self.point_data, point2frameidx, frame2batchidx, self.reference_frame)
        if len(update) != len(self):
            print("----------")
            print(len(update))
            print(len(self))
        assert len(update) == len(self)
        return update

    def to_inputs_list(self):
        # NOTE, this does NOT reset point-timesteps! Also, does this have to maintain order? only batch-order...
        inputs_list = []
        for i in range(len(self)):
            points = self.point_data['points'][self.point2batchidx == i]
            point2frameidx = self.point2frameidx[self.point2batchidx == i]
            _, point2frameidx = torch.unique(point2frameidx, return_inverse=True)

            boxes = self.frame_data['boxes'][self.frame2batchidx == i]
            confidences = self.frame_data['confidences'][self.frame2batchidx == i]

            sequence_data = GeometricData(x=points, boxes=boxes, point2frameidx=point2frameidx, confidences=confidences)
            inputs_list.append(sequence_data)

        return inputs_list

    def to(self, device):
        self._to(device)
        return self

    def _to(self, device):
        for reps in [self.object_data, self.frame_data, self.point_data]:
            for k in reps:
                reps[k] = reps[k].to(device)

        # put standalone reps on device
        self.point2frameidx = self.point2frameidx.to(device)
        self.frame2batchidx = self.frame2batchidx.to(device)
        self.point2batchidx = self.point2batchidx.to(device)
        self.device = device

    def clone(self):
        object_data, frame_data, point_data = {}, {}, {}
        cloned_reps = [object_data, frame_data, point_data]
        for i, reps in enumerate([self.object_data, self.frame_data, self.point_data]):
            for k in reps:
                cloned_reps[i][k] = reps[k].clone()
        point2frameidx = self.point2frameidx.clone()
        frame2batchidx = self.frame2batchidx.clone()
        return type(self)(object_data, frame_data, point_data, point2frameidx, frame2batchidx, ref_frame=self.reference_frame)

    def clip_sequences(self, time_context, points_context=None):
        """
        Args:
            time_context: Denotes the maximum time window to allow for each ST sequence
            points_context: Denote the maximum number of points allowed in a ST sequence

        Returns:
            BaseSTObjectRepresentation with an appropriate temporal and density context
        """
        assert self.reference_frame == "global"  # needs to be to correct for
        object_data, frame_data, point_data, point2frameidx, frame_data_mask_orig, point_data_mask = self.to_padded()
        M, TNmax= len(self), point_data_mask.size(1)

        # get initial mintimes and mintime boxes
        orig_min_times, orig_mintime_idxs = self.get_object_timesteps(which="min", return_idx=True)
        orig_mintime_boxes = self.frame_data['boxes'][orig_mintime_idxs]

        # identify frames outside our temporal context
        max_times = self.get_object_timesteps(which="max")
        time_diffs = max_times.unsqueeze(1) - frame_data['timesteps']  # B x T-frames
        time_context_mask = time_diffs < time_context
        frame_data_mask = frame_data_mask_orig & time_context_mask

        # identify unnecessary frames that exceed our point context
        if points_context is not None:
            # todo: fix me
            points_per_frame = torch.sum(point_data_mask, dim=1)
            ppf_padded, ppf_mask = self.to_batch_padded(points_per_frame, self.frame2batchidx)
            ppf_padded[~ppf_mask] = 0
            cum_ppf = torch.cumsum(ppf_padded.flip(dim=1), dim=1).flip(dim=1)  # todo: assumes sorted order!
            point_context_mask = cum_ppf < points_context
            frame_data_mask &= point_context_mask

        # update point-mask based on our removed frames
        point2frame_frameselectionidxs = point2frameidx[point_data_mask]
        point2frame_objselectionidxs = torch.arange(M).unsqueeze(-1).repeat(1, TNmax)[point_data_mask]
        point_data_mask[point_data_mask.clone()] = frame_data_mask[point2frame_objselectionidxs, point2frame_frameselectionidxs]

        # convert back to dense tensor with new missing-frames
        dense_rep = self.to_dense(object_data, frame_data, point_data, point2frameidx, frame_data_mask, point_data_mask)
        object_data, frame_data, point_data, point2frameidx, frame2batchidx = dense_rep
        frame_data['boxes'] = LidarBoundingBoxes(box=frame_data['boxes'], box_enc=None, size_anchors=self.size_clusters)
        if 'gt_boxes' in frame_data:
            frame_data['gt_boxes'] = LidarBoundingBoxes(box=frame_data['gt_boxes'], box_enc=None, size_anchors=self.size_clusters)

        # mean-center translations by updating global2locals
        min_times, mintime_idxs = scatter_min(frame_data['timesteps'], frame2batchidx, dim=0, dim_size=len(self))
        mintime_boxes = frame_data['boxes'][mintime_idxs]

        # get offset in local space
        local_mintime_boxes = mintime_boxes.coordinate_transfer(object_data['global2locals'], batch=torch.arange(M).to(frame2batchidx), inverse=False)
        center_offsets_local = -local_mintime_boxes.center.double()
        object_data['global2locals'][:, :3, 3] += center_offsets_local

        # mean-center timesteps
        point_data['points'][:, 3] = self.reset_point_times(frame_data['timesteps'], point2frameidx, frame2batchidx)

        # build final representation and return
        clipped_rep = type(self)(object_data, frame_data, point_data, point2frameidx, frame2batchidx, ref_frame=self.reference_frame)
        assert len(clipped_rep) == len(self)
        return clipped_rep

    def set(self, idxs, objects):
        # get maximum count of points and frames for padded representation
        get_max_batchsize = lambda batchidxs: torch.max(torch.unique(batchidxs, return_counts=True)[1]).item()
        frames_padding = max(get_max_batchsize(self.frame2batchidx), get_max_batchsize(objects.frame2batchidx))
        points_padding = max(get_max_batchsize(self.point2batchidx) if self.point2batchidx.size(0) > 0 else 1, get_max_batchsize(objects.point2batchidx)  if objects.point2batchidx.size(0) > 0 else 1)
        # convert everything to padded form
        object_data, frame_data, point_data, point2frameidx, frame_mask, point_mask = self.to_padded(frames_padding, points_padding)
        set_object_data, set_frame_data, set_point_data, set_point2frameidx, set_frame_mask, set_point_mask = objects.to_padded(frames_padding, points_padding)
        # update our representation
        for data_dict, set_data_dict in zip([object_data, frame_data, point_data], [set_object_data, set_frame_data, set_point_data]):
            for k in data_dict:
                try:
                    data_dict[k][idxs] = set_data_dict[k]
                except RuntimeError:
                    # note: gt-segmentation has an error in that set_data_dict contains a float!
                    # print("Error setting %s" % k)
                    # print(data_dict[k].dtype)
                    # print(set_data_dict[k].dtype)
                    data_dict[k][idxs] = set_data_dict[k].to(data_dict[k])
        point2frameidx[idxs] = set_point2frameidx
        frame_mask[idxs] = set_frame_mask
        point_mask[idxs] = set_point_mask
        # convert everything back to dense form
        object_data, frame_data, point_data, point2frameidx, frame2batchidx = \
            self.to_dense(object_data, frame_data, point_data, point2frameidx, frame_mask, point_mask)
        frame_data['boxes'] = LidarBoundingBoxes(box=frame_data['boxes'], box_enc=None, size_anchors=self.size_clusters)
        if 'gt_boxes' in frame_data:
            frame_data['gt_boxes'] = LidarBoundingBoxes(box=frame_data['gt_boxes'], box_enc=None, size_anchors=self.size_clusters)

        out = type(self)(object_data, frame_data, point_data, point2frameidx, frame2batchidx, ref_frame=self.reference_frame)
        return out


    def to_padded(self, frame_pad_size=None, point_pad_size=None):
        # convert frame info to padded form
        num_objects = len(self)
        padded_frame_data, frame_data_mask = {}, None
        for k in self.frame_data:
            pad_data, pad_mask = self.to_batch_padded(self.frame_data[k], self.frame2batchidx, frame_pad_size)
            if frame_data_mask is None: frame_data_mask = pad_mask
            padded_frame_data[k] = pad_data

        # convert point info to padded form
        if self.point2batchidx.size(0) > 0:
            # compute padded reps
            point2frameidx, point_data_mask = self.to_batch_padded(self.point2frameidx, self.point2batchidx, point_pad_size, batch_size=num_objects)
            padded_point_data = {}
            for k in self.point_data:
                pad_data, _ = self.to_batch_padded(self.point_data[k], self.point2batchidx, point_pad_size, batch_size=num_objects)
                padded_point_data[k] = pad_data

            # compute frame-offsets for each batch
            batch_frame_offsets = torch.cat([torch.zeros(1).to(self.device), torch.cumsum(torch.sum(frame_data_mask, dim=1), dim=0)]).long()
            point2frameidx -= batch_frame_offsets[:-1].view(-1, 1)
        else:
            point_pad_size = 1 if point_pad_size is None else point_pad_size
            point2frameidx, point_data_mask = torch.zeros((num_objects, point_pad_size), dtype=torch.long).to(self.device), torch.zeros((num_objects, point_pad_size), dtype=torch.bool).to(self.device)
            padded_point_data = {}
            for k, val in self.point_data.items():
                if len(val.size()) == 2:
                    padded_point_data[k] = torch.zeros((num_objects, point_pad_size, self.point_data[k].size(1))).to(self.device)
                else:
                    padded_point_data[k] = torch.zeros((num_objects, point_pad_size)).to(self.device)

        return self.object_data, padded_frame_data, padded_point_data, point2frameidx, frame_data_mask, point_data_mask

    def to_dense(self, object_data_padded, frame_data_padded, point_data_padded, point2frameidx_padded, frame_mask, point_mask):
        # compute frame-offsets for each batch
        batch_frame_offsets = torch.cat([torch.zeros(1).to(self.device), torch.cumsum(torch.sum(frame_mask, dim=1), dim=0)]).long()
        padded_batch_frame_offsets = torch.arange(frame_mask.size(0), dtype=torch.long).to(self.device) * frame_mask.size(1)
        point2frameidx_padded_raw = point2frameidx_padded + padded_batch_frame_offsets.view(-1, 1)
        point2frameidx_padded += batch_frame_offsets[:-1].view(-1, 1)

        # correct for frame-offset introduced by removing frames in-the-front of the sequence
        sequence_frame_offsets = torch.cumsum(~frame_mask, dim=1).long()

        # compute frame-to-batch idxs
        frame2batchidx_padded = torch.arange(frame_mask.size(0)).repeat_interleave(frame_mask.size(1)).to(self.device)
        frame2batchidx = frame2batchidx_padded[frame_mask.flatten()]

        # merge all the data to dense form
        point_data = {k: point_data_padded[k][point_mask] for k in point_data_padded}
        point2frameidx = point2frameidx_padded[point_mask]
        point2frameidx -= sequence_frame_offsets.flatten()[point2frameidx_padded_raw[point_mask]]

        # merge frame info
        frame_data = {k: frame_data_padded[k][frame_mask] for k in frame_data_padded}

        return object_data_padded, frame_data, point_data, point2frameidx, frame2batchidx

    def __getitem__(self, idx):
        indices = convert_idx_to_tensor(idx, len(self), self.device)
        if len(indices) == 0:
            return EmptySTObjectRepresentations(self.device, self.size_clusters, self.reference_frame)
        object_data, frame_data, point_data, point2frameidx, frame2batchidx, indices = self._get_item(indices)
        return type(self)(object_data, frame_data, point_data, point2frameidx, frame2batchidx, self.reference_frame)

    def _get_item(self, indices):
        if is_sorted(indices):
            return self._get_item_sorted(indices)
            # return self._get_item_unsorted(indices)
        else:
            return self._get_item_unsorted(indices)

    def _get_item_sorted(self, indices):
        # assert that point2frameidx and frame2batchidx are sorted
        assert torch.all((self.frame2batchidx[1:] - self.frame2batchidx[:-1]) >= 0)
        assert torch.all((self.point2frameidx[1:] - self.point2frameidx[:-1]) >= 0)

        # update object-level data
        object_data = {k: self.object_data[k][indices] for k in self.object_data}

        # update frame-level data
        indices_bool = torch.zeros(len(self), dtype=torch.bool).to(self.device)
        indices_bool[indices] = True
        frame_mask = indices_bool[self.frame2batchidx]
        frame_data = {k: self.frame_data[k][frame_mask] for k in self.frame_data}

        # update frame2batch idx
        frame2batchidx_raw = self.frame2batchidx[frame_mask]
        _, frame2batchidx = torch.unique(frame2batchidx_raw, return_inverse=True)

        # build mapping from original frame indices to new frame indices
        origframeidx2newframeidx = torch.zeros(self.frame2batchidx.size(0), dtype=torch.long).to(self.device)
        origframeidx2newframeidx[frame_mask] = torch.arange(frame2batchidx.size(0)).to(frame2batchidx)

        # update point-level data
        point_mask = indices_bool[self.point2batchidx]  # mask of which points to keep
        point_data = {k: self.point_data[k][point_mask] for k in self.point_data}
        point2frameidx = origframeidx2newframeidx[self.point2frameidx[point_mask]]

        return object_data, frame_data, point_data, point2frameidx, frame2batchidx, indices

    def _get_item_unsorted(self, indices):
        # get necessary padding to conver to padded form
        get_max_batchsize = lambda batchidxs: torch.max(torch.unique(batchidxs, return_counts=True)[1]).item()
        frames_padding = get_max_batchsize(self.frame2batchidx)
        points_padding = get_max_batchsize(self.point2batchidx) if self.point2batchidx.size(0) > 0 else 1

        # convert everything to padded form
        object_data, frame_data, point_data, point2frameidx, frame_mask, point_mask = self.to_padded(frames_padding, points_padding)

        # update our representation
        object_data = {k: object_data[k][indices] for k in object_data}
        frame_data = {k: frame_data[k][indices] for k in frame_data}
        point_data = {k: point_data[k][indices] for k in point_data}
        point2frameidx = point2frameidx[indices]
        frame_mask = frame_mask[indices]
        point_mask = point_mask[indices]

        # convert everything back to dense form
        object_data, frame_data, point_data, point2frameidx, frame2batchidx = \
            self.to_dense(object_data, frame_data, point_data, point2frameidx, frame_mask, point_mask)
        frame_data['boxes'] = LidarBoundingBoxes(box=frame_data['boxes'], box_enc=None, size_anchors=self.size_clusters)
        if 'gt_boxes' in frame_data:
            frame_data['gt_boxes'] = LidarBoundingBoxes(box=frame_data['gt_boxes'], box_enc=None, size_anchors=self.size_clusters)

        return object_data, frame_data, point_data, point2frameidx, frame2batchidx, indices

    def nms(self, threshold, timestep, trajectory_params):
        """
        Runs NMS to remove any overlapping object sequences, where overlap is measured at timestep.
        Args:
            timestep: The timestep to evaluate NMS on. Can also be string value "min" or "max", which indicates minimum
                      or maximum available timestep over all object observations

        Returns:

        """
        # get timestep if min or max is specified
        if isinstance(timestep, str):
            op = min if timestep == "min" else max
            timestep = op(self.get_object_timesteps(which=timestep)).item()

        # get all boxes at the timestep
        eval_times = torch.tensor([timestep], dtype=torch.double).to(self.device)
        boxes = self.estimate_boxes(eval_times, **trajectory_params)[0][0]
        confidences = self.get_object_confidences(timestep)

        # get boxes in BEV
        scene_centering = boxes[0:1].center[0:1, :2]
        boxes_bev = lidar2bev_box(boxes.box, scene_centering)

        # run NMS on all boxes
        keep_idxs = nms_gpu(boxes_bev, confidences, thresh=threshold)
        keep_idxs = torch.sort(keep_idxs)[0]

        # compute remove idxs
        remove_idxs = torch.ones(len(self), dtype=torch.bool).to(self.device)
        remove_idxs[keep_idxs] = False
        remove_idxs = torch.where(remove_idxs)[0]
        removed_objects = self[remove_idxs]

        # filter out from each object-representation
        out = self[keep_idxs]
        return out, keep_idxs, removed_objects

    def __delitem__(self, idx):
        raise RuntimeError("Object Representation does not support deleting items.")

    def __setitem__(self, key, value):
        raise RuntimeError("Object Representation does not support setting items.")

    def __len__(self):
        try:
            return torch.max(self.frame2batchidx).item() + 1
        except RuntimeError as e:
            # print("Runtime error indexing tensor!")
            # print("============")
            # print(e)
            # print(self.frame2batchidx)
            # print(self.point2frameidx)
            # print(self.point2batchidx)
            # print(self.points)
            # print(self.timesteps)
            # print(self.boxes)
            # print(self.confidences)
            # print("============")
            return 0

    @staticmethod
    def merge(objects1, objects2):
        if objects1 is None:
            return objects2
        elif objects2 is None:
            return objects1
        merge_keys = (BaseSTObjectRepresentations.OBJECT_ATTRS, BaseSTObjectRepresentations.FRAME_ATTRS, BaseSTObjectRepresentations.POINT_ATTRS)
        object_data, frame_data, point_data, frame2batchidx, point2frameidx = BaseSTObjectRepresentations._merge(objects1, objects2, keys=merge_keys)
        if frame_data['timesteps'].size(0) == 0:
            return EmptySTObjectRepresentations(objects1.device, objects1.size_clusters, objects1.reference_frame)
        return BaseSTObjectRepresentations(object_data, frame_data, point_data, point2frameidx, frame2batchidx, ref_frame=objects1.reference_frame)

    @staticmethod
    def _merge(objects1, objects2, keys):
        object_data = {k: torch.cat([objects1.object_data[k], objects2.object_data[k]]) for k in keys[0]}

        # stack frame-level information
        frame_data = {}
        for k in keys[1]:
            if torch.is_tensor(objects1.frame_data[k]):
                frame_data[k] = torch.cat([objects1.frame_data[k], objects2.frame_data[k]])
            elif isinstance(objects1.frame_data[k], LidarBoundingBoxes):
                frame_data[k] = LidarBoundingBoxes.concatenate([objects1.frame_data[k], objects2.frame_data[k]])
        frame2batchidx = torch.cat([objects1.frame2batchidx, objects2.frame2batchidx + len(objects1)])

        # stack point-level information
        point_data = {k: torch.cat([objects1.point_data[k], objects2.point_data[k]]) for k in keys[2]}
        point2frameidx = torch.cat([objects1.point2frameidx, objects2.point2frameidx + objects1.frame2batchidx.size(0)])

        return object_data, frame_data, point_data, frame2batchidx, point2frameidx

    @staticmethod
    def join(objects1, objects2):
        """
        NOTE: the reference frame of objects1 is kept!
        Returns:

        """
        assert objects1.reference_frame == "global" and objects2.reference_frame == "global"
        object_data = {}
        for k in objects1.object_data:
            if k == 'global2currentego':
                object_data[k] = objects2.object_data[k]
            else:
                object_data[k] = objects1.object_data[k]

        # batch-level information remains the same!
        frame2batch1 = objects1.frame2batchidx
        frame2batch2 = objects2.frame2batchidx
        frame2batchidx_unsorted = torch.cat([frame2batch1, frame2batch2])
        frame2batchidx, batch_sort_idxs = torch.sort(frame2batchidx_unsorted, stable=True)

        # join frame-level information
        frame_data = {}
        for k in BaseSTObjectRepresentations.FRAME_ATTRS:
            if torch.is_tensor(objects1.frame_data[k]):
                frame_data[k] = torch.cat([objects1.frame_data[k], objects2.frame_data[k]])[batch_sort_idxs]
            elif isinstance(objects1.frame_data[k], LidarBoundingBoxes):
                frame_data[k] = LidarBoundingBoxes.concatenate([objects1.frame_data[k], objects2.frame_data[k]])[batch_sort_idxs]

        # first convert points to padded representations for stability
        get_max_batchsize = lambda batchidxs: torch.max(torch.unique(batchidxs, return_counts=True)[1]).item()
        frames_padding = max(get_max_batchsize(objects1.frame2batchidx), get_max_batchsize(objects2.frame2batchidx))
        points_padding = max(get_max_batchsize(objects1.point2batchidx) if objects1.point2batchidx.size(0) > 0 else 1, get_max_batchsize(objects2.point2batchidx) if objects2.point2batchidx.size(0) > 0 else 1)
        _, _, obj1_point_data, obj1_point2frameidx, obj1_frame_data_mask, obj1_point_data_mask = objects1.to_padded(frames_padding, points_padding)
        _, _, obj2_point_data, obj2_point2frameidx, obj2_frame_data_mask, obj2_point_data_mask = objects2.to_padded(frames_padding, points_padding)
        point_data_padded = {k: torch.cat([obj1_point_data[k], obj2_point_data[k]], dim=0) for k in BaseSTObjectRepresentations.POINT_ATTRS}
        point2frameidx_padded = torch.cat([obj1_point2frameidx, obj2_point2frameidx], dim=0)
        point_data_mask = torch.cat([obj1_point_data_mask, obj2_point_data_mask], dim=0)
        frame_data_mask = torch.cat([obj1_frame_data_mask, obj2_frame_data_mask], dim=0)

        # re-order everything appropriately (note that frame2batchidx has us covered for object assignments)
        num_objects = len(objects1)
        object_idxs = torch.arange(num_objects).repeat(2).to(objects1.device)
        _, object_sort_idxs = torch.sort(object_idxs, stable=True)
        point_data_padded = {k: point_data_padded[k][object_sort_idxs] for k in point_data_padded}
        point2frameidx_padded = point2frameidx_padded[object_sort_idxs]
        point_data_mask = point_data_mask[object_sort_idxs]
        frame_data_mask = frame_data_mask[object_sort_idxs]

        # convert back
        _, _, point_data, point2frameidx, _ = objects1.to_dense({}, {}, point_data_padded, point2frameidx_padded, frame_data_mask, point_data_mask)

        # reset point-timesteps
        point_data['points'][:, 3] = BaseSTObjectRepresentations.reset_point_times(frame_data['timesteps'], point2frameidx, frame2batchidx)

        return BaseSTObjectRepresentations(object_data, frame_data, point_data, point2frameidx, frame2batchidx, ref_frame="global")

    @staticmethod
    def reset_point_times(timesteps, point2frameidx, frame2batchidx):
        point2batchidx = frame2batchidx[point2frameidx]
        per_obj_mintimes = - global_max_pool(-timesteps, frame2batchidx)
        per_point_mintimes = per_obj_mintimes[point2batchidx]
        per_point_timesteps = timesteps[point2frameidx]
        updated_times = per_point_timesteps - per_point_mintimes
        return updated_times


    @staticmethod
    def regression_1st_order(T, X, V, T_eval, padding_mask):
        """

        Args:
            T (torch.tensor): size (B, N) tensor of times for each object-sequence
            X (torch.tensor): size (B, N, dims) tensor of position observations at each timestep
            V (torch.tensor): size (B, N, dims) tensor of velocity observations at each timestep
            T_eval (torch.tensor): size (B, N') tensor of evaluation times
            padding_mask (torch.tensor): size (B, N) tensor with True indicating to ignore that value

        Returns:

        """
        B, N = T.size()

        # initialize solving matrices
        XV = torch.cat([X, V], dim=1) # B x 2N x 2
        A_X = torch.stack([torch.ones(B, N).to(T), T], dim=2)  # B x N x 2
        A_V = torch.stack([torch.zeros(B, N).to(T), torch.ones(B, N).to(T)], dim=2)
        A = torch.cat([A_X, A_V], dim=1)  # B x 2N x 2

        # zero out padding values
        padding_mask_stacked = torch.cat([padding_mask, padding_mask], dim=1)
        XV[padding_mask_stacked] = 0
        A[padding_mask_stacked] = 0

        # solve best-fit 1st order parametrics
        left_side = (A.transpose(1, 2) @ A).double()  # B x interp-order x interp-order
        right_side = (A.transpose(1, 2) @ XV).double()  # (B, interp_order, N) x (B, N, -1) --> (B, interp_order, box-dim)
        # parametrics = torch.linalg.solve(left_side, right_side).float()  # (B, interp-order, box-dim)

        try:
            parametrics = torch.linalg.solve(left_side, right_side).float()  # (B, interp-order, box-dim)
        except RuntimeError:  # singular, but should never be, so we have a bug
            try:
                print("Parametrics Error: singularity occured.")
                ranks = torch.linalg.matrix_rank(left_side)
                singular_idxs = torch.where(ranks < 2)[0]
                print("Singular matrices:")
                print(left_side[singular_idxs])
                print(T[singular_idxs])
                print(A[singular_idxs])
                print(padding_mask[singular_idxs])
                updated_ridge = torch.eye(2).to(left_side).unsqueeze(0).repeat(singular_idxs.size(0), 1, 1)
                updated_ridge[:, 0, 0] = 0
                updated_ridge *= 1e6
                left_side[singular_idxs] += updated_ridge
                parametrics = torch.linalg.solve(left_side, right_side).float()  # (B, interp-order, box-dim)
            except RuntimeError:
                print("Error in padding mask, because everything is zero...")
                ranks = torch.linalg.matrix_rank(left_side)
                singular_idxs = torch.where(ranks < 2)[0]
                updated_ridge = torch.eye(2).to(left_side).unsqueeze(0).repeat(singular_idxs.size(0), 1, 1)
                updated_ridge *= 1e6
                left_side[singular_idxs] += updated_ridge
                parametrics = torch.linalg.solve(left_side, right_side).float()  # (B, 2, 3)

        # plug parametrics into evaluation times
        A_X_eval = torch.stack([torch.ones(T_eval.size()).to(T_eval), T_eval], dim=1)  # (B, interp_order, num-evals)
        X_out = parametrics.transpose(1, 2) @ A_X_eval  # outputs (B, box-dim, 1)
        V_out = parametrics[:, 1, :]  # (B, 3)

        return X_out.squeeze(-1), V_out, parametrics.transpose(1, 2).contiguous()

    @staticmethod
    def to_batch_padded(tensor, val2batchidx, pad_size=None, batch_size=None):
        if isinstance(tensor, LidarBoundingBoxes):
            tensor = tensor.box

        orig_idxs, map2origidxs, counts = torch.unique(val2batchidx, sorted=True, return_inverse=True, return_counts=True)  # todo: what if this reorders!!
        if pad_size is None:
            pad_size = torch.max(counts).item()
        if batch_size is None:
            batch_size = torch.max(val2batchidx).item() + 1

        # convert into padded version
        grouped_vals, withval_mask = group_first_k_values(tensor, map2origidxs, pad_size)

        # map back to original indices
        if len(grouped_vals.size()) == 3:
            all_grouped_vals = torch.zeros(batch_size, pad_size, grouped_vals.size(-1)).to(grouped_vals)  # todo: this is incorrect!!
        elif len(grouped_vals.size()) == 2:
            all_grouped_vals = torch.zeros(batch_size, pad_size).to(grouped_vals)
        else:
            print(grouped_vals.size())
            raise RuntimeError("Currently to_batch_padded only supports inputs of size 1 or 2.")

        all_grouped_vals[orig_idxs] = grouped_vals
        all_withval_mask = torch.zeros(batch_size, pad_size).to(withval_mask)
        all_withval_mask[orig_idxs] = withval_mask

        return all_grouped_vals, all_withval_mask

    @staticmethod
    def affine_transform_points(points, batch, affine, pre_translation=None, inverse=False):
        affine_pnts = points.clone()
        affine_xyz = affine_pnts[:, :3]

        if pre_translation is not None:
            translations = pre_translation[batch]
            affine_xyz += translations

        if inverse:
            affine = affine.inverse()
        affines = affine[batch]
        affine_xyz = affines[:, :3, :3] @ affine_xyz.view(-1, 3, 1).double() + affines[:, :3, 3:4]
        affine_xyz = affine_xyz.float().squeeze(-1)
        affine_pnts[:, :3] = affine_xyz
        return affine_pnts

class EmptySTObjectRepresentations(BaseSTObjectRepresentations):

    def __init__(self, device, size_clusters, reference_frame="local"):
        self.device = device
        self.size_clusters = size_clusters
        self.reference_frame = reference_frame

    @property
    def object_data(self):
        return {"global2locals": self.global2locals, 'shape_features': self.shape_features, "global2currentego": self.global2currentego}

    @property
    def frame_data(self):
        return {"boxes": self.boxes, "confidences": self.confidences, "timesteps": self.timesteps, "instance_ids": self.instance_ids,
                "gt_boxes": self.boxes, "with_gt_mask": self.with_gt_mask, "gt_classification": self.gt_classification}

    @property
    def point_data(self):
        return {"points": self.points, "gt_segmentation": self.gt_segmentation, "segmentation": self.segmentation}

    @property
    def global2locals(self):
        return torch.zeros((0, 4, 4), dtype=torch.double).to(self.device)

    @property
    def global2currentego(self):
        return torch.zeros((0, 4, 4), dtype=torch.double).to(self.device)

    @property
    def boxes(self):
        empty_boxes = LidarBoundingBoxes(torch.zeros(0, 13), torch.zeros(0, 14 + self.size_clusters.size(0)), self.size_clusters)
        empty_boxes = empty_boxes.to(self.device)
        return empty_boxes

    @property
    def with_gt_mask(self):
        return torch.zeros(0, dtype=torch.bool).to(self.device)

    @property
    def gt_boxes(self):
        return self.boxes

    @property
    def gt_classification(self):
        return torch.zeros(0, dtype=torch.long).to(self.device)

    @property
    def gt_segmentation(self):
        return torch.zeros(0, 1).to(self.device)

    @property
    def tnocs(self):
        return self.points

    @property
    def segmentation(self):
        return torch.zeros(0, 1).to(self.device)

    @property
    def shape_features(self):
        return torch.zeros(0).to(self.device)

    @property
    def confidences(self):
        return torch.zeros(0).to(self.device)

    @property
    def instance_ids(self):
        return torch.zeros(0, dtype=torch.long).to(self.device)

    @property
    def timesteps(self):
        return torch.zeros(0, dtype=torch.double).to(self.device)

    @property
    def points(self):
        return torch.zeros((0, 4)).to(self.device)

    @property
    def frame2batchidx(self):
        return torch.zeros(0, dtype=torch.long).to(self.device)

    @property
    def point2frameidx(self):
        return torch.zeros(0, dtype=torch.long).to(self.device)

    @property
    def point2batchidx(self):
        return torch.zeros(0, dtype=torch.long).to(self.device)

    def change_basis_local2global(self):
        self.reference_frame = "global"
        return self

    def change_basis_global2local(self):
        self.reference_frame = "local"
        return self

    def change_basis_currentego2global(self):
        self.reference_frame = "global"
        return self

    def change_basis_global2currentego(self):
        self.reference_frame = "ego"
        return self

    def estimate_boxes(self, eval_times, relative_time_context=0.4, time_context=1.5):
        empty_parametrics = torch.zeros(0, 4, 2).to(self.device)
        return [self.boxes] * eval_times.size(0), [empty_parametrics] * eval_times.size(0)

    def get_object_confidences(self, which):
        assert isinstance(which, float) or isinstance(which, str)
        if isinstance(which, str):
            assert which in ["mean", "max"]
        return torch.tensor([]).to(self.device)

    def get_object_timesteps(self, which="all", return_idx=False):
        assert which in ["all", "min", "max"]
        if not return_idx:
            return torch.tensor([]).to(self.device)
        else:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)

    def to_inputs_list(self):
        return []

    def to(self, device):
        self.device = device
        return self

    def clip_sequences(self, time_context, points_context=None):
        return self

    def remove_empty_frames(self):
        return self

    def set(self, idxs, objects):
        if len(idxs) > 0:
            raise IndexError("Attempting to set elements in an empty object representation.")
        return self

    def clone(self):
        return EmptySTObjectRepresentations(self.device, self.size_clusters, self.reference_frame)

    def __getitem__(self, idx):
        indices = convert_idx_to_tensor(idx, len(self), self.device)
        if len(indices) > 0:
            raise IndexError("Indexing into an empty set of objects.")
        return self

    def __len__(self):
        return 0


class MergedEnhancedAndBaseObjectRepresentations:

    def __init__(self, enhanced_objects, base_objects, device):
        """

        Args:
            enhanced_objects: ProcessedSTObjectRepresentations containing our refined sequences
            base_objects: BaseSTObjectRepresentations containing our ignored sequences
        """
        if enhanced_objects is None and base_objects is None:
            raise RuntimeError("MergedEnhancedAndBaseObjectRepresentations cannot take two empty object sets.")
        self.enhanced_objects = enhanced_objects if enhanced_objects is not None else EmptySTObjectRepresentations(device, base_objects.boxes.size_anchors, base_objects.reference_frame)
        self.base_objects = base_objects if base_objects is not None else EmptySTObjectRepresentations(device, enhanced_objects.boxes.size_anchors, enhanced_objects.reference_frame)
        self.device = device

    def estimate_boxes(self, timesteps, trajectory_params):
        enhanced_boxes, enhanced_params = self.enhanced_objects.estimate_boxes(timesteps, **trajectory_params)
        other_boxes, other_params = self.base_objects.estimate_boxes(timesteps, **trajectory_params)

        all_boxes = []
        for i in range(timesteps.size(0)):
            timestep_boxes = LidarBoundingBoxes.concatenate([enhanced_boxes[i], other_boxes[i]])
            all_boxes.append(timestep_boxes)

        all_params = []
        for i in range(timesteps.size(0)):
            timestep_params = torch.cat([enhanced_params[i], other_params[i]])
            all_params.append(timestep_params)

        return all_boxes, all_params

    def get_object_confidences(self, which):
        confidences = [self.enhanced_objects.get_object_confidences(which), self.base_objects.get_object_confidences(which)]
        return torch.cat(confidences)

    def get_object_timesteps(self, which="all"):
        timesteps = [self.enhanced_objects.get_object_timesteps(which), self.base_objects.get_object_timesteps(which)]
        return torch.cat(timesteps)


    def sequence_nms(self, threshold, trajectory_params):
        """
        Runs NMS to remove any overlapping object sequences, where overlap is measured at timestep.
        Args:
            timestep: The timestep to evaluate NMS on. Can also be string value "min" or "max", which indicates minimum
                      or maximum available timestep over all object observations

        Returns:

        """
        # get timestep if min or max is specified
        timesteps = torch.unique(self.timesteps)
        timesteps = torch.sort(timesteps)[0]
        if timesteps.size(0) > 10:
            timestep_idxs = [i * (timesteps.size(0) // 10) for i in range(10)]
            timesteps = timesteps[timestep_idxs]
        num_timesteps = timesteps.size(0)

        # get all boxes at the timestep
        eval_times = torch.tensor(timesteps, dtype=torch.double).to(self.device)
        enhanced_boxes = self.enhanced_objects.estimate_boxes(eval_times, **trajectory_params)[0]
        base_boxes = self.base_objects.estimate_boxes(eval_times, **trajectory_params)[0]
        boxes = [LidarBoundingBoxes.concatenate([enhanced_boxes[i], base_boxes[i]], dim=0) for i in range(num_timesteps)]
        num_objects = len(self)

        # compute IOUs at each frame
        max_ious = torch.zeros((num_objects, num_objects), dtype=torch.float).to(self.device)
        for i in range(num_timesteps):
            scene_centering = boxes[i][0:1].center[0:1, :2]
            boxes_bev = lidar2bev_box(boxes[i].box, scene_centering)
            iou_bev = boxes_iou_bev(boxes_bev, boxes_bev)
            max_ious = torch.maximum(max_ious, iou_bev)
        max_ious *= (~torch.eye(num_objects, dtype=torch.bool).to(self.device))  # set identity to 0

        # get mean confidences
        confidences = [self.enhanced_objects.get_object_confidences("mean"), self.base_objects.get_object_confidences("mean")]
        confidences = torch.cat(confidences, dim=0)
        confidence_ordering = torch.argsort(confidences, descending=True)

        # get list of objects to remove
        remove_idxs = set()
        for i in range(num_objects):
            obj_idx = confidence_ordering[i].item()
            if obj_idx in remove_idxs:
                continue
            overlapping_idxs = torch.where(max_ious[obj_idx] > threshold)[0]
            for idx in overlapping_idxs:
                remove_idxs.add(idx.item())
        remove_idxs = torch.tensor(list(remove_idxs), dtype=torch.long).to(self.device)
        remove_idxs = torch.sort(remove_idxs)[0]

        # compute keep idxs
        keep_idxs = torch.ones(len(self), dtype=torch.bool).to(self.device)
        keep_idxs[remove_idxs] = False
        keep_idxs = torch.where(keep_idxs)[0]
        keep_idxs_split = [keep_idxs[(keep_idxs < len(self.enhanced_objects))], keep_idxs[(keep_idxs >= len(self.enhanced_objects))] - len(self.enhanced_objects)]

        # get removed objects
        remove_idxs_split = [remove_idxs[(remove_idxs < len(self.enhanced_objects))], remove_idxs[(remove_idxs >= len(self.enhanced_objects))] - len(self.enhanced_objects)]
        remove_enhanced_objects, removed_base_objects = self.enhanced_objects[remove_idxs_split[0]], self.base_objects[remove_idxs_split[1]]
        removed_objects = MergedEnhancedAndBaseObjectRepresentations(remove_enhanced_objects, removed_base_objects, self.device)

        # filter out from each object-representation
        enhanced_objects = self.enhanced_objects[keep_idxs_split[0]]
        base_objects = self.base_objects[keep_idxs_split[1]]
        out = MergedEnhancedAndBaseObjectRepresentations(enhanced_objects, base_objects, self.device)

        return out, keep_idxs, removed_objects


    def to(self, device):
        self.enhanced_objects = self.enhanced_objects.to(device)
        self.base_objects = self.base_objects.to(device)
        self.device = device
        return self


    def to_base_rep(self):
        if isinstance(self.base_objects, EmptySTObjectRepresentations):
            return self.enhanced_objects.to_base_rep()
        elif isinstance(self.enhanced_objects, EmptySTObjectRepresentations):
            return self.base_objects
        else:
            return BaseSTObjectRepresentations.merge(self.enhanced_objects, self.base_objects)

    def change_basis_currentego2global(self):
        self.enhanced_objects = self.enhanced_objects.change_basis_currentego2global()
        self.base_objects = self.base_objects.change_basis_currentego2global()
        return self

    def change_basis_global2currentego(self):
        self.enhanced_objects = self.enhanced_objects.change_basis_global2currentego()
        self.base_objects = self.base_objects.change_basis_global2currentego()
        return self

    def clone(self):
        enhanced_objects = self.enhanced_objects.clone()
        base_objects = self.base_objects.clone()
        return MergedEnhancedAndBaseObjectRepresentations(enhanced_objects, base_objects, self.device)

    def __len__(self):
        return len(self.enhanced_objects) + len(self.base_objects)

    @property
    def points(self):
        return torch.cat([self.enhanced_objects.points, self.base_objects.points])

    @property
    def point2batchidx(self):
        point2batchidx = torch.cat([self.enhanced_objects.point2batchidx, self.base_objects.point2batchidx + len(self.enhanced_objects)])
        return point2batchidx

    @property
    def frame2batchidx(self):
        frame2batchidx = torch.cat([self.enhanced_objects.frame2batchidx, self.base_objects.frame2batchidx + len(self.enhanced_objects)])
        return frame2batchidx

    @property
    def point2frameidx(self):
        merged_base_object = BaseSTObjectRepresentations.merge(self.enhanced_objects, self.base_objects)
        return merged_base_object.point2frameidx

    @property
    def confidences(self):
        confidences = torch.cat([self.enhanced_objects.confidences, self.base_objects.confidences])
        return confidences

    @property
    def global2locals(self):
        return torch.cat([self.enhanced_objects.global2locals, self.base_objects.global2locals])

    @property
    def global2currentego(self):
        return torch.cat([self.enhanced_objects.global2currentego, self.base_objects.global2currentego])

    @property
    def boxes(self):
        return LidarBoundingBoxes.concatenate([self.enhanced_objects.boxes, self.base_objects.boxes])

    @property
    def instance_ids(self):
        return torch.cat([self.enhanced_objects.instance_ids, self.base_objects.instance_ids])

    @property
    def timesteps(self):
        return torch.cat([self.enhanced_objects.timesteps, self.base_objects.timesteps])



def convert_idx_to_tensor(idx, length, device):
    if isinstance(idx, slice):
        slice_idxs = idx.indices(length)
        indices = torch.arange(slice_idxs[0], slice_idxs[1], slice_idxs[2]).to(device)
    elif torch.is_tensor(idx):
        if idx.dtype == torch.bool:
            indices = torch.where(idx)[0]
        elif idx.dtype == torch.long:
            indices = idx
        else:
            raise RuntimeError("If indices are a tensor, must be of type bool or long.")
    elif isinstance(idx, list):
        indices = torch.tensor(idx).to(device)
    elif isinstance(idx, int):
        indices = idx
    else:
        raise RuntimeError("Indices must be one of [slice, tensor, list, int'. Got %s" % type(idx))
    return indices
