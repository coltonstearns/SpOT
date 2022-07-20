import torch

class EgomotionAggregate:

    def __init__(self, seq_len, device, sample_id=None):
        self.seq_len = seq_len
        self.device = device
        self.sample_id = sample_id

    def egomotion_aggregate(self, proposals, points):
        n_frames, n_labels = proposals['glob2sensor'].size(0), torch.sum(proposals['haslabel_mask']).item()

        # note: egomotion is more accurate in keyframes for NuScenes...?
        aggregation_idxs = torch.where(proposals['keyframe_mask'])[0]
        aggregation_idx = aggregation_idxs[-1].item() if len(aggregation_idxs) > 0 else (n_frames-1)
        aggregate_ref_frame = proposals['glob2sensor'][aggregation_idx:aggregation_idx+1]

        # aggregate input boxes
        proposals['proposal_boxes'] = proposals['proposal_boxes'].coordinate_transfer(proposals['glob2sensor'], batch=torch.arange(n_frames).to(self.device), inverse=True)
        proposals['proposal_boxes'] = proposals['proposal_boxes'].coordinate_transfer(aggregate_ref_frame, batch=torch.zeros(n_frames, dtype=torch.long).to(self.device), inverse=False)

        # aggregate points
        sensor2globs = torch.inverse(proposals['glob2sensor'])
        points['points'] = [self.ego_transform_points(points['points'][i], sensor2globs[i], aggregate_ref_frame[0]) for i in range(n_frames)]

        # aggregate GT boxes
        proposals['gt_boxes'] = proposals['gt_boxes'].coordinate_transfer(proposals['glob2sensor'][proposals['haslabel_mask']], batch=torch.arange(n_labels).to(self.device), inverse=True)
        proposals['gt_boxes'] = proposals['gt_boxes'].coordinate_transfer(aggregate_ref_frame, batch=torch.zeros(n_labels, dtype=torch.long).to(self.device), inverse=False)

        # update glob2sensor to single reference frame
        proposals['glob2sensor'] = aggregate_ref_frame.squeeze(0)

        return proposals, points

    def ego_transform_points(self, pnts, ego1_to_global, global_to_ego2):
        glob_pnts = (pnts.double() @ ego1_to_global[:3, :3].T) + ego1_to_global[:3, 3].view(1, 3)
        ego2_pnts = (glob_pnts @ global_to_ego2[:3, :3].T) + global_to_ego2[:3, 3].view(1, 3)
        return ego2_pnts.float()
