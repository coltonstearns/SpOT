import numpy as np
import torch

class OriginCenter:

    def __init__(self, seq_len, backwards_time):
        self.seq_len = seq_len
        self.backwards = backwards_time

    def origin_center(self, proposals, points):
        orig_timesteps = proposals['timesteps']
        centered_info = self._origin_center_sequence(proposals['input_boxes'], proposals['proposal_boxes'], proposals['gt_boxes'],
                                                     points['points'], proposals['timesteps'], proposals['hasframe_mask'])
        proposals['input_boxes'] = centered_info[0]
        proposals['proposal_boxes'] = centered_info[1]
        proposals['gt_boxes'] = centered_info[2]
        points['points'] = centered_info[3]
        proposals['timesteps'] = centered_info[4]
        caspr2scene_translation = centered_info[5]

        proposals['caspr2scene_trans'] = caspr2scene_translation
        proposals['orig_timesteps'] = orig_timesteps
        return proposals, points

    def _origin_center_sequence(self, input_boxes, proposal_boxes, gt_boxes, points, timesteps, hasframe_mask):
        # get index of bounding box to zero-shift to
        centering_idx = torch.argmin(timesteps) if not self.backwards else torch.argmax(timesteps)
        device = input_boxes.device

        # center points
        N, N_gt = len(points), gt_boxes.size(0)
        centering_trans = input_boxes.center[centering_idx:centering_idx+1]
        centered_points = [pnts - centering_trans for pnts in points]

        # center bounding boxes
        input_boxes = input_boxes.translate(-centering_trans, batch=torch.zeros(N, dtype=torch.long).to(device))
        proposal_boxes = proposal_boxes.translate(-centering_trans, batch=torch.zeros(N, dtype=torch.long).to(device))
        gt_boxes = gt_boxes.translate(-centering_trans, batch=torch.zeros(N_gt, dtype=torch.long).to(device))

        # appropriately shift timesteps
        if not self.backwards:
            centered_times = timesteps - timesteps[centering_idx]
        else:
            centered_times = timesteps[centering_idx] - timesteps

        return input_boxes, proposal_boxes, gt_boxes, centered_points, centered_times, centering_trans
