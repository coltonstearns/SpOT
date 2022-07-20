
import torch
import numpy as np

class AugmentSequences:

    def __init__(self, training_params, seq_len, min_seq_pts, overlap_threshold, device):
        self.shift_augment = training_params['train-shift-augment']
        self.rotation_noise = training_params['point-rotation-noise']
        self.scale_noise = training_params['point-scale-noise']
        self.overlap_threshold = overlap_threshold
        self.seq_len = seq_len
        self.min_seq_pts = min_seq_pts
        self.device = device

    def augment(self, proposals, points):
        if self.shift_augment:
            proposals, points = self._augment_shift(proposals, points)
        proposals["proposal_boxes"], proposals["gt_boxes"], points["points"] = \
            self._augment_scale_rotate_flip(proposals["proposal_boxes"], proposals["gt_boxes"],
                                            points["points"], proposals["hasframe_mask"], proposals["haslabel_mask"])

        return proposals, points

    def _augment_shift(self, proposals, points):
        # compute frame shift
        n_frames, n_frames_gt = proposals['haslabel_mask'].size(0), torch.sum(proposals['haslabel_mask']).item()
        if n_frames_gt > 0:
            lastframe_idx = torch.where(proposals['haslabel_mask'])[0][-1].item()
        else:
            lastframe_idx = len(proposals['haslabel_mask'])

        # shift based on chi-squared distribution
        # shift = (torch.randn(1)**2 / 4 * n_frames).int()  # chi-squared distribution
        if lastframe_idx == 0:
            shift = 0
        else:
            shift = torch.randint(lastframe_idx, (1,)).item()
        shift = min(shift, lastframe_idx)
        shift = max(shift, 0)
        shifted_points = points['points'][shift:]
        if self._sequence_num_pts(shifted_points) < self.min_seq_pts:
            shift = 0

        # compute shift index in full-sequence reference frame
        fullseq_mask = torch.zeros(proposals['hasframe_mask'].size(0)).to(self.device)
        fullseq_mask[proposals['hasframe_mask']] = torch.ones(n_frames).to(self.device)
        fullseq_mask = (torch.cumsum(fullseq_mask, dim=0) - 1) < shift
        proposals['hasframe_mask'][fullseq_mask] = False

        # compute shift in GT reference frame
        if n_frames_gt > 0:
            shift_idx_bool = proposals['haslabel_mask'].clone() * 0
            shift_idx_bool[shift:] = 1
            shift_idx_gt = torch.where(shift_idx_bool[proposals['haslabel_mask']])[0][0].item()
        else:
            shift_idx_gt = 0

        # shift everything
        for k, val in proposals.items():
            if k == "glob2sensor" or k == "class" or k == "class_idx":
                continue
            elif k == 'gt_boxes':
                proposals[k] = val[shift_idx_gt:]
            else:
                proposals[k] = val[shift:]

        # shift points
        for k, val in points.items():
            if k == 'points':
                points[k] = val[shift:]
            elif k == 'gt_segmentation':
                points[k] = val[shift_idx_gt:]

        return proposals, points

    def _sequence_num_pts(self, sequence):
        seq_num_pts = []
        for instance_pts in sequence:
            inst_num_pts = 0 if instance_pts is None else instance_pts.size(0)
            seq_num_pts.append(inst_num_pts)
        return sum(seq_num_pts)

    def _augment_scale_rotate_flip(self, proposal_boxes, gt_boxes, points, hasframe_mask, haslabel_mask):
        noise_rotation = torch.tensor([np.random.uniform(self.rotation_noise[0], self.rotation_noise[1])])
        noise_scale = torch.tensor([np.random.uniform(self.scale_noise[0], self.scale_noise[1])])
        flip = np.random.rand() > 0.5
        num_frames, num_labels = len(points), torch.sum(haslabel_mask).item()

        # augment points and boxes
        if flip:
            proposal_boxes, points = proposal_boxes.flip(bev_direction='horizontal', points=points)
        proposal_boxes, points = proposal_boxes.rotate(noise_rotation, torch.zeros(num_frames, dtype=torch.long).to(self.device), points=points)
        proposal_boxes = proposal_boxes.scale(noise_scale, torch.zeros(num_frames, dtype=torch.long).to(self.device))
        points = [point_batch * noise_scale for point_batch in points]

        # augment gt boxes
        if num_labels > 0:
            if flip:
                gt_boxes = gt_boxes.flip(bev_direction='horizontal')
            gt_boxes = gt_boxes.rotate(noise_rotation, torch.zeros(num_labels, dtype=torch.long).to(self.device))
            gt_boxes = gt_boxes.scale(noise_scale, torch.zeros(num_labels, dtype=torch.long).to(self.device))

        return proposal_boxes, gt_boxes, points


class AugmentBoxes:

    def __init__(self, augment, training_params, seq_len, min_seq_pts, device):
        self.rotation_noise = training_params['bbox-rotation-noise']
        self.translation_noise = training_params['bbox-translation-noise']
        self.scale_noise = training_params['bbox-scale-noise']
        self.frame_rotation_noise = training_params['bbox-rotation-noise-per-frame']
        self.frame_translation_noise = training_params['bbox-translation-noise-per-frame']

        self.seq_len = seq_len
        self.min_seq_pts = min_seq_pts
        self.device = device
        self.perform_augment = augment

    def augment(self, proposals):
        augmented_boxes = self._augment_boxes(proposals["proposal_boxes"], proposals["hasframe_mask"])
        proposals['input_boxes'] = augmented_boxes
        return proposals

    def _augment_boxes(self, proposal_boxes, hasframe_mask):
        # set up noise augmentation as tensors
        noise_rotation = torch.tensor([np.random.uniform(self.rotation_noise[0], self.rotation_noise[1])]).to(self.device)
        noise_translation = np.random.uniform(self.translation_noise[0], self.translation_noise[1], size=(1,3))
        noise_translation = torch.from_numpy(noise_translation).to(self.device)
        noise_scale = torch.tensor([np.random.uniform(self.scale_noise[0], self.scale_noise[1])]).to(self.device)
        frame_rotation_noise = np.random.uniform(self.frame_rotation_noise[0], self.frame_rotation_noise[1], size=(proposal_boxes.size(0),))
        frame_translation_noise = np.random.uniform(self.frame_translation_noise[0], self.frame_translation_noise[1], size=(proposal_boxes.size(0), 3))
        frame_rotation_noise = torch.from_numpy(frame_rotation_noise).to(self.device).float()
        frame_translation_noise = torch.from_numpy(frame_translation_noise).to(self.device).float()

        # add noise to box sequence
        augmented_boxes, N = proposal_boxes.clone(), proposal_boxes.size(0)
        if self.perform_augment:
            orig_center = proposal_boxes.center
            apply_all, apply_perframe = torch.zeros(N, dtype=torch.long).to(self.device), torch.arange(N).to(self.device)
            augmented_boxes = augmented_boxes.translate(-orig_center, batch=apply_perframe)
            augmented_boxes = augmented_boxes.rotate(angle=noise_rotation,  batch=apply_all)
            augmented_boxes = augmented_boxes.rotate(angle=frame_rotation_noise,  batch=apply_perframe)
            augmented_boxes = augmented_boxes.scale(scale=noise_scale,  batch=apply_all)
            augmented_boxes = augmented_boxes.translate(translation=noise_translation,  batch=apply_all)
            augmented_boxes = augmented_boxes.translate(translation=frame_translation_noise,  batch=apply_perframe)
            augmented_boxes = augmented_boxes.translate(orig_center, batch=apply_perframe)

        return augmented_boxes