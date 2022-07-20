import numpy as np
import torch
import os

class PointLoader:

    def __init__(self, data_dir, device, load_func):
        self.data_dir = data_dir
        self.device = device
        self.load_func = load_func

    def load_points(self, object_sequence, hasframe_mask, haslabel_mask):
        # expand label-mask to entire sequence
        haslabel_mask_expanded = torch.zeros(hasframe_mask.size(0), dtype=torch.bool).to(self.device)
        haslabel_mask_expanded[hasframe_mask] = haslabel_mask

        # load all points iteratively
        obj_seq_pnts = []
        obj_foreground_segs = []
        for i, obj_instance in enumerate(object_sequence):
            pc, foreground_seg = self._load_obj_instance_points(obj_instance)
            if hasframe_mask[i]:
                pc = pc if pc is not None else torch.zeros(0, 3).to(self.device)
                obj_seq_pnts.append(pc)
            if haslabel_mask_expanded[i]:
                foreground_seg = foreground_seg if foreground_seg is not None else torch.zeros((0, 1), dtype=torch.bool).to(self.device)
                obj_foreground_segs.append(foreground_seg)
        return {"points": obj_seq_pnts, "gt_segmentation": obj_foreground_segs}

    def _load_obj_instance_points(self, obj_instance):
        if obj_instance is None:
            return None, None

        instance_filename = "%s_%s_%s.npz" % (obj_instance.keyframe_id, obj_instance.instance_id, obj_instance.timestep)
        data = self.load_func(instance_filename)
        pc, foreground_seg = data['points'], data['segmentation']
        pc = torch.from_numpy(pc).to(self.device)
        if foreground_seg.shape != (0,):
            foreground_seg = torch.tensor(foreground_seg, dtype=torch.bool).to(self.device)
        else:
            foreground_seg = None
        return pc, foreground_seg

