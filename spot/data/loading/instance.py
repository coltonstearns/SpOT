import numpy as np
import torch


class ObjectProposal:

    def __init__(self,
                 timestep=None,
                 keyframe_id=None,
                 world2sensor=None,
                 is_keyframe=False,
                 instance_id=None,
                 bounding_box=None,
                 object_class=None,
                 confidence=None,
                 gt_status='with-gt',
                 gt_instance_id=None,
                 gt_bounding_box=None,
                 gt_is_moving=None,
                 point_data_path='',
                 num_pts=None,
                 surrounding_context_factor=None,
                 ):
        """

        Args:
            instance_id:
            timestep:
            bounding_box:
            object_type: The object type
            confidence: [0,1] indicating original prediction confidence
            num_pts:
            gt_type: one of ['with-gt', 'in-gt-sequence', 'false-positive'], where fp stands for false-positive
            gt_bbox:
            point_data_path:
            is_moving: Boolean indicating if object is stationary or in motion
        """
        assert gt_status in ['with-gt', 'in-gt-sequence', 'false-positive']

        # frame properties
        self.timestep = timestep
        self.keyframe_id = keyframe_id
        self.world2sensor = world2sensor
        self.is_keyframe = is_keyframe

        # prediction properties
        self.instance_id = instance_id
        self.bounding_box = bounding_box
        self.object_class = object_class
        self.confidence = confidence

        # ground truth properties
        self.gt_status = gt_status
        self.gt_instance_id = gt_instance_id
        self.gt_bounding_box = gt_bounding_box
        self.gt_is_moving = gt_is_moving

        # point properties
        self.point_data_path = point_data_path
        self.num_pts = num_pts
        self.surrounding_context_factor = surrounding_context_factor

    def np2torch(self, device):
        assert isinstance(self.bounding_box, np.ndarray)
        bounding_box = torch.from_numpy(self.bounding_box[:13]).to(device).float()
        world2sensor = torch.from_numpy(self.world2sensor).to(device)
        if self.gt_bounding_box is not None:
            assert isinstance(self.gt_bounding_box, np.ndarray)
            gt_bounding_box = torch.from_numpy(self.gt_bounding_box[:13]).to(device).float()  # [xyz, wlh, yaw, vel, acc]
        else:
            gt_bounding_box = None
        return ObjectProposal(timestep=self.timestep, keyframe_id=self.keyframe_id, world2sensor=world2sensor,
                              is_keyframe=self.is_keyframe, instance_id=self.instance_id, bounding_box=bounding_box,
                              object_class=self.object_class, confidence=self.confidence, gt_status=self.gt_status,
                              gt_instance_id=self.gt_instance_id, gt_bounding_box=gt_bounding_box, gt_is_moving=self.gt_is_moving,
                              point_data_path=self.point_data_path, num_pts=self.num_pts, surrounding_context_factor=self.surrounding_context_factor)

    def dump_to_dict(self):
        gt_bounding_box = self.gt_bounding_box.tolist() if self.gt_bounding_box is not None else None
        out_dict = {"bounding_box": self.bounding_box.tolist(), "object_class": self.object_class,
                    "instance_id": self.instance_id, "gt_status": self.gt_status,
                    "gt_bounding_box": gt_bounding_box, "point_data_path": self.point_data_path,
                    "timestep": self.timestep, "num_pts": self.num_pts, "surrounding_context_factor": self.surrounding_context_factor,
                    "confidence": self.confidence, "keyframe_id": self.keyframe_id, "gt_is_moving": self.gt_is_moving,
                    "world2sensor": self.world2sensor.tolist(), "is_keyframe": self.is_keyframe, "gt_instance_id": self.gt_instance_id}
        return out_dict

    def load_from_dict(self, load_dict):
        for k, val in load_dict.items():
            if k == "bounding_box":
                setattr(self, k, np.array(val))
            elif k == "world2sensor":
                setattr(self, k, np.array(val, dtype=np.float64))
            elif k == "gt_bounding_box" and val is not None:
                setattr(self, k, np.array(val))
            else:
                setattr(self, k, val)

