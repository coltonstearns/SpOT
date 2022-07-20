import torch
from spot.io.globals import NUSCENES_CLASSES, WAYMO_CLASSES
from spot.data.box3d import LidarBoundingBoxes

class ProposalLoader:

    def __init__(self, device, seq_len, size_anchors, dataset_source):
        self.device = device
        self.seq_len = seq_len
        self.size_anchors = size_anchors
        self.dataset_source = dataset_source

    def load_proposals(self, object_sequence):
        # put into tensor format
        formatted_object_seq = self.format_seq_meta(object_sequence, self.device)

        # get proposal frame masks as tensors
        haslabel_mask, hasframe_mask, keyframe_mask = self._get_frame_masks(formatted_object_seq)
        haslabel_mask = haslabel_mask[hasframe_mask]
        keyframe_mask = keyframe_mask[hasframe_mask]
        sensor_ref_frames = self._get_sensor_refs(formatted_object_seq)

        # get proposal info as list of lists
        input_boxes = self._extract_from_obj_seqs_meta(formatted_object_seq, "bounding_box", dtype="tensor")
        gt_boxes = self._extract_from_obj_seqs_meta(formatted_object_seq, "gt_bounding_box", dtype="tensor")
        timesteps = self._extract_from_obj_seqs_meta(formatted_object_seq, "timestep", dtype="double")
        confidences = self._extract_from_obj_seqs_meta(formatted_object_seq, "confidence", dtype="float")
        gt_instance_ids = self._extract_from_obj_seqs_meta(formatted_object_seq, "gt_instance_id", dtype="string")

        # get uniquely formatted info for classes and tp/fp labels
        obj_class, obj_classidx = self._format_class_info(formatted_object_seq)  # one class per sequence, not per object
        tpfp_labels = self._get_label_info(formatted_object_seq)

        # format boxes in our bounding box object
        input_boxes = LidarBoundingBoxes(box=input_boxes[:, :13], box_enc=None, size_anchors=self.size_anchors)
        gt_boxes = LidarBoundingBoxes(box=gt_boxes[:, :13], box_enc=None, size_anchors=self.size_anchors)

        res = {"haslabel_mask": haslabel_mask, "hasframe_mask": hasframe_mask, "keyframe_mask": keyframe_mask, "glob2sensor": sensor_ref_frames,
               "proposal_boxes": input_boxes, "gt_boxes": gt_boxes, "confidences": confidences, "timesteps": timesteps, "class": obj_class,
               "class_idx": obj_classidx, "labels": tpfp_labels, "instance_ids": gt_instance_ids}
        return res

    def _extract_from_obj_seqs_meta(self, obj_seq, property_name, dtype):
        properties = []
        for obj in obj_seq:
            if obj is not None:
                if getattr(obj, property_name) is not None:
                    properties.append(getattr(obj, property_name))

        if dtype == "tensor":
            if len(properties) == 0 and property_name == "gt_bounding_box":
                properties = torch.zeros(0, 13)
            else:
                properties = torch.stack(properties, dim=0).to(self.device)
        elif dtype == "float":
            properties = torch.tensor(properties, dtype=torch.float).to(self.device)
        elif dtype == "double":
            properties = torch.tensor(properties, dtype=torch.double).to(self.device)
        elif dtype == "string":
            properties = torch.tensor([hash(prop) for prop in properties], dtype=torch.long).to(self.device)

        return properties

    def _format_class_info(self, obj_seq):
        for obj in obj_seq:
            if obj is not None:
                if obj.object_class is not None:
                    class_formatted = obj.object_class
                    break
        if self.dataset_source == "nuscenes":
            class_idx = torch.tensor([NUSCENES_CLASSES[class_formatted]]).to(self.device)
        else:
            class_idx = torch.tensor([WAYMO_CLASSES[class_formatted]]).to(self.device)

        return class_formatted, class_idx

    def _get_label_info(self, obj_seq):  # todo: update me...
        labels = []
        for obj in obj_seq:
            if obj is not None:
                labels.append(getattr(obj, "gt_status") != "false-positive")

        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        return labels

    def format_seq_meta(self, raw_sequence, device):
        """
        Converts numpy array bounding boxes to LidarInstanceBoxes on the appropriate device
        Args:
            raw_sequence:
            device:

        Returns:

        """
        device_raw_seq = []
        for frame_idx in range(len(raw_sequence)):
            if raw_sequence[frame_idx] is None:
                device_raw_seq.append(None)
            else:
                obj_meta = raw_sequence[frame_idx]
                obj_meta = obj_meta.np2torch(device)
                device_raw_seq.append(obj_meta)
        return device_raw_seq

    def _get_frame_masks(self, object_seq):
        haslabel_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        hasframe_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        iskeyframe_mask = torch.zeros(self.seq_len, dtype=torch.bool)

        for i, obj in enumerate(object_seq):
            if obj is not None:
                haslabel_mask[i] = obj.gt_status in ['with-gt']
                iskeyframe_mask[i] = obj.is_keyframe
                hasframe_mask[i] = True

        return haslabel_mask.to(self.device), hasframe_mask.to(self.device), iskeyframe_mask.to(self.device)

    def _get_sensor_refs(self, object_sequence):
        obj_sensor_ref_frames = []
        for obj in object_sequence:
            if obj is not None:
                obj_sensor_ref_frames.append(obj.world2sensor)
        obj_sensor_ref_frames = torch.stack(obj_sensor_ref_frames, dim=0)
        return obj_sensor_ref_frames
