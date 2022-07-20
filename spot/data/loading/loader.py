import torch
import numpy as np

from spot.data.loading.pipeline.load_proposals import ProposalLoader
from spot.data.loading.pipeline.load_points import PointLoader
from spot.data.loading.pipeline.ego_transform import EgomotionAggregate
from spot.data.loading.pipeline.augment import AugmentSequences, AugmentBoxes
from spot.data.loading.pipeline.center_to_origin import OriginCenter
from spot.data.loading.pipeline.tensorize import TensorizePoints, PointCanonicalizer
from torch_geometric.data.data import Data as GeometricData
from spot.io.globals import NUSCENES_CLASSES, NUSCENES_BINNING_CLUSTER_SIZES, WAYMO_CLASSES, WAYMO_BINNING_CLUSTER_SIZES
from zipfile import ZipFile
import io
import os

class CasprObjectSequences:

    def __init__(self, data_dir, class_name, loading_params, train_aug_params, dataset_source, device='cpu'):

        """
        Args:
            object_seqs_meta (list[list[:ObjectProposal:]]: Outer list is a batch of sequences. Inner list
                                                                is a sequence of object-frames.
        """
        self.device = device
        self.data_dir = data_dir

        # necessary sequence filter parameters
        self.min_frame_pts = loading_params['sequence-filters']['min-frame-pts']
        self.min_seq_pts = loading_params['sequence-filters']['min-seq-pts']
        self.min_labeled_pts = loading_params['sequence-filters']['min-labeled-pts']
        self.overlap_threshold = loading_params['sequence-filters']['overlap-threshold']

        # loading parameters
        self.seq_len = loading_params['sequence-properties']['sequence-length']
        self.num_pts = loading_params['sequence-properties']['max-num-pts']
        self.dataset_source = dataset_source
        if dataset_source == "nuscenes":
            self.size_groupings = torch.tensor(NUSCENES_BINNING_CLUSTER_SIZES[NUSCENES_CLASSES[class_name]]).to(device)
        else:
            self.size_groupings = torch.tensor(WAYMO_BINNING_CLUSTER_SIZES[WAYMO_CLASSES[class_name]]).to(device)

        self.num_size_groupings = self.size_groupings.size(0)
        self.backwards_time = loading_params['sequence-properties']['backward']
        self.ego_corrected_coords = loading_params['sequence-properties']['ego-correct-coordinates']

        # training parameters
        self.train_aug_params = train_aug_params

        # zipfile with data
        self.use_zipfile = loading_params['dataset-properties']['use-zipfile']
        if self.use_zipfile:
            with open(self.data_dir + ".zip", "rb") as f:
                zip_contents = f.read()
            self.zipfile = ZipFile(io.BytesIO(zip_contents))
        else:
            self.zipfile = None

    def read_data(self, filename):
        if self.use_zipfile:
            data = np.load(self.zipfile.open(filename))
        else:
            data = np.load(os.path.join(self.data_dir, filename))
        return data

    def generate_caspr_seq(self, raw_sequence):
        """
        Given a raw sequence of <ObjectProposal>,
        Args:
            raw_sequence list[ObjectProposal]: List of ObjectProposal objects or None if the frame is empty
            label: binary variable, True indicates this is the object, False otherwise

        """
        # sort object-seqs-meta
        sorted_object_seq = self._sort_raw_sequence(raw_sequence)

        # load proposals
        proposal_loader = ProposalLoader(self.device, self.seq_len, self.size_groupings, self.dataset_source)
        proposals = proposal_loader.load_proposals(sorted_object_seq)

        # load points
        point_loader = PointLoader(self.data_dir, self.device, self.read_data)
        points = point_loader.load_points(sorted_object_seq, proposals['hasframe_mask'], proposals['haslabel_mask'])

        # ego-correct points and proposals --> store ego-info
        if self.ego_corrected_coords:
            egomotion_correction = EgomotionAggregate(self.seq_len, self.device)
            proposals, points = egomotion_correction.egomotion_aggregate(proposals, points)

        # training augmentations
        if self.train_aug_params['train-augment']:
            augmenter = AugmentSequences(self.train_aug_params, self.seq_len, self.min_seq_pts, self.overlap_threshold, self.device)
            proposals, points = augmenter.augment(proposals, points)

        # add bounding box noise
        augment_boxes = self.train_aug_params['train-augment-bboxes']
        box_augmenter = AugmentBoxes(augment_boxes, self.train_aug_params, self.seq_len, self.min_seq_pts, self.device)
        proposals = box_augmenter.augment(proposals)

        # sequence-normalization (center)
        centering = OriginCenter(self.seq_len, self.backwards_time)
        proposals, points = centering.origin_center(proposals, points)

        # tensorize (points; boxes)
        tensorize_points = TensorizePoints(self.seq_len, self.num_pts, self.device)
        points = tensorize_points.points2tensor(proposals, points)

        # tensorize boxes
        point_canonicalizer = PointCanonicalizer(self.seq_len, self.num_pts, self.num_size_groupings, self.device)
        points = point_canonicalizer.canonicalize_points(proposals, points)

        # format everything
        out = self.format_output(proposals, points)
        return out

    def format_output(self, proposals, points):
        # format output labels
        output_dict = {"boxes": proposals["gt_boxes"], "tnocs": points["gt_nocs"], "keyframe_mask": proposals["keyframe_mask"],
                       "segmentation": points["gt_segmentation"], "classification": proposals["labels"],
                       "haslabel_mask": proposals["haslabel_mask"], "instance_ids": proposals["instance_ids"]}
        gt_labels = GeometricData(**output_dict)

        # format local-ref-frame to scene-ref-frame info
        local2scene_dict = {"origin2ego_trans": proposals["caspr2scene_trans"],
                            "glob2ego": proposals["glob2sensor"], "proposal_tnocs": points["proposed_nocs"],
                            "proposal_boxes": proposals["proposal_boxes"],
                            "confidences": proposals["confidences"], "class_idxs": proposals["class_idx"],
                            "orig_timesteps": proposals["orig_timesteps"]}
        local2scene = GeometricData(**local2scene_dict)

        # combine into a single pytorch_geometric object
        sequence_data = GeometricData(x=points["points"], boxes=proposals["input_boxes"], point2frameidx=points["point2frameidx"],
                                      confidences=proposals["confidences"],
                                      labels=gt_labels, reference_frame=local2scene)
        return sequence_data

    def _sort_raw_sequence(self, raw_seq):
        times = np.zeros(len(raw_seq), dtype=np.double)
        for i, obj in enumerate(raw_seq):
            if obj is not None:
                times[i] = obj.timestep
            else:
                times[i] = np.inf
        sort_idxs = np.argsort(times)
        sorted_obj_seq = [raw_seq[idx] for idx in sort_idxs]
        return sorted_obj_seq