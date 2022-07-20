import numpy as np
import torch
import os
import os.path as osp
from zipfile import ZipFile
from spot.data.loading.instance import ObjectProposal
from spot.data.loading.loader import CasprObjectSequences
from spot.data.loading.sequences import Sequences
from preprocessing.common.points import ObjectPoints
import time


class SceneIO:

    def __init__(self, scene_id, base_dir=''):

        # =============================================================
        # ================ Preprocessing Internal Rep =================
        # =============================================================
        self.scene_id = scene_id
        self.duplicate_instances_count = 0

        # Internal rep for preprocessing samples
        self.frames = []  # list[dict[instance_id --> :obj:`ObjectProposal`],
        self.timesteps = []
        self.keyframe_ids = []

        # Set base dir if applicable; if is empty, we assume we're loading
        if base_dir:
            self.save_dir = osp.join(base_dir, scene_id)
            if not osp.isdir(self.save_dir):
                os.makedirs(self.save_dir)
            self.zipfile_out = ZipFile(self.save_dir + ".zip", "w")
        else:
            self.save_dir = ''  # initializing

    def add_frame(self, boxes, points, keyframe_id, world2sensor, is_keyframe=False):
        """
        Args:
            boxes (list<Box3D>): ordered list of all boxes in this frame
            points (ObjectPoints): contains ordered list of all box lidar-point information
            keyframe_id (str): unique identifier of the closest keyframe
            world2sensor (np.ndarray): 4x4 affine transformation from world-reference to sensor reference
            is_keyframe (bool): is this frame a keyframe (ie used during official evaluation protocols)
        Returns:
        """
        for i, box in enumerate(boxes):
            # extract GT status and if applicable, give a unique GT instance ID
            gt_status = box.get_gt_status()
            if gt_status == "false-positive":
                box.gt_instance_id = "False-Positive-%s" % box.instance_id

            # format prediction boxes
            box_array = box.to_array().flatten()

            # format gt boxes
            gt_box_array = box.gt_to_array().flatten() if gt_status == "with-gt" else None

            self.add(box.timestep,
                     keyframe_id,
                     world2sensor,
                     is_keyframe,
                     box.instance_id,
                     box_array,
                     box.class_name,
                     box.score,
                     gt_status,
                     box.gt_instance_id,
                     gt_box_array,
                     box.gt_is_moving,
                     points.points[i],
                     points.num_points[i],
                     points.segmentation[i],
                     points.surrounding_context_factor)

    def add(self,
            timestep,
            keyframe_id,
            world2sensor,
            is_keyframe,
            instance_id,
            bbox,
            object_class,
            confidence,
            gt_status,
            gt_instance_id,
            gt_bbox,
            gt_is_moving,
            points,
            num_points,
            segmentation,
            surrounding_context_factor):
        """
        Add a new observation of an object to the frame that is closest to sample_timestep.
        Args:
            timestep (float): Time of sample observation, in seconds.
            world2sensor (np.ndarry): 4x4 affine transformation from global to lidar reference frame
            bbox (np.ndarray): Bounding box data that was use in this crop. Format is input to LiDARInstance3DBoxesCaspr
            gt_bbox (np.ndarray): Corresponding ground truth bounding box for this object. Format is input to LiDARInstance3DBoxesCaspr
            gt_status (str): One of ['with-gt', 'in-gt-track', 'false-positive']
            num_points (int): Approximate number of points that belong to the object
        """
        closest_sample_idx = torch.argmin(torch.abs(torch.tensor(self.timesteps, dtype=torch.double) - timestep)).item()
        point_data_path = self.record_object_points(timestep, keyframe_id, instance_id, points, segmentation)

        # convert bounding boxes to numpy arrays (keeping as LidarBboxes takes too much memory)
        object_prop = ObjectProposal(timestep=timestep, keyframe_id=keyframe_id, world2sensor=world2sensor,
                                     is_keyframe=is_keyframe, instance_id=instance_id, bounding_box=bbox,
                                     object_class=object_class, confidence=confidence, gt_status=gt_status,
                                     gt_instance_id=gt_instance_id, gt_bounding_box=gt_bbox, gt_is_moving=gt_is_moving,
                                     point_data_path=point_data_path, num_pts=num_points,
                                     surrounding_context_factor=surrounding_context_factor)

        if instance_id in self.frames[closest_sample_idx]:
            self.duplicate_instances_count += 1
            print("Error! Duplicate %s" % self.duplicate_instances_count)
        self.frames[closest_sample_idx][instance_id] = object_prop

    def record_object_points(self, timestep, keyframe_id, instance_id, points, segmentation):
        point_data_path = osp.join(self.save_dir, "%s_%s_%s.npz" % (keyframe_id, instance_id, timestep))
        np.savez(point_data_path, points=points, segmentation=segmentation)
        self.zipfile_out.write(point_data_path, arcname="%s_%s_%s.npz" % (keyframe_id, instance_id, timestep))
        return point_data_path

    def close_filesystem_pipe(self):
        """
        If adding data to filesystem, call this once done.
        """
        self.zipfile_out.close()
        self.zipfile_out = None

    def init_new_frame(self, timestep, keyframe_id):
        self.frames.append({})
        self.keyframe_ids.append(keyframe_id)
        self.timesteps.append(timestep)

    def init_new_sample_if_not_exist(self, timestep, keyframe_id, threshold=0.001):
        if len(self.timesteps) == 0:
            closest_sample_tdiff = np.inf
        else:
            closest_sample_tdiff = torch.min(torch.abs(torch.tensor(self.timesteps, dtype=torch.double) - timestep)).item()

        if closest_sample_tdiff < threshold:
            return
        else:
            self.frames.append({})
            self.keyframe_ids.append(keyframe_id)
            self.timesteps.append(timestep)

    def dump_to_dict(self):
        frames_serialized = []
        for sample_dict in self.frames:
            serialized_sample_dict = {inst_id: object_prop.dump_to_dict() for inst_id, object_prop in sample_dict.items()}
            frames_serialized.append(serialized_sample_dict)

        serialization = {'frames': frames_serialized}
        save_attrs = ['scene_id', 'timesteps', 'keyframe_ids', 'save_dir', 'duplicate_instances_count']
        for save_attr in save_attrs:
            serialization[save_attr] = getattr(self, save_attr)
        return serialization

    def load_from_dict(self, load_dict):
        for k, val in load_dict.items():
            if k == 'frames':
                frames = []
                for frames_serialized in val:
                    frame = {}
                    for inst_id, serial_obj_prop in frames_serialized.items():
                        obj_prop = ObjectProposal()
                        obj_prop.load_from_dict(serial_obj_prop)
                        frame[inst_id] = obj_prop
                    frames.append(frame)
                setattr(self, k, frames)
            else:
                setattr(self, k, val)




class SceneParser:

    def __init__(self, saved_scene, object_type, dataset_source, loading_params, tracking_baseline):
        # transfer info from saved scene
        self.scene_id = saved_scene.scene_id
        self.frames = saved_scene.frames
        self.all_timesteps = saved_scene.timesteps
        self.repeated_keyframe_ids = saved_scene.keyframe_ids
        self.data_dir = saved_scene.save_dir
        self.dataset_source = dataset_source

        # store loading info
        self.tracking_baseline = tracking_baseline
        self.object_type = object_type

        # load info on sequence properties
        self.loading_params = loading_params
        self.allow_false_detections = loading_params['sequence-properties']['allow-false-positive-frames']

        # create list of unique sample IDs (because currently we ~10x repeats from lidar sweeps)
        self.keyframe_ids, self.keyframe_timesteps = self._compute_sample_ids_and_timesteps()

    def sort_samples(self):
        if len(self.frames) == 0:
            return
        sorted_frame_idxs = np.argsort(self.all_timesteps)
        self.frames = [self.frames[sort_idx] for sort_idx in sorted_frame_idxs]
        self.repeated_keyframe_ids = [self.repeated_keyframe_ids[sort_idx] for sort_idx in sorted_frame_idxs]
        self.all_timesteps = [self.all_timesteps[sort_idx] for sort_idx in sorted_frame_idxs]
        self.keyframe_ids, self.keyframe_timesteps = self._compute_sample_ids_and_timesteps()

    def _compute_sample_ids_and_timesteps(self):
        # create list of unique sample IDs (because currently we ~10x repeats from lidar sweeps)
        np_rep_keyframe_ids, np_all_timesteps = np.array(self.repeated_keyframe_ids), np.array(self.all_timesteps)
        keyframe_ids, u_idxs, inv_idxs = np.unique(np_rep_keyframe_ids, return_index=True, return_inverse=True)

        # sample timestep is the MAXIMUM time of all sample observances
        keyframe_timesteps = np.array([np.max(np_all_timesteps[inv_idxs == i]) for i in range(len(keyframe_ids))])

        # re-order into sorted version
        sort_idxs = np.argsort(u_idxs)
        keyframe_ids = keyframe_ids[sort_idxs].tolist()
        sample_timesteps = keyframe_timesteps[sort_idxs].tolist()
        return keyframe_ids, sample_timesteps

    def parse(self, train_aug_params):
        self.sort_samples()

        # samples, sample_ids, timesteps, object_type, sequence_params, tracking_baseline
        sequence_loader = Sequences(self.frames, self.repeated_keyframe_ids, self.all_timesteps, self.object_type, self.loading_params, self.tracking_baseline)
        if self.allow_false_detections:
            seqs, seq2keyframe, removed_seqs, removed_seq2keyframe = sequence_loader.prepare_sequences(gt_statuses=('with-gt', 'in-gt-sequence', 'false-positive'))
        else:
            seqs, seq2keyframe, removed_seqs, removed_seq2keyframe = sequence_loader.prepare_sequences(gt_statuses=('with-gt', 'in-gt-sequence'))

        loader = CasprObjectSequences(data_dir=self.data_dir, class_name=self.object_type, loading_params=self.loading_params, train_aug_params=train_aug_params, dataset_source=self.dataset_source)
        scene_objects = SceneObjects(self.scene_id, seqs, seq2keyframe, self.keyframe_ids, self.keyframe_timesteps, loader)
        ignored_scene_objects = SceneObjects(self.scene_id, removed_seqs, removed_seq2keyframe, self.keyframe_ids, self.keyframe_timesteps, loader)
        return scene_objects, ignored_scene_objects


class SceneObjects:

    def __init__(self, scene_id, sequences, sequence2keyframe, keyframe_ids, keyframe_timesteps, loader):
        self.scene_id = scene_id
        self.sequences = sequences
        self.sequence2keyframe = sequence2keyframe
        self.keyframe_ids = keyframe_ids
        self.keyframe_timesteps = keyframe_timesteps
        self.loader = loader

    def get(self, idx):

        """

        Args:
            idxs:
        Returns:

        """
        raw_sequence = self.sequences[idx]
        sequences = self.loader.generate_caspr_seq(raw_sequence)

        return sequences

    def keyframeidx2seqidxs(self, sample_idx):
        if len(self.sequence2keyframe) == 0:
            return np.array([])
        keyframe_id = self.keyframe_ids[sample_idx]
        idxs = np.where(np.array(self.sequence2keyframe) == np.array(keyframe_id))[0]
        idxs = idxs.tolist()
        return idxs

    @property
    def num_sequences(self):
        if self.sequences is None:
            return 0
        return len(self.sequences)

    @property
    def num_samples(self):
        return len(self.keyframe_ids)

