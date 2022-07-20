import numpy as np
import torch
from spot.io.wandb_utils import get_wandb_boxes, get_coord_frame, UNIT_BOX_CORNERS, get_boxdir_points
import wandb
from spot.ops.torch_utils import torch_to_numpy
from spot.io.globals import NUSCENES_CLASSES
from pyquaternion import Quaternion
from spot.ops.data_utils import quaternion_yaw


class DataLoaderVisualization:

    def __init__(self, scene_objects_container):
        self.scene_objects_container = scene_objects_container

    def viz_sequences(self, inputs, gt, objref2sceneref, bbox_coder, loading_params, sample_id=None):
        wandb.init("spot-loading-viz")
        # seq_data = self.scene_objects_container.get_sequence(idxs, bbox_coder, loading_params, train_aug_params)
        # inputs, gt, objref2sceneref = seq_data
        input_pc, input_boxes = inputs['points'], inputs['boxes']
        n_objseqs = input_pc.size(0)
        class_idx = NUSCENES_CLASSES[self.scene_objects_container.object_type]

        all_inputs, all_crop_bboxes, all_gt_bboxes = [], [], []
        formatted_boxes = []
        for seq_idx in range(n_objseqs):
            negative_sample = gt['classification'][seq_idx] < 1
            # get input points for the sequence
            seq_input = input_pc[seq_idx, :, :, :]
            seq_haslabel = gt['haslabel_mask'][seq_idx]
            if loading_params['load-gt'] and not negative_sample:
                nocs_output = gt['tnocs'][seq_idx][seq_haslabel]
                nocs_mask = gt['segmentation'][seq_idx][seq_haslabel]

            # get tensor of all GT bounding boxes
            if loading_params['load-gt'] and not negative_sample:
                seq_gtboxes = gt['boxes'][seq_idx][seq_haslabel]
                seq_gtboxes = bbox_coder.decode_caspr(seq_gtboxes[:, :8], seq_gtboxes[:, 8:], class_idx)
                seq_gtboxes = bbox_coder.bbox_mmdet2caspr(seq_gtboxes)
            else:
                seq_gtboxes = torch.zeros((0, 7)).to(loading_params['device'])

            # get tensor of all input-crop bounding boxes
            seq_frame_indicator = objref2sceneref['hasframe_mask'][seq_idx]
            seq_origbboxes = input_boxes[seq_idx][seq_frame_indicator]
            seq_origbboxes = bbox_coder.decode_caspr(seq_origbboxes[:, :8], seq_origbboxes[:, 8:], class_idx)
            seq_origbboxes = bbox_coder.bbox_mmdet2caspr(seq_origbboxes)

            # visualize
            self._viz_seq(seq_input, seq_origbboxes, seq_gtboxes)
            if loading_params['load-gt'] and not negative_sample:
                self._viz_nocs(seq_input, nocs_output, nocs_mask)

            # get all values in scene ref-frame
            local2global = objref2sceneref["origin2ego_trans"][seq_idx].view(1, 1, 3)
            seq_input_globref, seq_gtboxes_globref, seq_origbboxes_globref = seq_input, seq_gtboxes, seq_origbboxes
            seq_input_globref[:, :, :3] += local2global
            seq_gtboxes_globref[:, :3] += local2global.squeeze(0)
            seq_origbboxes_globref[:, :3] += local2global.squeeze(0)

            # get it all in global ref-frame now!
            ego2glob_affine = torch.inverse(objref2sceneref['glob2ego'][seq_idx])  # 4 x 4 tensor

            # format predicted global boxes
            all_prop_boxes = seq_origbboxes_globref.view(-1, 7)
            for k in range(all_prop_boxes.size(0)):
                # transform whole thing!
                dims = all_prop_boxes[k, 3:6]
                center = (all_prop_boxes[k:k+1, :3].double() @ ego2glob_affine[:3, :3].T)
                center += ego2glob_affine[:3, 3:4].T
                center = center.float()
                quat = Quaternion(axis=[0, 0, 1], radians=all_prop_boxes[k, 6])
                glob_quat = Quaternion(matrix=ego2glob_affine[:3, :3].to('cpu').data.numpy() @ quat.rotation_matrix)
                yaw = torch.tensor([quaternion_yaw(glob_quat)])
                formatted_box = torch.cat([center.flatten(), dims, yaw])
                formatted_boxes.append(formatted_box.to('cpu').data.numpy().reshape(7))

            # record scene ref-frame values
            all_inputs.append(seq_input_globref)
            all_crop_bboxes.append(seq_origbboxes_globref)
            all_gt_bboxes.append(seq_gtboxes_globref)

        formatted_boxes = np.stack(formatted_boxes)

        # glob_gt_boxes = get_glob_nusc_boxes(sample_id)
        # wandb.init("caspr_data")
        # viz_pred_boxes(formatted_boxes, glob_gt_boxes, name="Full-Pipeline Pred Boxes")

        all_inputs = torch.cat(all_inputs)
        all_crop_bboxes = torch.cat(all_crop_bboxes)
        all_gt_bboxes = torch.cat(all_gt_bboxes)
        self._viz_scene(all_inputs, all_crop_bboxes, all_gt_bboxes)
        wandb.finish()

    def _viz_seq(self, seq_input, input_bboxes, gt_bboxes):
        """
        Provides visualization for sequence returned in get_sequence.
        """
        in_wandb_boxes, in_dir_points, in_dir_point_colors, gt_wandb_boxes, gt_dir_points, gt_dir_point_colors = \
            self._get_wandb_boxes(input_bboxes, gt_bboxes)
        coord_frame, coord_frame_colors = get_coord_frame()

        seq_viz = seq_input.to('cpu').data.numpy()
        if seq_viz.shape[2] == 4:
            seq_colors = (seq_viz[:, :, 3] / max(0.05, np.max(seq_viz[:, :, 3]))).reshape(-1, 1) * np.array([[150, 75, 255]])
            seq_viz = seq_viz[:, :, :3].reshape(-1, 3)
        else:
            seq_colors = np.ones((seq_viz.shape[0] * seq_viz.shape[1], 1)) * np.array([[0, 255, 255]])
            seq_viz = seq_viz.reshape(-1, 3)
        full_viz = np.hstack((np.vstack((seq_viz, in_dir_points, gt_dir_points, coord_frame)), np.vstack((seq_colors, in_dir_point_colors, gt_dir_point_colors, coord_frame_colors))))
        wandb.log({"Visualize Loaded Data": wandb.Object3D({
            "type": "lidar/beta",
            "points": full_viz,
            "boxes": np.array(in_wandb_boxes.tolist() + gt_wandb_boxes.tolist())})})

    def _viz_nocs(self, seq_input, seq_nocs, seq_masks):
        """
        Provides visualization for sequence returned in get_sequence.
        """
        coord_frame, coord_frame_colors = get_coord_frame()
        coord_frame = coord_frame /2 + np.array([[2.0, 0.0, 0.0]])
        seq_input, seq_nocs, seq_masks = torch_to_numpy([seq_input, seq_nocs, seq_masks])

        seq_input_viz = seq_input[:, :, :3].reshape(-1, 3) - np.array([[-5.0, 0.0, 0.0]])
        seq_colors = (seq_input[:, :, 3] / np.max(seq_input[:, :, 3])).reshape(-1, 1) * np.array([[150, 75, 255]])

        seq_nocs_viz = seq_nocs[:, :, :3].reshape(-1, 3)
        nocs_colors = np.floor((seq_nocs_viz * 255 / 2) + 127) * seq_masks[:, :].reshape((-1, 1))
        nocs_box = {'corners': (np.array(UNIT_BOX_CORNERS) - np.array([[0.5, 0.5, 0.5]])).tolist(),
                    'label': "NOCS",
                    "color": [255, 255, 255]}

        full_viz = np.hstack((np.vstack((seq_input_viz, seq_nocs_viz, coord_frame)), np.vstack((seq_colors, nocs_colors, coord_frame_colors))))
        wandb.log({"Visualize NOCS Data": wandb.Object3D({
            "type": "lidar/beta",
            "points": full_viz,
            "boxes": np.array([nocs_box])})})

    def _viz_scene(self, scene_pts, input_bboxes, gt_bboxes):
        in_wandb_boxes, in_dir_points, in_dir_point_colors, gt_wandb_boxes, gt_dir_points, gt_dir_point_colors = \
            self._get_wandb_boxes(input_bboxes, gt_bboxes)
        coord_frame, coord_frame_colors = get_coord_frame()

        scene_viz = scene_pts[:, :, :3].view(-1, 3).to('cpu').data.numpy()
        scene_clrs = (scene_pts[:, :, 3].to('cpu').data.numpy() / np.max(scene_pts[:, :, 3].to('cpu').data.numpy())).reshape(-1, 1) * np.array([[100, 175, 60]])
        full_viz = np.hstack((np.vstack((scene_viz, in_dir_points, gt_dir_points, coord_frame)), np.vstack((scene_clrs, in_dir_point_colors, gt_dir_point_colors, coord_frame_colors))))
        wandb.log({"Visualize Processed Scene Data": wandb.Object3D({
            "type": "lidar/beta",
            "points": full_viz,
            "boxes": np.array(in_wandb_boxes.tolist() + gt_wandb_boxes.tolist())})})

    def _get_wandb_boxes(self, input_bboxes, gt_bboxes):
        in_bboxes_np, gt_bboxes_np = torch_to_numpy([input_bboxes, gt_bboxes])

        in_wandb_boxes = get_wandb_boxes(in_bboxes_np, colors=np.array([[255, 0, 255]]).repeat(in_bboxes_np.shape[0], axis=0), scale_factor=1, viz_offset=0)
        in_dir_points, in_dir_point_colors = get_boxdir_points(in_bboxes_np, colors=np.array([[255, 0, 255]]).repeat
            (in_bboxes_np.shape[0], axis=0), scale_factor=1, viz_offset=0)

        gt_wandb_boxes = get_wandb_boxes(gt_bboxes_np, colors=np.array([[0, 255, 0]]).repeat(gt_bboxes_np.shape[0], axis=0), scale_factor=1, viz_offset=0)
        gt_dir_points, gt_dir_point_colors = get_boxdir_points(gt_bboxes_np, colors=np.array([[0, 255, 0]]).repeat
            (gt_bboxes_np.shape[0], axis=0), scale_factor=1, viz_offset=0)

        return in_wandb_boxes, in_dir_points, in_dir_point_colors, gt_wandb_boxes, gt_dir_points, gt_dir_point_colors