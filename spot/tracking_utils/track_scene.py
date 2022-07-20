import torch
from spot.tracking_utils.association import TrackManager
from spot.ops.transform_utils import translation2affine
from spot.data.box3d import LidarBoundingBoxes
from torch_geometric.data import Batch
from spot.tracking_utils.object_representation import BaseSTObjectRepresentations, EmptySTObjectRepresentations
from spot.io.wandb_utils import viz_bboxes_and_nocs_from_raw_preds
from torch_geometric.loader import DataListLoader
import time
from spot.io.run_utils import split_inputs
import torch.nn.functional as F
import gc
import threading
from spot.data.sequences_dataset import SHARED_SCENE_IDX_COUNTER, SHARED_OMITTED_SCENE_IDX_COUNTER
DEBUG = False


class SceneTracking:

    def __init__(self, scene_idx, dataloader, dataloader_omitted, tpointnet, management_cfg, trajectory_cfg,
                 association_cfg, device, parallel, max_pnts_in_batch, track_startidx, class_name, dataset_source,
                 media_outdir):
        # network properties
        self.tpointnet = tpointnet
        self.device = device
        self.parallel = parallel
        self.max_pnts_in_batch = max_pnts_in_batch
        self.dataset_source = dataset_source
        self.media_outdir = media_outdir

        # dataset properties
        self.scene_idx = scene_idx
        # self.dataloader = dataloader
        # self.dataloader_omitted = dataloader_omitted
        self.dataloader_cache = DataloaderCache(dataloader, dataloader_omitted)

        # box refinement options
        self.refinement_strategy = management_cfg['refinement']['refinement-strategy']
        self.refinement_components = management_cfg['refinement']['refinement-components']
        assert self.refinement_strategy in ['detections-and-tracks', 'tracks-only', 'none']
        assert self.refinement_components in ['confidences-and-boxes', 'confidences-only', 'boxes-only', 'confidences-and-boxes-no-size', 'none']
        self.confidence_anneal_rate = management_cfg['refinement']['confidence-anneal-rate']
        self.auto_regressive_refinement = management_cfg['refinement']['auto-regressive-refinement']

        # nms options
        self.nms_threshold = management_cfg['nms']['threshold']
        self.when_nms = management_cfg['nms']['when']

        # all track configuration parameters
        self.track_trajectory_params = trajectory_cfg.copy()
        self.track_management_params = management_cfg
        self.track_assoc_params = association_cfg

        # create track manager for the scene
        self.track_manager = TrackManager(global_track_counter=track_startidx, class_name=class_name,
                                          management_params=self.track_management_params,
                                          association_params=self.track_assoc_params,
                                          trajectory_params=self.track_trajectory_params,
                                          media_outdir=self.media_outdir)


    def run(self):
        sample_tokens, scene_boxes, scene_confidences, scene_velocities, track_vs_detection = [], [], [], [], []
        scene_trackids, runtimes = [], []
        num_scene_tracks = 0

        num_scene_samples = self.dataloader_cache.dataloader.dataset.num_samples(scene_idx=self.scene_idx)
        assert self.dataloader_cache.dataloader.dataset.cur_sample_idx == 0

        thread = threading.Thread(target=self.dataloader_cache.cache, args=tuple(), daemon=True)
        thread.start()
        for sample_idx in range(num_scene_samples):
            sample_token = self.dataloader_cache.dataloader.dataset.get_sample_token(scene_idx=self.scene_idx, sample_idx=sample_idx)
            sample_tokens.append(sample_token)

            # evaluate the sample
            sample_outputs, thread = self.eval_sample(sample_idx, sample_token, thread, last_sample=sample_idx==num_scene_samples-1)
            success, keyframe_boxes, confidences, track_ids, vels, is_tracking_box, num_dets, runtime = sample_outputs

            if not success:
                scene_boxes.append(None); scene_confidences.append(None); scene_velocities.append(None); scene_trackids.append(None); track_vs_detection.append(None)
            else:
                scene_boxes.append(keyframe_boxes.to('cpu').data.numpy())
                scene_confidences.append(confidences.to('cpu').data.numpy())
                scene_velocities.append(vels.to('cpu').data.numpy())
                scene_trackids.append(track_ids.to('cpu').data.numpy())
                track_vs_detection.append(is_tracking_box.to('cpu').data.numpy())
                num_scene_tracks += num_dets
                if runtime is not None:
                    runtimes.append(runtime)

        print("Average Network Time in Scene Evaluation:")
        if len(runtimes) > 0:
            print(sum(runtimes) / len(runtimes))

        return sample_tokens, scene_boxes, scene_confidences, scene_velocities, scene_trackids, track_vs_detection, num_scene_tracks, runtimes

    def eval_sample(self, sample_idx, sample_token, thread, last_sample=False):
        # get and process refinable detections
        sample_timestep = self.dataloader_cache.dataloader.dataset.get_sample_timestep(scene_idx=self.scene_idx, sample_idx=sample_idx)

        thread.join()
        loaded_data, loaded_ignored_data = self.dataloader_cache.get(self.scene_idx, sample_idx)

        if not last_sample:
            thread = threading.Thread(target=self.dataloader_cache.cache, args=tuple(), daemon=True)
            thread.start()
        # loaded_data, loaded_ignored_data = self.load_data()
        refined_detections = self._load_detections(loaded_data, refine=False)
        other_detections = self._load_detections(loaded_ignored_data, refine=False)

        # skip this sample if we don't have any detections at all
        if refined_detections is None and other_detections is None:
            self.track_manager.update_noobvservations()
            return (False, None, None, None, None, None, None, None), thread

        # merge detections to one baserep
        detections = BaseSTObjectRepresentations.merge(refined_detections, other_detections)

        # run extra nms
        if self.when_nms == "before-refinement":
            detections, _, removed_detections = detections.nms(self.nms_threshold, "max", self.track_trajectory_params)

        # associate detections with existing tracks
        st_detections, instance_ids = self.track_manager.associate(detections, visualize=DEBUG, sample_id=sample_token)
        assert torch.all((st_detections.frame2batchidx[1:] - st_detections.frame2batchidx[:-1]) >= 0)

        # run forward pass on spatio-temporal detections
        unrefined_st_detections = st_detections.clone()
        perform_refinement = not (self.refinement_strategy in ["none"] or self.refinement_components == "none")
        runtime = 0
        if perform_refinement:
            st_detections.enhanced_objects, runtime = self._process_st_detections(st_detections.enhanced_objects, visualize=DEBUG)

        # run NMS after-the-fact
        if self.when_nms == "after-refinement":
            # st_detections, nms_keep_idxs, removed_detections = st_detections.nms(self.nms_threshold, "max", self.track_trajectory_params)
            st_detections, nms_keep_idxs, removed_detections = st_detections.sequence_nms(self.nms_threshold, self.track_trajectory_params)

            instance_ids = instance_ids[nms_keep_idxs]

        # update our stored bounding boxes
        if self.auto_regressive_refinement:
            track_update = st_detections.clone()
            track_update.enhanced_objects.frame_data['confidences'] = unrefined_st_detections.enhanced_objects.frame_data['confidences']
            track_update = track_update.to_base_rep()
        else:
            track_update = unrefined_st_detections.to_base_rep()
        instance_ids = self.track_manager.update(track_update, instance_ids)

        assert torch.all((st_detections.frame2batchidx[1:] - st_detections.frame2batchidx[:-1]) >= 0)

        # get refined detections for this keyframe
        sample_timestep_tnsr = torch.tensor([sample_timestep], dtype=torch.double).to(self.device)
        if self.dataset_source == "waymo":
            st_detections = st_detections.clone().change_basis_global2currentego()
        refined_keyframes = st_detections.estimate_boxes(sample_timestep_tnsr, trajectory_params=self.track_trajectory_params)

        velocities = refined_keyframes[0][0].box[:, 7:10]
        refined_keyframes = refined_keyframes[0][0].box[:, :7]
        refined_confidences = st_detections.get_object_confidences(sample_timestep)
        is_tracking_box = torch.ones(refined_keyframes.size(0), dtype=torch.bool).to(self.device)

        # any tracking-id that equals -1 is not a tracking box
        # non_tracking_boxes = instance_ids == -1
        # is_tracking_box[non_tracking_boxes] = False
        tracking_idxs = instance_ids != -1
        refined_keyframes = refined_keyframes[tracking_idxs]
        refined_confidences = refined_confidences[tracking_idxs]
        instance_ids = instance_ids[tracking_idxs]
        velocities = velocities[tracking_idxs]
        is_tracking_box = is_tracking_box[tracking_idxs]


        return (True, refined_keyframes, refined_confidences, instance_ids, velocities, is_tracking_box, len(detections), runtime), thread

    # def load_data(self):
    #     succeeded, attempts = False, 0
    #     while not succeeded:
    #         try:
    #             loaded_data = list(self.dataloader)
    #             loaded_ignored_data = list(self.dataloader_omitted)
    #             succeeded = True
    #         except RuntimeError:  # occurs in dataloader timeout due which could happen in large system parallelization
    #             self._catch_dataloader_deadlock()
    #             attempts += 1
    #         if attempts > 10:
    #             raise RuntimeError("Faild to load data 10 times.")
    #
    #     return loaded_data, loaded_ignored_data

    def _load_detections(self, objs, refine=False):
        if len(objs) == 0:
            return None

        inputs_list, gt_list, caspr2scene_list = self._format_inputs(objs)
        inputs, point2frameidx, gt, caspr2scene, frame2batch, _ = format_pytorchgeo_data(inputs_list, gt_list, caspr2scene_list, self.device)
        glob2local = translation2affine(-caspr2scene.origin2ego_trans).double() @ caspr2scene.glob2ego.double()

        # set up all data for object representations to store
        object_data = {"global2locals": glob2local, "global2currentego": caspr2scene.glob2ego.double()}
        frame_data = {"boxes": inputs.boxes, "confidences": inputs.confidences, "timesteps": caspr2scene.orig_timesteps, "instance_ids": gt.instance_ids,
                      "gt_boxes": gt.boxes, "with_gt_mask": gt.haslabel_mask, "gt_classification": gt.classification}
        point_data = {"points": inputs.x, "gt_segmentation": gt.segmentation}

        # if refine:
        #     predictions = self._forward_pass(inputs_list)  # was inputs_list
        #     object_data.update(shape_features=predictions['latent_encoding'])
        #     if self.refinement_components == "confidences-only":
        #         frame_data.update(confidences=F.sigmoid(predictions['classification']).flatten())
        #     else:
        #         frame_data.update(boxes=predictions['bbox_regression'], confidences=F.sigmoid(predictions['classification']).flatten())
        #     point_data.update(segmentation=predictions['segmentation'])
        #     detections = ProcessedSTObjectRepresentations(object_data, frame_data, point_data, point2frameidx, frame2batch, ref_frame="local")
        # else:
        detections = BaseSTObjectRepresentations(object_data, frame_data, point_data, point2frameidx, frame2batch, ref_frame="local")

        # move back to global reference frame
        detections = detections.change_basis_local2global()

        return detections

    def _process_st_detections(self, st_detections, visualize=False):
        if isinstance(st_detections, EmptySTObjectRepresentations):
            return st_detections, None
        st_detections = st_detections.change_basis_global2local()
        st_detections_list = st_detections.to_inputs_list()
        st_predictions, runtime = self._forward_pass(st_detections_list, visualize)

        # build new, updated ST detections
        if len(st_predictions) > 0:
            # update latent embedding
            # st_detections.object_data.update(shape_features=st_predictions['latent_encoding'])

            # update confidences based on annealing rate
            predicted_confidences = F.sigmoid(st_predictions['classification']).flatten()
            anneal_confidence = predicted_confidences - st_detections.frame_data['confidences']  # orig=0.7, ours=0.9, diff = 0.2
            anneal_confidence *= self.confidence_anneal_rate

            # update sequence info
            if self.refinement_components == "confidences-only":
                st_detections.frame_data.update(confidences=st_detections.frame_data['confidences'] + anneal_confidence)
            elif self.refinement_components == "boxes-only":
                st_detections.frame_data.update(boxes=st_predictions['bbox_regression'])
            elif self.refinement_components == "confidences-and-boxes-no-size":
                orig_sizes = st_detections.frame_data['boxes'].wlh.clone()
                boxes_update = st_predictions['bbox_regression'].box.clone()
                boxes_update[:, 3:6] = orig_sizes
                boxes_update = LidarBoundingBoxes(box=boxes_update, box_enc=None, size_anchors=st_predictions['bbox_regression'].size_anchors)
                st_detections.frame_data.update(boxes=boxes_update, confidences=st_detections.frame_data['confidences'] + anneal_confidence)
            else:  # boxes and confidences
                st_detections.frame_data.update(boxes=st_predictions['bbox_regression'], confidences=st_detections.frame_data['confidences'] + anneal_confidence)

            # st_detections.point_data.update(segmentation=st_predictions['segmentation'])
            refined_st_detections = st_detections
            refined_st_detections = refined_st_detections.change_basis_local2global()
        else:
            refined_st_detections = EmptySTObjectRepresentations(self.device, st_detections.size_clusters, st_detections.reference_frame)  # empty representation
        return refined_st_detections, runtime

    def _boxes2global(self, boxes, caspr2scene, frame2batch):
        boxes = boxes.translate(caspr2scene.origin2ego_trans, batch=frame2batch)
        boxes = boxes.coordinate_transfer(caspr2scene.glob2ego, batch=frame2batch, inverse=True)
        return boxes

    def _format_inputs(self, inputs):
        # format input info
        ins_formatted = [inputs[j][0].to(self.device) for j in range(len(inputs))]

        # format GT
        gt = [inputs[j][0].labels for j in range(len(inputs))]

        # format Caspr2Scene Info
        caspr2scene = [inputs[j][0].reference_frame for j in range(len(inputs))]
        for j in range(len(inputs)):
            del inputs[j][0].labels
            del inputs[j][0].reference_frame

        return ins_formatted, gt, caspr2scene

    def _forward_pass(self, inputs_list, visualize=False):
        """
        Takes object-sequence inputs and runs refinement network, returning all netowkr predictions as well as the
        global-space bounding boxes
        Args:
            inputs:

        Returns:

        """
        # run through model
        batched_inputs_list = split_inputs(inputs_list, self.max_pnts_in_batch)
        predictions = {}
        runtime = 0
        for i in range(len(batched_inputs_list)):
            t_start = time.time()
            if self.parallel:
                batch_predictions = self.tpointnet(batched_inputs_list[i])
                batch_predictions['bbox_regression'] = LidarBoundingBoxes(None, batch_predictions['bbox_regression'], size_anchors=self.tpointnet.module.size_clusters.to(self.device).float())
            else:
                batch_inputs = Batch.from_data_list(batched_inputs_list[i]).to(self.device)
                batch_predictions = self.tpointnet(batch_inputs)
                if torch.is_tensor(batch_predictions['bbox_regression']):  # check for if original model is training parallel
                    batch_predictions['bbox_regression'] = LidarBoundingBoxes(None, batch_predictions['bbox_regression'], size_anchors=self.tpointnet.size_clusters.to(self.device).float())
            runtime += time.time() - t_start

            update_dict(predictions, batch_predictions)
            if visualize:
                inputs, point2frameidx, _, _, _, _ = format_pytorchgeo_data(batched_inputs_list[i], gt=None, local2scene=None, device=self.device)
                viz_bboxes_and_nocs_from_raw_preds(batch_predictions, inputs, point2frameidx, len(batched_inputs_list[i]), self.media_outdir)


        return predictions, runtime

    # def _catch_dataloader_deadlock(self):
    #     cur_dataset = self.dataloader.dataset
    #     cur_omitted_dataset = self.dataloader_omitted.dataset
    #     num_workers = self.dataloader.num_workers
    #     del self.dataloader
    #     del self.dataloader_omitted
    #     gc.collect()
    #     self.dataloader = DataListLoader(cur_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
    #     self.dataloader_omitted = DataListLoader(cur_omitted_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
    #     time.sleep(10)


def update_dict(dictionary, update):
    for k, val in update.items():
        if k not in dictionary:
            dictionary[k] = val
        elif k in dictionary and torch.is_tensor(val):
            dictionary[k] = torch.cat([dictionary[k], val])
        elif k in dictionary and isinstance(val, LidarBoundingBoxes):
            dictionary[k] = LidarBoundingBoxes.concatenate([dictionary[k], val])
        else:
            pred_list = [torch.cat([dictionary[k][j], val[j]]) for j in range(len(val))]
            dictionary[k] = tuple(pred_list)


def format_pytorchgeo_data(inputs, gt, local2scene, device):
    # get num-frame offsets due to many bounding boxes per sequence
    frame2batch_counts = torch.tensor([inputs[i].boxes.box.size(0) for i in range(len(inputs))]).to(device)
    framecount_offsets = torch.cumsum(frame2batch_counts, dim=0)
    framecount_offsets = torch.cat([torch.zeros(1).to(framecount_offsets), framecount_offsets])[:-1]
    frame2batch = torch.cat([torch.ones(frame2batch_counts[i].item(), dtype=torch.long) * i for i in range(len(frame2batch_counts))]).to(device)

    # update inputs
    inputs = Batch.from_data_list(inputs).to(device)
    point2frameidx = inputs.point2frameidx + framecount_offsets[inputs.batch]
    inputs.boxes = LidarBoundingBoxes.concatenate(inputs.boxes).to(inputs.x)

    # get assigned GT
    if gt is not None:
        gt = Batch.from_data_list(gt).to(device)

        # format gt boxes back to full per-frame
        gt.boxes = LidarBoundingBoxes.concatenate(gt.boxes).to(inputs.x)
        gt_boxes = torch.zeros(gt.haslabel_mask.size(0), 13).to(inputs.x)
        gt_boxes[gt.haslabel_mask] = gt.boxes.box
        gt.boxes = LidarBoundingBoxes(box=gt_boxes, box_enc=None, size_anchors=gt.boxes.size_anchors)

        # format gt segmentation back to full per-frame
        gt_seg = torch.zeros(point2frameidx.size(0), 1).to(gt.segmentation)
        if gt.segmentation.size(0) > 0:
            gt_seg_mask = gt.haslabel_mask[point2frameidx]
            gt_seg[gt_seg_mask] = gt.segmentation
        gt.segmentation = gt_seg

    else:
        gt = None

    # get local-to-scene
    if local2scene is not None:
        local2scene = Batch.from_data_list(local2scene).to(device)
        local2scene.proposal_boxes = LidarBoundingBoxes.concatenate(local2scene.proposal_boxes).to(inputs.x)
        local2scene.glob2ego = local2scene.glob2ego.view(-1, 4, 4)
    else:
        local2scene = None

    return inputs, point2frameidx, gt, local2scene, frame2batch, framecount_offsets



class DataloaderCache:

    def __init__(self, dataloader, dataloader_omitted):
        self.dataloader = dataloader
        self.dataloader_omitted = dataloader_omitted
        self.cur_data, self.cur_data_omitted = None, None
        self.cached = False

    def get(self, scene_idx, sample_idx):
        if not self.cached:
            self.cache()

        self.cached = False
        # Important: As the dataset is copied to each thread, this is just for state-keeping in the main thread!
        if self.dataloader.num_workers > 0:
            self.dataloader.dataset.next()
            self.dataloader_omitted.dataset.next()

        # Check that we got a valid sample
        success = True
        for val in SHARED_SCENE_IDX_COUNTER.values():
            success &= (val[0] == scene_idx and val[1] == sample_idx)
        for val in SHARED_OMITTED_SCENE_IDX_COUNTER.values():
            success &= (val[0] == scene_idx and val[1] == sample_idx)
        if not success:
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            print("Our dataloader deviated from expected scene indices!")
            print("Resetting....")
            print("!!!!!!!!!!!!!!!!!!!!!!!!")
            self.reset_to_scene_sample_idxs(scene_idx, sample_idx)
            return self.get(scene_idx, sample_idx)

        return self.cur_data, self.cur_data_omitted

    def cache(self):
        succeeded, attempts = False, 0
        while not succeeded:
            try:
                loaded_data = list(self.dataloader)
                loaded_ignored_data = list(self.dataloader_omitted)
                succeeded = True
            except RuntimeError:  # occurs in dataloader timeout due which could happen in large system parallelization
                self._catch_dataloader_deadlock()
                attempts += 1
            if attempts > 10:
                raise RuntimeError("Failed to load data 10 times.")

        self.cur_data, self.cur_data_omitted = loaded_data, loaded_ignored_data
        self.cached = True

    def _catch_dataloader_deadlock(self):
        cur_dataset = self.dataloader.dataset
        cur_omitted_dataset = self.dataloader_omitted.dataset
        num_workers = self.dataloader.num_workers
        del self.dataloader
        del self.dataloader_omitted
        gc.collect()
        self.dataloader = DataListLoader(cur_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
        self.dataloader_omitted = DataListLoader(cur_omitted_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
        time.sleep(10)

    def reset_to_scene_sample_idxs(self, scene_idx, sample_idx):
        cur_dataset = self.dataloader.dataset
        cur_omitted_dataset = self.dataloader_omitted.dataset
        num_workers = self.dataloader.num_workers
        del self.dataloader
        del self.dataloader_omitted
        gc.collect()

        # reset main-thread scene-indices
        cur_dataset.cur_scene_idx = scene_idx
        cur_dataset.cur_sample_idx = sample_idx
        cur_omitted_dataset.cur_scene_idx = scene_idx
        cur_omitted_dataset.cur_sample_idx = sample_idx

        # re-spawn dataloader threads
        self.dataloader = DataListLoader(cur_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)
        self.dataloader_omitted = DataListLoader(cur_omitted_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=num_workers > 0)