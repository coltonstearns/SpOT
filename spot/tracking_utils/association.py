import torch
from spot.tracking_utils.helpers import greedy_match_confidences_detections, greedy_match_confidences_tracks
from scipy.optimize import linear_sum_assignment
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from spot.io.wandb_utils import get_wandb_boxes, get_boxdir_points
import wandb
import seaborn
from spot.tracking_utils.association_metrics import AssociationL1Distance, ConfidenceRescorer
from spot.tracking_utils.object_representation import MergedEnhancedAndBaseObjectRepresentations, BaseSTObjectRepresentations
from spot.ops.torch_utils import torch_to_numpy
import os
NUM_CLRS = 30

DEBUG = False

class TrackManager:

    def __init__(self, global_track_counter, class_name, management_params, association_params, trajectory_params, media_outdir):

        self.tracks = []

        # track management parameters
        self.global_track_counter = global_track_counter
        self.class_name = class_name
        self.max_without_hit = management_params['track-maintenance']["no-hits-kill-thresh"]
        self.min_hits = management_params['track-maintenance']["min-hits-usability"]
        self.max_time_window = management_params['track-maintenance']["max-time-window"]
        self.points_context = management_params['track-maintenance']["max-seq-pts"]
        self.track_init_threshold = management_params['track-maintenance']['track-initialization-conf-thresh']
        self.media_outdir = media_outdir

        # track refinement parameters
        self.confidence_refinement_thresh = management_params['refinement']["confidence-refinement-threshold"]
        self.refinement_strategy = management_params['refinement']['refinement-strategy']
        self.refinement_components = management_params['refinement']['refinement-components']
        self.min_refinement_age = management_params['refinement']['age-refinement-threshold']

        # track objects bookkeeping
        self.tracks = None
        self.track_ids = None
        self.track_ages = None
        self.track_frames_since_hit = None

        # instantiate association metrics
        self.l1_scorer = AssociationL1Distance(association_params, trajectory_params, class_name)
        self.confidence_rescorer = ConfidenceRescorer(association_params['matching']['greedy-sort-by'])

        # l1 association parameters
        self.match_algorithm = association_params['matching']['algorithm']
        self.greedy_match_by = association_params['matching']['greedy-sort-by']
        self.trajectory_params = trajectory_params

    def associate(self, detections, visualize=False, sample_id=""):
        """
        """

        device = detections.device

        # start with this for detections
        if self.tracks is None or len(self.tracks) == 0:
            instance_ids = torch.ones(len(detections)).to(device) * -1
            detections = MergedEnhancedAndBaseObjectRepresentations(None, detections, detections.device)
            return detections, instance_ids

        if visualize:
            self.viz_associations_pre(detections)

        # compute L1 distances
        distance_matrix = self.l1_scorer.compute_l1_matrix(self.tracks, detections)

        # Merge L1 and shape affinities; re-order to also weight in detection and track confidences
        distance_matrix, idx_map = self.confidence_rescorer.rescore(self.tracks, detections, distance_matrix)

        # run association scheme
        matches = self._associate(distance_matrix, idx_map)

        # visualize our association pipeline
        if visualize:
            self.viz_associations_post(detections, matches, sample_id)

        # update detections to include matched spatio-temporal context
        st_detections, track_ids, det_ages = self.build_detection_sequences(detections, matches)
        assert torch.all((st_detections.frame2batchidx[1:] - st_detections.frame2batchidx[:-1]) >= 0)

        # clip spatio-temporal context
        st_detections = st_detections.clip_sequences(time_context=self.max_time_window, points_context=self.points_context)

        # update into refinable vs not-refinable
        st_detections, track_ids = self.separate_refinable_vs_other(st_detections, track_ids, det_ages)

        return st_detections, track_ids

    def _associate(self, distance_matrix, idxs_map):
        # get matches; matched_indices is a (Kx2) numpy array of [det-idx, track-idx]
        if self.match_algorithm == 'greedy':
            if self.greedy_match_by == "detections":
                matched_indices = greedy_match_confidences_detections(distance_matrix.to('cpu').data.numpy())
            else:
                matched_indices = greedy_match_confidences_tracks(distance_matrix.to('cpu').data.numpy())
        elif self.match_algorithm == "hungarian":
            matched_indices = linear_sum_assignment(distance_matrix.to('cpu').data.numpy())  # hungarian algorithm
        else:
            raise RuntimeError("Matching algorithm is not one of the following: ['greedy', 'hungarian-threshold', 'hungarian']")

        # filter out matched with low IOU
        matches = {}
        for m in matched_indices:
            if self.greedy_match_by == "detections":
                det_idx = idxs_map[m[0]].item()
                matches[det_idx] = m[1]
            else:
                track_idx = idxs_map[m[1]].item()
                matches[m[0]] = track_idx
        return matches

    def build_detection_sequences(self, detections, matches):
        if len(matches) == 0:
            return detections, -torch.ones(len(detections)), torch.ones(len(detections))

        matched_det_idxs, matched_track_idxs = [], []
        for det_idx, track_idx in matches.items():
            matched_det_idxs.append(det_idx); matched_track_idxs.append(track_idx)
        matched_det_idxs = torch.tensor(matched_det_idxs); matched_track_idxs = torch.tensor(matched_track_idxs)

        # sort based on track indices, as __getitem__ does NOT guarentee correct order for MergedRep!
        matched_track_idxs, track_sort_idxs = torch.sort(matched_track_idxs)
        matched_det_idxs = matched_det_idxs[track_sort_idxs]

        # combine matched detections and tracks to form base object representations
        matched_detections, matched_tracks = detections[matched_det_idxs], self.tracks[matched_track_idxs]
        st_detections = BaseSTObjectRepresentations.join(matched_tracks.to_base_rep(), matched_detections)

        # map back to original detection order
        st_detections = detections.set(matched_det_idxs, st_detections)

        # get track ids
        instance_ids = - torch.ones(len(detections), dtype=torch.long)
        instance_ids[matched_det_idxs.long()] = self.track_ids[matched_track_idxs.long()]

        # get the ages of all ST detections
        ages = torch.ones(len(detections))
        ages[matched_det_idxs.long()] += self.track_ages[matched_track_idxs.long()]

        return st_detections, instance_ids, ages

    def separate_refinable_vs_other(self, st_detections, track_ids, st_ages):
        # identify which objects are refineable via confidence and having points
        mean_confidences = st_detections.get_object_confidences(which="mean")
        confidence_viability = mean_confidences >= self.confidence_refinement_thresh
        wpoints_viability = st_detections.get_withpoints_mask()
        age_viability = (st_ages >= self.min_refinement_age).to(wpoints_viability)
        refinement_viability = confidence_viability & wpoints_viability & age_viability

        # build new merged object representation
        refinable_objects = st_detections[refinement_viability]
        if self.refinement_strategy != "none" and self.refinement_components != "none":
            refinable_objects = refinable_objects.remove_empty_frames()
        other_objects = st_detections[~refinement_viability]
        st_detections = MergedEnhancedAndBaseObjectRepresentations(refinable_objects, other_objects, st_detections.device)

        # re-order track idxs accordingly
        track_ids = torch.cat([track_ids[refinement_viability], track_ids[~refinement_viability]])

        return st_detections, track_ids

    def update(self, detections, track_ids):
        # for first sample in a new scene
        num_detections = len(detections)
        if self.tracks is None or len(self.tracks) == 0:
            self.tracks = detections
            self.track_ids = torch.arange(self.global_track_counter, self.global_track_counter + num_detections)
            self.global_track_counter += num_detections
            self.track_ages = torch.ones(len(self.tracks))
            self.track_frames_since_hit = torch.zeros(len(self.tracks))
            return self.track_ids

        # first, update matched tracks
        matched_detections, matched_track_ids, = detections[track_ids >= 0], track_ids[track_ids >= 0]
        sorted_matched_track_idxs, sort_order = torch.where((self.track_ids.view(-1, 1) - matched_track_ids.view(1, -1)) == 0)
        matched_track_idxs = sorted_matched_track_idxs[torch.argsort(sort_order)]
        assert torch.all((matched_detections.frame2batchidx[1:] - matched_detections.frame2batchidx[:-1]) >= 0)

        # set new st detections in place of old tracks
        if len(matched_detections) > 0:
            updated_tracks = self.tracks.set(matched_track_idxs, matched_detections)  # reordering_map[orig_idxs] --> updated track idxs
        else:
            updated_tracks = self.tracks
        assert torch.all((updated_tracks.frame2batchidx[1:] - updated_tracks.frame2batchidx[:-1]) >= 0)
        if DEBUG:
            self.viz_associations_update(detections, track_ids, save_name="Update 1")

        # update all existing track ages
        self.track_ages += 1

        # second, remove all tracks that are too old
        unmatched_track_idxs = torch.ones(len(self.track_ids), dtype=torch.bool)
        unmatched_track_idxs[matched_track_idxs] = False
        self.track_frames_since_hit[unmatched_track_idxs] += 1
        self.track_frames_since_hit[matched_track_idxs] = 0
        updated_tracks = self._remove_old_tracks(updated_tracks)
        assert torch.all((updated_tracks.frame2batchidx[1:] - updated_tracks.frame2batchidx[:-1]) >= 0)

        # third, add in new tracks
        new_detections = detections[track_ids == -1]
        confidence_mask = new_detections.get_object_confidences(which="mean") >= self.track_init_threshold
        confident_detections = new_detections[confidence_mask]
        num_new = len(confident_detections)
        track_ids = track_ids.long()

        if num_new > 0:
            self.tracks = BaseSTObjectRepresentations.merge(updated_tracks, confident_detections)
            self.track_ids = torch.cat([self.track_ids, torch.arange(self.global_track_counter, self.global_track_counter+num_new)])
            self.track_frames_since_hit = torch.cat([self.track_frames_since_hit, torch.zeros(num_new)])
            self.track_ages = torch.cat([self.track_ages, torch.ones(num_new)])

            # update new detection track-id values
            new_track_ids = track_ids[track_ids == -1]
            new_track_ids[confidence_mask] = torch.arange(self.global_track_counter, self.global_track_counter+num_new).long()
            track_ids[track_ids == -1] = new_track_ids
            self.global_track_counter += num_new
        else:
            self.tracks = updated_tracks

        if DEBUG:
            self.viz_associations_update(detections, track_ids, save_name="Update 2")
        assert torch.all((self.tracks.frame2batchidx[1:] - self.tracks.frame2batchidx[:-1]) >= 0)

        return track_ids  # note that these are in a different order than that of our tracks!

    def _remove_old_tracks(self, tracks):
        keep_mask = self.track_frames_since_hit < self.max_without_hit
        tracks = tracks[keep_mask]
        self.track_ids = self.track_ids[keep_mask]
        self.track_ages = self.track_ages[keep_mask]
        self.track_frames_since_hit = self.track_frames_since_hit[keep_mask]
        return tracks

    def update_noobvservations(self):
        if self.track_frames_since_hit is None:
            return
        self.track_frames_since_hit += 1
        updated_tracks = self._remove_old_tracks(self.tracks)
        self.tracks = updated_tracks

    def viz_associations_pre(self, detections):
        """
        Operates entirely in numpy
        """

        # overarching color schemes
        track_color_pallette = np.array([[150, 170, 150]] * len(self.tracks))
        det_color_pallette = np.array([[0, 0, 0]] * len(detections))
        # track_box_color = np.array([[0, 0, 255]] * len(self.tracks))
        detection_box_color = np.array([[255, 0, 255]] * len(detections))


        clrs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]
        num_clrs = len(clrs)
        # color scheme for splines
        for track_idx, track_id in enumerate(self.track_ids):
            palette_idx = track_id.long().item() % num_clrs
            track_color_pallette[track_idx] = (np.array(list(clrs[palette_idx])) * 255).tolist()
            # palette_idx += 1
        track_box_color = track_color_pallette


        # generate color scheme for quadratic forescasts
        time_cmap = plt.get_cmap("coolwarm")
        track_spline_pallette = [time_cmap for i in range(len(self.tracks))]
        detection_center_pallette = [time_cmap for i in range(len(detections))]

        self._viz_helper(detections, track_color_pallette, det_color_pallette, track_box_color, detection_box_color, track_spline_pallette, detection_center_pallette, save_name="Pre-Association Scene")

    def viz_associations_post(self, detections, matches, sample_id):
        # overarching color schemes
        device = detections.device
        # clrs = seaborn.color_palette(palette='husl', n_colors=NUM_CLRS)
        clrs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
                         [1, 1, 1]]
        num_clrs = len(clrs)

        track_color_pallette = np.array([[0, 0, 0]] * len(self.tracks))
        det_color_pallette = np.array([[0, 0, 0]] * len(detections))
        track_box_color = np.array([[0, 0, 0]] * len(self.tracks))
        detection_box_color = np.array([[0, 0, 0]] * len(detections))


        # color scheme for splines
        track_spline_pallette = [ListedColormap([[0.65, 0.65, 0.65]]) for i in range(len(self.tracks))]
        detection_center_pallette = [ListedColormap([[0.0, 0.0, 0.0]]) for i in range(len(detections))]
        # palette_idx = 0
        for det_idx, track_idx in matches.items():
            palette_idx = self.track_ids[track_idx].long().item() % num_clrs
            cmap = ListedColormap([list(clrs[palette_idx])])
            track_spline_pallette[track_idx] = cmap
            detection_center_pallette[det_idx] = cmap
            track_color_pallette[track_idx] = (np.array(list(clrs[palette_idx])) * 255).tolist()
            det_color_pallette[det_idx] = (np.array(list(clrs[palette_idx])) * 255).tolist()  # todo: reset me
            # palette_idx += 1
        track_box_color = track_color_pallette
        detection_box_color = det_color_pallette

        self._viz_helper(detections, track_color_pallette, det_color_pallette, track_box_color, detection_box_color, track_spline_pallette, detection_center_pallette, save_name="Post-Association Scene", matches=matches, sample_id=sample_id)

    def viz_associations_update(self, detections, instance_ids, save_name="Update 0"):
        clrs = seaborn.color_palette(palette='husl', n_colors=NUM_CLRS)
        track_color_pallette = np.array([[150, 170, 150]] * len(self.tracks))
        det_color_pallette = np.array([[0, 0, 0]] * len(detections))
        track_box_color = np.array([[0, 0, 255]] * len(self.tracks))
        detection_box_color = np.array([[255, 0, 255]] * len(detections))

        # color scheme for splines
        detection_center_pallette = [ListedColormap([[0.0, 0.0, 0.0]]) for i in range(len(detections))]
        for det_idx, instance_id in enumerate(instance_ids):
            palette_idx = instance_id.long().item() % NUM_CLRS
            cmap = ListedColormap([list(clrs[palette_idx])])
            detection_center_pallette[det_idx] = cmap
            det_color_pallette[det_idx] = (np.array(list(clrs[palette_idx])) * 255).tolist()

        track_spline_pallette = [ListedColormap([[0.65, 0.65, 0.65]]) for i in range(len(self.tracks))]
        for track_idx in range(len(self.tracks)):
            palette_idx = self.track_ids[track_idx].long().item() % NUM_CLRS
            cmap = ListedColormap([list(clrs[palette_idx])])
            track_spline_pallette[track_idx] = cmap
            track_color_pallette[track_idx] = (np.array(list(clrs[palette_idx])) * 255).tolist()
            # palette_idx += 1

        self._viz_helper(detections, track_color_pallette, det_color_pallette, track_box_color, detection_box_color, track_spline_pallette, detection_center_pallette, save_name=save_name)


    def _viz_helper(self, detections, trk_pallet, det_pallet, trk_box_pallet, det_box_pallet, trk_spline_pallet, det_spline_pallet, save_name, matches=None, sample_id=''):
        # get times of visualization
        min_time = torch.min(self.tracks.get_object_timesteps("min")).unsqueeze(0)
        max_time = torch.max(detections.get_object_timesteps("max")).unsqueeze(0)
        matching_time = torch.max(self.tracks.get_object_timesteps("max")).unsqueeze(0)

        # get detection and track boxes at the timestep
        det_match_boxes, detection_parametrics = detections.estimate_boxes(matching_time, **self.trajectory_params)
        track_match_boxes, track_parametrics = self.tracks.estimate_boxes(matching_time, **self.trajectory_params)
        det_max_boxes, _ = detections.estimate_boxes(max_time, **self.trajectory_params)
        track_min_boxes, _ = self.tracks.estimate_boxes(min_time, **self.trajectory_params)
        det_match_boxes, track_match_boxes, det_max_boxes, track_min_boxes, detection_parametrics, track_parametrics = det_match_boxes[0].box, track_match_boxes[0].box, det_max_boxes[0].box, track_min_boxes[0].box, detection_parametrics[0], track_parametrics[0]
        det_match_boxes, track_match_boxes, det_max_boxes, track_min_boxes, detection_parametrics, track_parametrics = torch_to_numpy([det_match_boxes, track_match_boxes, det_max_boxes, track_min_boxes, detection_parametrics, track_parametrics])

        # get time range of parametrics
        track_avg_confidences = self.tracks.get_object_confidences(which="mean")
        det_avg_confidences = detections.get_object_confidences(which="mean")
        track_avg_confidences, det_avg_confidences = torch_to_numpy([track_avg_confidences, det_avg_confidences])

        # set up parametrics points
        trk_time_offset = (min_time - matching_time).item()
        trk_timespan, det_timespan = torch.abs(min_time - matching_time).item(), torch.abs(max_time - matching_time).item()
        num_trajectory_samples = 200
        trk_parametrics_times = np.arange(num_trajectory_samples) / num_trajectory_samples * trk_timespan + trk_time_offset
        trk_parametrics_times = np.stack([trk_parametrics_times**i for i in range(2)])
        trk_parametric_points = (track_parametrics @ trk_parametrics_times).transpose((0, 2, 1)).reshape(-1, 4)[:, :3]  # M x 200 x 4
        clr_times = np.arange(num_trajectory_samples) / num_trajectory_samples * trk_timespan / (trk_timespan + det_timespan)
        trk_parametric_clrs = np.array([trk_spline_pallet[track_idx](clr_times) for track_idx in range(len(self.tracks))]).reshape(-1, 4)[:, :3] * 255

        # set up detection parametrics
        det_parametrics_times = np.arange(num_trajectory_samples) / num_trajectory_samples * (max_time - matching_time).item()
        det_parametrics_times = np.stack([det_parametrics_times**i for i in range(2)])
        det_parametric_points = (detection_parametrics @ det_parametrics_times).transpose((0, 2, 1)).reshape(-1, 4)[:, :3]  # M x 200 x 4
        try:
            clr_times = np.arange(num_trajectory_samples) / num_trajectory_samples * det_timespan / (trk_timespan + det_timespan) + (trk_timespan / trk_timespan + det_timespan)
        except ZeroDivisionError:
            clr_times = np.zeros(num_trajectory_samples)
        det_parametric_clrs = np.array([det_spline_pallet[det_idx](clr_times) for det_idx in range(len(detections))]).reshape(-1, 4)[:, :3] * 255

        # set up points
        det_pnts = detections.points.to('cpu').data.numpy()[:, :3]
        det_pnt_clrs = det_pallet[detections.point2batchidx.to('cpu').data.numpy()]
        track_pnts = self.tracks.points.to('cpu').data.numpy()[:, :3]
        track_pnt_clrs = trk_pallet[self.tracks.point2batchidx.to('cpu').data.numpy()]

        # set up boxes
        track_names = np.round(track_avg_confidences, decimals=4).astype("str").tolist()
        # track_names = ["id=%s idx=%s" % (self.track_ids[i], i) for i in range(len(self.tracks))]
        # track_names = ["track-" + track_names[i] for i in range(len(track_names))]
        # det_names = np.round(det_avg_confidences, decimals=4).astype("str").tolist()
        # det_names = ["det-" + det_names[i] for i in range(len(det_names))]
        # det_names = ["idx=%s" % i for i in range(len(detections))]
        det_names = ["" for i in range(len(detections))]

        wandb_trk_match_boxes = get_wandb_boxes(track_match_boxes[:, :7], trk_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)), name=track_names)
        wandb_trk_min_boxes = get_wandb_boxes(track_min_boxes[:, :7], trk_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)), name=track_names)
        wandb_det_match_boxes = get_wandb_boxes(det_match_boxes[:, :7], det_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)), name=det_names)
        wandb_det_max_boxes = get_wandb_boxes(det_max_boxes[:, :7], det_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)), name=det_names)

        # get box directional points
        wandb_trk_match_boxes_points, wandb_trk_match_boxes_clrs = get_boxdir_points(track_match_boxes[:, :7], trk_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        wandb_trk_min_boxes_points, wandb_trk_min_boxes_clrs = get_boxdir_points(track_min_boxes[:, :7], trk_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        wandb_det_match_boxes_points, wandb_det_match_boxes_clrs = get_boxdir_points(det_match_boxes[:, :7], det_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)))
        wandb_det_max_boxes_points, wandb_det_max_boxes_clrs = get_boxdir_points(det_max_boxes[:, :7], det_box_pallet, scale_factor=1.0, viz_offset=np.zeros((1, 3)))

        # put everything together and log
        viz_points = np.vstack([trk_parametric_points, det_parametric_points, track_pnts, det_pnts, wandb_trk_match_boxes_points, wandb_trk_min_boxes_points, wandb_det_match_boxes_points, wandb_det_max_boxes_points])
        viz_clrs = np.vstack([trk_parametric_clrs, det_parametric_clrs, track_pnt_clrs, det_pnt_clrs, wandb_trk_match_boxes_clrs, wandb_trk_min_boxes_clrs, wandb_det_match_boxes_clrs, wandb_det_max_boxes_clrs])
        wandb_boxes = np.array(wandb_trk_match_boxes.tolist() + wandb_trk_min_boxes.tolist() + wandb_det_match_boxes.tolist() + wandb_det_max_boxes.tolist())
        wandb_pc = np.hstack([viz_points, viz_clrs])

        wandb.log({save_name: wandb.Object3D({"type": "lidar/beta", "points": wandb_pc, "boxes": wandb_boxes})})

        # save file locally
        wandb_current_step = wandb.run._step - 1
        local_savename = os.path.join(self.media_outdir, save_name + "_wandbstep_%s_%s.npz" % (wandb_current_step, sample_id))

        # track point info
        track_points = self.tracks.points.to('cpu').data.numpy()[:, :4]
        track_point2frame = self.tracks.point2frameidx.to('cpu').data.numpy()
        track_frame2batch = self.tracks.frame2batchidx.to('cpu').data.numpy()

        # detection point info
        det_pnts = detections.points.to('cpu').data.numpy()[:, :3]
        det_point2frame = detections.point2frameidx.to('cpu').data.numpy()
        det_frame2batch = self.tracks.frame2batchidx.to('cpu').data.numpy()

        # get ego-origin of current detections
        object_timesteps = self.tracks.get_object_timesteps(which="max")
        current_time_idx = torch.argmax(object_timesteps)
        ego_loc = torch.inverse(self.tracks.global2currentego[current_time_idx])[:3, 3] # size (3,) tensor
        ego_loc = ego_loc.to('cpu').data.numpy()

        # get matches
        matches = matches

        # boxes info
        track_boxes = self.tracks.boxes.box.to('cpu').data.numpy()  # center-lwh-heading-vel-acc (
        track_confidences = self.tracks.confidences.to('cpu').data.numpy()
        det_boxes = detections.boxes.box.to('cpu').data.numpy()
        det_confidences = detections.confidences.to('cpu').data.numpy()

        np.savez(local_savename, track_points=track_points, track_point2frame=track_point2frame, track_frame2sequence=track_frame2batch,
                 detection_points=det_pnts, detection_point2frame=det_point2frame, detection_frame2sequence=det_frame2batch,
                 ego_position=ego_loc, assignments=matches, track_bboxes=track_boxes, detection_bboxes=det_boxes,
                 track_confidences=track_confidences, detection_confidences=det_confidences)



