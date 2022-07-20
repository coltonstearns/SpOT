
import numpy as np

class Sequences:

    def __init__(self, frames, keyframe_ids, timesteps, object_class, sequence_params, tracking_baseline):
        self.frames = frames
        self.keyframe_ids = keyframe_ids
        self.timesteps = timesteps
        self.object_class = object_class
        self.tracking_baseline = tracking_baseline

        # parameters for sequence generation
        self.dataset_source = sequence_params['dataset-properties']['dataset-source']
        self.tracking_mode = sequence_params['dataset-properties']['tracking-mode']
        if self.tracking_mode:
            self.seq_len = 12 if self.dataset_source == "nuscenes" else 1
            self.seqs_span_many_samples = False
            self.allow_false_detections = True
            self.min_keyframes = 1
        else:
            self.seq_len = sequence_params['sequence-properties']['sequence-length']
            self.seqs_span_many_samples = sequence_params['sequence-properties']['seqs-span-many-keyframes']
            self.allow_false_detections = sequence_params['sequence-properties']['allow-false-positive-frames']
            self.min_keyframes = sequence_params['sequence-properties']['min-keyframes']

        # sequence filtering parameters
        self.permitted_frame_skips = sequence_params['sequence-filters']['permitted-frame-skips']
        self.min_frame_pts = sequence_params['sequence-filters']['min-frame-pts']
        self.min_labeled_pts = sequence_params['sequence-filters']['min-labeled-pts']
        self.min_seq_pts = sequence_params['sequence-filters']['min-seq-pts']
        self.only_moving_objects = sequence_params['sequence-filters']['only-moving-objects']
        self.conf_thresh = sequence_params['sequence-filters']['confidence-threshold']
        self.overlap_threshold = sequence_params['sequence-filters']['overlap-threshold']

    def prepare_sequences(self, gt_statuses):
        # get list of all unique observation IDs (observations are unique object-track tokens)
        track_tokens, tracktoken2idx, idx2tracktoken = self._get_observation_ids(gt_statuses)

        # get tables indicating if sliding-window track viability
        numpoints_table, kf_viability, ismoving_viability, confidence_viability, observations_table = self._sliding_window_viabilities(track_tokens, tracktoken2idx)

        # get final validity table
        validity_table, allvalid_table, kf_exist_table = self._compute_track_validity_table(track_tokens, observations_table, numpoints_table, kf_viability, ismoving_viability, confidence_viability)

        # generate a list of lists; outer list is for each instance observation; inner list is the sweep start-idx of the sequence
        sequence_idxs, filtered_sequence_idxs = self._get_sequence_idxs(track_tokens, validity_table, allvalid_table)
        sequences, seq2sample = self._generate_sequences(sequence_idxs, idx2tracktoken, False, observations_table, numpoints_table, kf_viability, kf_exist_table, allvalid_table)
        filtered_sequences, filtered_seq2sample = self._generate_sequences(filtered_sequence_idxs, idx2tracktoken, True, observations_table, numpoints_table, kf_viability, kf_exist_table, allvalid_table)

        return sequences, seq2sample, filtered_sequences, filtered_seq2sample

    def _get_observation_ids(self, gt_statuses):
        # get all object-type instance ids to start with
        scene_observations = set()
        for i, frame in enumerate(self.frames):
            for inst_id in set(frame.keys()):
                if frame[inst_id].object_class != self.object_class:
                    continue
                if frame[inst_id].gt_status not in gt_statuses:
                    continue

                scene_observations.add(self.get_observation_id(self.keyframe_ids[i], inst_id))

        scene_observations = list(scene_observations)
        obsid2idx = {obs_id: i for i, obs_id in enumerate(scene_observations)}
        idx2obsid = {i: obs_id for i, obs_id in enumerate(scene_observations)}
        return scene_observations, obsid2idx, idx2obsid

    def _sliding_window_viabilities(self, track_tokens, tracktoken2idx):
        M, T = len(track_tokens), len(self.timesteps)
        observations_table = np.zeros((T, M), dtype=np.bool)
        numpoints_table = np.zeros((T, M), dtype=np.uint16)
        kf_viability = np.zeros((T, M), dtype=np.bool)
        ismoving_viability = np.zeros((T, M), dtype=np.bool)
        confidence_viability = np.zeros((T, M), dtype=np.bool)
        for time_idx, frame in enumerate(self.frames):
            for inst_id in set(frame.keys()):
                track_token = self.get_observation_id(self.keyframe_ids[time_idx], inst_id)
                if track_token not in track_tokens:
                    continue

                # check viabilities
                num_pts = frame[inst_id].num_pts
                is_moving = frame[inst_id].gt_is_moving
                iskf = frame[inst_id].is_keyframe
                is_confident = frame[inst_id].confidence >= self.conf_thresh

                # update tables
                track_idx = tracktoken2idx[track_token]
                numpoints_table[time_idx, track_idx] = num_pts
                kf_viability[time_idx, track_idx] = iskf
                ismoving_viability[time_idx, track_idx] = is_moving
                confidence_viability[time_idx, track_idx] = is_confident
                observations_table[time_idx, track_idx] = True

        if not self.only_moving_objects:
            ismoving_viability = np.ones((T, M), dtype=np.bool)

        return numpoints_table, kf_viability, ismoving_viability, confidence_viability, observations_table

    def _compute_track_validity_table(self, track_tokens, obvservations_table, numpoints_table, kf_table, ismoving_table, confidence_table):
        M, T = len(track_tokens), len(self.timesteps)
        validity = np.zeros((T, M), dtype=np.bool)
        kf_existance_table = np.zeros((T, M), dtype=np.bool)  # because we need KF thresholds regardless of seq validity
        hasobs_table = np.zeros((T, M), dtype=np.bool)

        for i in range(T):
            start_idx, end_idx = max(0, i - self.seq_len + 1), i + 1
            if not self.seqs_span_many_samples:
                cur_sampleid = self.keyframe_ids[i]
                sampleid_invalid_offset = np.sum(np.array(self.keyframe_ids)[start_idx:end_idx] != cur_sampleid)
                start_idx += sampleid_invalid_offset

            # only use confident frames
            is_confident = confidence_table[start_idx:end_idx]

            # frame point constraints
            frame_enough_pts = (numpoints_table[start_idx:end_idx] > self.min_frame_pts) * is_confident
            num_points = np.sum(numpoints_table[start_idx:end_idx] * frame_enough_pts, axis=0)
            frame_skips = self.seq_len - np.sum(frame_enough_pts, axis=0)

            # keyframe observation validity
            num_kf_points = np.sum(numpoints_table[start_idx:end_idx] * kf_table[start_idx:end_idx] * is_confident, axis=0)
            num_kfs = np.sum(kf_table[start_idx:end_idx] * is_confident, axis=0)
            num_kfs_allconf = np.sum(kf_table[start_idx:end_idx], axis=0)
            # other validity checks
            ismoving = np.any(ismoving_table[start_idx:end_idx], axis=0)
            hasobservation = np.any(obvservations_table[start_idx:end_idx], axis=0)

            # final validity check
            point_validity = (num_points >= self.min_seq_pts) & (frame_skips <= self.permitted_frame_skips)
            kf_validity = (num_kfs >= self.min_keyframes) & (num_kf_points >= self.min_labeled_pts)
            kf_existance = num_kfs_allconf >= self.min_keyframes
            other_validity = ismoving
            valid = point_validity & kf_validity & other_validity
            validity[i, :] = valid
            kf_existance_table[i, :] = kf_existance
            hasobs_table[i, :] = hasobservation

        # build table of filtered-out sequences
        complete_table = kf_existance_table & obvservations_table

        return validity, complete_table, kf_existance_table

    def _get_sequence_idxs(self, track_tokens, validity_table, allvalid_table):
        M, T = len(track_tokens), len(self.timesteps)
        sequence_idxs, filtered_sequence_idxs = [], []
        for j in range(M):
            # get object validity mask
            object_valid_idxs = np.where(validity_table[:, j] > 0)[0]

            # for multiple valid sequences
            disjoint_valid_idxs = [object_valid_idxs[0]] if len(object_valid_idxs) > 0 else []
            for i in range(1, len(object_valid_idxs)):
                if (object_valid_idxs[i] - disjoint_valid_idxs[-1]) >= self.overlap_threshold:
                    disjoint_valid_idxs.append(object_valid_idxs[i])

            # get all-possible-sequences; record those that were filtered-out due to insufficient points/frames/etc
            object_all_idxs = np.where(allvalid_table[:, j] > 0)[0]
            if len(object_all_idxs) == 0 and self.min_keyframes < 2:
                print("No valid sequences for an object. This should not happen!")

            disjoint_filtered_idxs, existing_idxs = [], np.array([np.inf] + disjoint_valid_idxs)
            for i in range(len(object_all_idxs)):
                is_disjoint = np.min(np.abs(existing_idxs - object_all_idxs[i])) >= self.overlap_threshold
                if is_disjoint:
                    disjoint_filtered_idxs.append(object_all_idxs[i])
                    existing_idxs = np.concatenate([existing_idxs, np.array([object_all_idxs[i]])])

            # only take object sequences with enough points
            sequence_idxs.append(disjoint_valid_idxs)
            filtered_sequence_idxs.append(disjoint_filtered_idxs)

        return sequence_idxs, filtered_sequence_idxs


    def _generate_sequences(self, sequence_idxs, idx2tracktoken, empty, observations_table, numpoints_table, kf_viability, kf_existance, all_valid):
        M, T = len(sequence_idxs), len(self.timesteps)

        sequences = []
        sequence_sample_assignments = []
        for j in range(M):
            if len(sequence_idxs[j]) == 0:
                continue

            track_token = idx2tracktoken[j]
            inst_id = self.observationid2instanceid(track_token)
            for frame_idx in sequence_idxs[j]:
                seq = self._generate_sequence(inst_id, frame_idx, empty=empty)
                sequences.append(seq)
                sequence_sample_assignments.append(self.keyframe_ids[frame_idx])
        return sequences, sequence_sample_assignments


    def _generate_sequence(self, inst_id, frame_idx, empty=False):
        seq = []
        keyframe_id = self.keyframe_ids[frame_idx]
        for k in range(self.seq_len):
            if frame_idx - k >= 0:
                frame = self.frames[frame_idx - k]
                sweep_sample_id = self.keyframe_ids[frame_idx - k]
                in_kf_sample = True if self.seqs_span_many_samples else sweep_sample_id == keyframe_id
                if inst_id in frame and in_kf_sample:
                    if frame[inst_id].num_pts >= self.min_frame_pts or empty:
                        seq.append(frame[inst_id])
                    else:
                        seq.append(None)
                else:
                    seq.append(None)
            else:
                seq.append(None)
        return seq


    def get_observation_id(self, sample_id, instance_id):
        if not self.seqs_span_many_samples:
            observation_id = "_".join([sample_id, instance_id])
        else:
            observation_id = instance_id
        return observation_id

    def observationid2instanceid(self, observation_id):
        return observation_id.split("_")[-1]