from ..frame_data import FrameData
from ..update_info_data import UpdateInfoData
from ..data_protos import BBox, Validity
import numpy as np, mot_3d.utils as utils
from ..tracklet import Tracklet
from ..association import associate_dets_to_tracks


class RedundancyModule:
    def __init__(self, configs):
        self.configs = configs
        self.mode = configs['redundancy']['mode']
        self.asso = configs['running']['asso']
        self.det_score = configs['redundancy']['det_score_threshold'][self.asso]
        self.det_threshold = configs['redundancy']['det_dist_threshold'][self.asso]
        self.motion_model_type = configs['running']['motion_model']
    
    @property
    def back_step(self):
        return self.motion_model_type == 'velo'
    
    def infer(self, trk: Tracklet, input_data: FrameData, time_lag=None):
        if self.mode == 'bbox':
            return self.bbox_redundancy(trk, input_data)
        elif self.mode == 'mm':
            return self.motion_model_redundancy(trk, input_data, time_lag)
        else:
            return self.default_redundancy(trk, input_data)
    
    def default_redundancy(self, trk: Tracklet, input_data: FrameData):
        pred_bbox = trk.get_state()
        return pred_bbox, 0, None
    
    def motion_model_redundancy(self, trk: Tracklet, input_data: FrameData, time_lag):
        # get the motion model prediction / current state
        pred_bbox = trk.get_state()

        # associate to low-score detections
        dists = list()
        dets = input_data.dets
        related_indexes = [i for i, det in enumerate(dets) if det.s > self.det_score]
        candidate_dets = [dets[i] for i in related_indexes]

        # if back step the dets for association
        if self.back_step:
            velos = input_data.aux_info['velos']
            candidate_velos = [velos[i] for i in related_indexes]

        # association
        for i, det in enumerate(candidate_dets):
            # if use velocity model, back step the detection
            if self.back_step:
                velo = candidate_velos[i]
                pd_det = utils.back_step_det(det, velo, time_lag)
            else:
                pd_det = det

            if self.asso == 'iou':
                dists.append(utils.iou3d(pd_det, pred_bbox)[1])
            elif self.asso == 'giou':
                dists.append(utils.giou3d(pd_det, pred_bbox))
            elif self.asso == 'm_dis':
                trk_innovation_matrix = trk.compute_innovation_matrix()
                inv_innovation_matrix = np.linalg.inv(trk_innovation_matrix)
                dists.append(utils.m_distance(pd_det, pred_bbox, inv_innovation_matrix))
            elif self.asso == 'euler':
                dists.append(utils.m_distance(pd_det, pred_bbox))

        if self.asso in ['iou', 'giou'] and (len(dists) == 0 or np.max(dists) < self.det_threshold):
            result_bbox = pred_bbox
            update_mode = 0
        elif self.asso in ['m_dis', 'euler'] and (len(dists) == 0 or np.min(dists) > self.det_threshold):
            result_bbox = pred_bbox
            update_mode = 0
        else:
            result_bbox = pred_bbox
            update_mode = 3
        return result_bbox, update_mode, {'velo': np.zeros(2)} 
    
    def bbox_redundancy(self, trk: Tracklet, input_data: FrameData):
        candidate_dets = [det for det in input_data.dets if det.s > self.det_score]
        pred_bbox = BBox.array2bbox(trk.motion_model.history[-1].reshape(-1))
    
        ious = list()
        for det in candidate_dets:
            ious.append(utils.iou3d(det, pred_bbox)[1])
        
        if len(ious) == 0 or np.max(ious) < self.det_threshold:
            result_bbox = pred_bbox
            update_mode = 0
        else:
            max_index = np.argmax(ious)
            result_bbox = candidate_dets[max_index]
            if ious[max_index] > 0.7:
                update_mode = 1
            else:
                update_mode = 3
        return result_bbox, update_mode
    
    def bipartite_infer(self, input_data: FrameData, tracklets):
        dets = input_data.dets
        det_indexes = [i for i, det in enumerate(dets) if det.s >= self.det_score]
        dets = [dets[i] for i in det_indexes]

        # prediction and association
        trk_preds = list()
        for trk in tracklets:
            trk_preds.append(trk.predict(input_data.time_stamp, input_data.aux_info['is_key_frame']))
        
        matched, unmatched_dets, unmatched_trks = associate_dets_to_tracks(dets, trk_preds, 
            'bipartite', 'giou', 1 - self.det_threshold, None)
        for k in range(len(matched)):
            matched[k][0] = det_indexes[matched[k][0]]
        for k in range(len(unmatched_dets)):
            unmatched_dets[k] = det_indexes[unmatched_dets[k]]

        result_bboxes, update_modes = [], []
        for t, trk in enumerate(tracklets):
            if t not in unmatched_trks:
                for k in range(len(matched)):
                    if matched[k][1] == t:
                        d = matched[k][0]
                        break
                result_bboxes.append(trk_preds[t])
                update_modes.append(4)
            else:
                result_bboxes.append(trk_preds[t])
                update_modes.append(0)
        return result_bboxes, update_modes
