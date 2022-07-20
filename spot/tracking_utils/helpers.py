# We implemented our method on top of AB3DMOT's KITTI tracking open-source code

import numpy as np

def greedy_match(distance_matrix):
  '''
  Find the one-to-one matching using greedy allgorithm choosing small distance
  distance_matrix: (num_detections, num_tracks)
  '''
  matched_indices = []

  num_detections, num_tracks = distance_matrix.shape
  distance_1d = distance_matrix.reshape(-1)
  index_1d = np.argsort(distance_1d)
  index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
  detection_id_matches_to_tracking_id = [-1] * num_detections
  tracking_id_matches_to_detection_id = [-1] * num_tracks
  for sort_i in range(index_2d.shape[0]):
    detection_id = int(index_2d[sort_i][0])
    tracking_id = int(index_2d[sort_i][1])
    if np.isfinite(distance_matrix[detection_id, tracking_id]) and tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
      tracking_id_matches_to_detection_id[tracking_id] = detection_id
      detection_id_matches_to_tracking_id[detection_id] = tracking_id
      matched_indices.append([detection_id, tracking_id])

  matched_indices = np.array(matched_indices)
  return matched_indices


def greedy_match_confidences_detections(distance_matrix):
    dist = np.copy(distance_matrix)
    matched_indices = []
    for detection_idx in range(dist.shape[0]):
        track_idx = dist[detection_idx].argmin()
        if np.isfinite(dist[detection_idx, track_idx]):
            dist[:, track_idx] = np.inf
            matched_indices.append([detection_idx, track_idx])
    return np.array(matched_indices).reshape(-1, 2)


def greedy_match_confidences_tracks(distance_matrix):
    dist = np.copy(distance_matrix)
    matched_indices = []
    for track_idx in range(dist.shape[1]):
        detection_idx = dist[:, track_idx].argmin()
        if np.isfinite(dist[detection_idx, track_idx]):
            dist[detection_idx, :] = np.inf
            matched_indices.append([detection_idx, track_idx])
    return np.array(matched_indices).reshape(-1, 2)
