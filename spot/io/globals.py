import numpy as np
from waymo_open_dataset import label_pb2

# ================================================

# Size binning parameters
NUSCENES_BINNING_CLUSTER_SIZES = [[[4.8, 1.8, 1.5]],  # car
                 [[2.4560939, 6.73778078, 2.73004906]],  # truck
                 [[2.87427237, 12.01320693, 3.81509561]],  # trailer
                 [[0.60058911, 1.68452161, 1.27192197]],  # bicycle
                 [[0.66344886, 0.7256437, 1.75748069]],  # pedestrian
                 [[0.39694519, 0.40359262, 1.06232151]],  # traffic_cone
                 [[2.49008838, 0.48578221, 0.98297065]],
                 [[0.7058911, 1.7452161, 1.5192197]]]  # barrier

WAYMO_BINNING_CLUSTER_SIZES = [[[4.8, 1.8, 1.5]], # unknown
                               [[4.8, 1.8, 1.5], [10.0, 2.6, 3.2], [2.0, 1.0, 1.6]], # vehicle
                               [[0.9, 0.9, 1.7]],  # pedestrian
                               [[0.7, 0.7, 0.7]],  # sign
                               [[0.6, 1.7, 1.3]]]  # bicycle

TRANSFORMER_POSITIONAL_ENCODING_SCALES = {'car': 20, # nuscenes
                                          'pedestrian': 8, # nuscenes+waymo
                                          'bicycle': 10,  # nuscenes
                                          'motorcycle': 20,
                                          'cyclist': 10,  # waymo
                                          'truck': 25, # nuscenes
                                          'trailer': 25, # nuscenes
                                          'bus': 40,
                                          'vehicle': 20}  # waymo

# ================================================
# NuScenes class info

NUSCENES_CLASSES = {'car': 0,
                    'truck': 1,
                    'trailer': 2,
                    'bicycle': 3,
                    'pedestrian': 4,
                    'traffic_cone': 5,
                    'barrier': 6,
                    'motorcycle': 7,
                    'bus': 8,
                    'construction_vehicle': 9,}

WAYMO_CLASSES = { 'unknown': 0,
                  'vehicle': 1,
                  'pedestrian': 2,
                  'sign': 3,
                  'cyclist': 4}

WAYMO_CLASSPROTO2NAME = {
    label_pb2.Label.TYPE_VEHICLE: 'vehicle',
    label_pb2.Label.TYPE_PEDESTRIAN: 'pedestrian',
    label_pb2.Label.TYPE_SIGN: 'sign',
    label_pb2.Label.TYPE_CYCLIST: 'cyclist',
}

WAYMO_NAME2CLASSPROTO = {
    'vehicle': label_pb2.Label.TYPE_VEHICLE,
    'pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
    'sign': label_pb2.Label.TYPE_SIGN,
    'cyclist': label_pb2.Label.TYPE_CYCLIST,
}

WAYMO_IDX2CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

NUSCENES_IDX2CLASSES = ['car', 'truck', 'trailer', 'bicycle', 'pedestrian', 'traffic_cone', 'barrier', 'motorcycle', 'bus', 'construction_vehicle']

DEFAULT_ATTRIBUTE = {
    'car': 'vehicle.parked',
    'pedestrian': 'pedestrian.moving',
    'trailer': 'vehicle.parked',
    'truck': 'vehicle.parked',
    'bus': 'vehicle.moving',
    'motorcycle': 'cycle.without_rider',
    'construction_vehicle': 'vehicle.parked',
    'bicycle': 'cycle.without_rider',
    'barrier': '',
    'traffic_cone': '',
}

