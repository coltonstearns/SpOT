import os
import json
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.tracking.data_classes import TrackingConfig, TrackingBox, TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes import NuScenes



import os
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pyquaternion import Quaternion

from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.tracking.constants import TRACKING_COLORS, PRETTY_TRACKING_NAMES
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingMetricDataList
from nuscenes.utils.data_classes import Box

Axis = Any


class TrackingRenderer:
    """
    Class that renders the tracking results in BEV and saves them to a folder.
    """
    def __init__(self, save_path):
        """
        :param save_path:  Output path to save the renderings.
        """
        self.save_path = save_path
        self.id2color = {}  # The color of each track.

    def render(self, timestamp: int, frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param timestamp: timestamp for the rendering
        :param frame_pred: list of prediction boxes
        """
        # Init.
        print('Rendering {}'.format(timestamp))
        fig, ax = plt.subplots()

        # Plot predicted boxes.
        for b in frame_pred:
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)

            # Determine color for this tracking id.
            if b.tracking_id not in self.id2color.keys():
                self.id2color[b.tracking_id] = (float(hash(b.tracking_id + 'r') % 256) / 255,
                                                float(hash(b.tracking_id + 'g') % 256) / 255,
                                                float(hash(b.tracking_id + 'b') % 256) / 255)

            # Render box. Highlight identity switches in red.
            color = self.id2color[b.tracking_id]
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=0.8)

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)), dpi=1000)
        plt.close(fig)








file = "/home/colton/Documents/SpOT-material/nuscenes-test-results/results.json"
nusc_dataroot = "/media/colton/ColtonSSD/nuscenes-raw"
save_path = "/home/colton/Documents/SpOT-material/vis_nuscenes_test"
if not os.path.exists(save_path):
    os.makedirs(save_path)

pred_boxes, meta = load_prediction(file, 10000, TrackingBox, verbose=True)
nusc = NuScenes(version="v1.0-test", verbose=True, dataroot=nusc_dataroot)
pred_boxes = add_center_dist(nusc, pred_boxes)

pred_tracks = create_tracks(pred_boxes, nusc, eval_split="test", gt=False)
print("-----")
print(meta)

# visualize boxes
renderer = TrackingRenderer(save_path)
for i, scene_token in enumerate(pred_tracks.keys()):
    scene_tracks = pred_tracks[scene_token]  # this is a default_dict of lists
    for timestep, frame_boxes in scene_tracks.items():
        renderer.render(timestep, frame_boxes)





