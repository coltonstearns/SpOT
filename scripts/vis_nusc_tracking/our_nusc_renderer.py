
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

    def render(self, timestamp: int, frame_gt: List[TrackingBox], frame_pred: List[TrackingBox]) \
            -> None:
        """
        Render function for a given scene timestamp
        :param timestamp: timestamp for the rendering
        :param frame_gt: list of ground truth boxes
        :param frame_pred: list of prediction boxes
        """
        # Init.
        print('Rendering {}'.format(timestamp))
        # switches = events[events.Type == 'SWITCH']
        # switch_ids = switches.HId.values
        switch_ids = []
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
            if b.tracking_id in switch_ids:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=('r', 'r', color), linewidth=0.8)
            else:
                color = self.id2color[b.tracking_id]
                box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=0.8)

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}.png'.format(timestamp)), dpi=1000)
        plt.close(fig)

        # Plot GT boxes.
        print(">>")
        print(len(frame_gt))
        fig, ax = plt.subplots()

        for b in frame_gt:
            color = 'k'
            box = Box(b.ego_translation, b.size, Quaternion(b.rotation), name=b.tracking_name, token=b.tracking_id)
            color = (float(hash(b.tracking_id + 'r') % 256) / 255,
                                            float(hash(b.tracking_id + 'g') % 256) / 255,
                                            float(hash(b.tracking_id + 'b') % 256) / 255)
            box.render(ax, view=np.eye(4), colors=(color, color, color), linewidth=0.8)

        # Plot ego pose.
        plt.scatter(0, 0, s=96, facecolors='none', edgecolors='k', marker='o')
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

        # Save to disk and close figure.
        fig.savefig(os.path.join(self.save_path, '{}_gt.png'.format(timestamp)), dpi=1000)
        # plt.clf()
        plt.close(fig)