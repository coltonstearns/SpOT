import torch
from pytorch3d.ops import box3d_overlap
import numpy as np
from pyquaternion import Quaternion
import copy
from preprocessing.nuscenes.utils import quaternion_yaw


class Box3D:

    def __init__(self,
                 center,
                 orientation,
                 wlh,
                 velocity,
                 score,
                 class_name,
                 attribute,
                 instance_id,
                 timestep,
                 gt_instance_id=None,
                 with_gt_box=False,
                 gt_center=None,
                 gt_orientation=None,
                 gt_wlh=None,
                 gt_velocity=None,
                 gt_acceleration=None,
                 gt_is_moving=None
                 ):
        """
        Note: in NuScenes, we can a value for <gt_instance_id> while not having any other GT parameters, due to
        only observing GT box properties at keyframes.
        """
        # predicted characteristics from pretrained 3D detector
        self.center = center  # size (3,1)
        self.orientation = orientation  # size (1,1) or Quaternion
        self.wlh = wlh  # size (3,1)
        self.velocity = velocity  # size (3,1)
        self.score = score
        self.class_name = class_name
        self.attribute = attribute
        self.instance_id = instance_id
        self.timestep = timestep

        # assigned GT characteristics (if applicable)
        self.gt_instance_id = gt_instance_id
        self.with_gt_box = with_gt_box
        self.gt_center = gt_center  # size (3,1)
        self.gt_orientation = gt_orientation  # size (1,1) or Quaternion
        self.gt_wlh = gt_wlh  # size (3,1)
        self.gt_velocity = gt_velocity  # size (3,1)
        self.gt_acceleration = gt_acceleration  # size (3,1)
        self.gt_is_moving = gt_is_moving

    def update_gt(self, box):
        self.with_gt_box = True
        self.gt_instance_id = box.gt_instance_id
        self.gt_center = box.gt_center
        self.gt_orientation = box.gt_orientation
        self.gt_wlh = box.gt_wlh
        self.gt_velocity = box.gt_velocity
        self.gt_acceleration = box.gt_acceleration
        self.gt_is_moving = box.gt_is_moving
        if self.attribute is None and box.attribute is not None:
            self.attribute = box.attribute

    def remove_gt_box(self):
        self.with_gt_box = False
        self.gt_center = None
        self.gt_orientation = None
        self.gt_wlh = None
        self.gt_velocity = None
        self.gt_acceleration = None

    def get_gt_status(self):
        if self.with_gt_box:
            gt_status = "with-gt"
        elif not self.with_gt_box and self.gt_instance_id is not None:
            gt_status = "in-gt-sequence"
        else:
            gt_status = "false-positive"
        return gt_status

    def to_array(self):
        if isinstance(self.orientation, Quaternion):
            yaw = np.array([[quaternion_yaw(self.orientation)]])
        else:
            yaw = self.orientation

        box_array = np.concatenate([self.center, self.wlh, yaw, self.velocity, np.zeros((3, 1))], axis=0)
        return box_array.flatten()

    def gt_to_array(self):
        if isinstance(self.orientation, Quaternion):
            yaw = np.array([[quaternion_yaw(self.gt_orientation)]])
        else:
            yaw = self.gt_orientation
        box_array = np.concatenate([self.gt_center, self.gt_wlh, yaw, self.gt_velocity, self.gt_acceleration], axis=0)
        return box_array.flatten()

    def prediction_corners(self):
        """
        Computes box corners of predictions
        Returns: 8x3 tensor of box corners

        """
        return self._corners(self.center, self.orientation, self.wlh)

    def gt_corners(self):
        return self._corners(self.gt_center, self.gt_orientation, self.gt_wlh)

    def _corners(self, center, orientation, wlh):
        corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
        corners_norm = corners_norm[[0, 4, 6, 2, 1, 5, 7, 3]]
        corners_norm = corners_norm - np.array([[0.5, 0.5, 0.5]])

        # compute origin-aligned corners
        corners = wlh.reshape((1, 3)) * corners_norm.reshape((8, 3))
        corners = corners.transpose()

        # rotate corners
        if isinstance(orientation, Quaternion):
            rot_mat = orientation.rotation_matrix
        else:
            rot_mat = np.array([[np.cos(orientation.item()), np.sin(orientation.item()), 0],
                                [-np.sin(orientation.item()), np.cos(orientation.item()), 0],
                                [0, 0, 1]])
        corners = np.dot(rot_mat, corners)

        # translate corners
        corners += center.reshape(3, 1)
        corners = corners.transpose()
        return corners

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x
        if self.with_gt_box:
            self.gt_center += x

    def rotate(self, rotation) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """

        if isinstance(rotation, Quaternion):
            rot_mat = rotation.rotation_matrix
            self.orientation = rotation * self.orientation
            if self.with_gt_box:
                self.gt_orientation = rotation * self.gt_orientation

        else:
            if isinstance(rotation, np.ndarray):
                rot_mat = rotation
                v = np.dot(rotation, np.array([1, 0, 0]))
                rotation = np.arctan2(v[1], v[0])

            else:  # assume is a float indicating yaw
                rot_mat = np.array([[np.cos(rotation), np.sin(rotation), 0],
                                    [-np.sin(rotation), np.cos(rotation), 0],
                                    [0, 0, 1]])
            self.orientation = (rotation + self.orientation) % (2*np.pi)
            self.orientation -= 2*np.pi if np.any(self.orientation > np.pi) else self.orientation
            if self.with_gt_box:
                self.gt_orientation = (rotation + self.gt_orientation) % (2*np.pi)
                self.gt_orientation -= 2 * np.pi if np.any(self.gt_orientation > np.pi) else self.gt_orientation

        # apply to center and velocity
        self.center = np.dot(rot_mat, self.center)
        self.velocity = np.dot(rot_mat, self.velocity)

        # apply to gt-center, gt-velocity, and gt-acceleration if applicable
        if self.with_gt_box:
            self.gt_center = np.dot(rot_mat, self.gt_center)
            self.gt_velocity = np.dot(rot_mat, self.gt_velocity)
            self.gt_acceleration = np.dot(rot_mat, self.gt_acceleration)

    def copy(self):
        # predicted characteristics from pretrained 3D detector
        return copy.deepcopy(self)


def associate(gt_boxes, pred_boxes, threshold, distance_type="l2"):
    """

    Args:
        gt_boxes (list<Box3D>):
        pred_boxes (list<Box3D>):
        threshold: we do not consider a match (1) above this threshold for l2 (2) below this threshold for IOU
        distance_type: one of ["l2", "3D-IOU"]

    Returns:
        (list<Box3D>): updated list of boxes, where information of assigned boxes is merged
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return pred_boxes

    # Sort predictions by confidence
    scores = [box.score for box in pred_boxes]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(scores))][::-1]

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    updated_boxes = []
    dist_fn = iou_3d if distance_type=="3D-IOU" else l2
    dists = dist_fn(gt_boxes, pred_boxes)

    # we always compare less than for thresholding
    if distance_type == "3D-IOU":
        threshold *= -1
        dists *= -1

    # run through and assign predicted boxes to GT labels
    for pred_idx, box in enumerate(pred_boxes):
        valid_gts = np.array([gt_box.class_name == box.class_name for gt_box in gt_boxes])
        this_dists = dists[:, pred_idx].copy()
        this_dists[~valid_gts] = np.inf
        gt_idx = np.argmin(this_dists)
        valid_match = dists[gt_idx, pred_idx] <= threshold
        if valid_match:
            box.update_gt(gt_boxes[gt_idx])
        updated_boxes.append(box)

    return updated_boxes


def iou_3d(gts, preds):
    """
    Computes a matrix of 3D IOUs between two sets of bounding boxes.
    Args:
        gt (list<Box3D>): first set of boxes
        preds (list<Box3D>): second set of boxes

    Returns:
        ious (np.array)
    """
    # convert boxes to tensors
    gt_corners = np.stack([gt.gt_corners() for gt in gts])  # M x 8 x 3
    gt_corners = torch.from_numpy(gt_corners)

    # get predicted corners
    pred_corners = np.stack([pred.prediction_corners() for pred in preds])  # N x 8 x 3
    pred_corners = torch.from_numpy(pred_corners)

    # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
    intersection_vol, iou_3d = box3d_overlap(gt_corners.float(), pred_corners.float())
    return iou_3d.data.numpy()


def l2(gts, preds):
    gt_centers = np.transpose(np.stack([gt.gt_center for gt in gts]), axes=(0, 2, 1)).reshape((-1, 1, 3))  # M x 3
    pred_centers = np.transpose(np.stack([pred.center for pred in preds]), axes=(0, 2, 1)).reshape((1, -1, 3))
    dists = np.linalg.norm(gt_centers[:, :, :2] - pred_centers[:, :, :2], axis=2)
    return dists


