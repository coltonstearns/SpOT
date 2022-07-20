import torch

import numpy as np
from third_party.mmdet3d.ops.roiaware_pool3d.points_in_boxes import points_in_boxes_batch, points_in_boxes_cpu

from preprocessing.common.points import ObjectPoints


def points_in_box_regions(boxes, points, scale_factor):
    if len(boxes) == 0 or points.shape[0] == 0:
        return False, None

    # get box-point-mask around each proposed region
    boxes_mmdet = bboxes2mmdetformat(boxes, gt=False)
    boxes_mmdet[:, 3:6] *= scale_factor
    point_mask = mmdet_boxes_point_assignments(boxes_mmdet, points)

    # if applicable, get GT box region
    withgt_mask = np.array([box.with_gt_box for box in boxes])
    gt_point_masks = np.zeros((len(boxes), points.shape[0]), dtype=np.bool)
    if np.sum(withgt_mask) > 0:
        gt_boxes_mmdet = bboxes2mmdetformat(boxes, gt=True)
        truepositive_gt_point_masks = mmdet_boxes_point_assignments(gt_boxes_mmdet, points)
        gt_point_masks[withgt_mask] = truepositive_gt_point_masks

    # format into list of points
    box_points, box_segmentations, num_pnts = [], [], []
    for i in range(len(boxes)):
        # extract this box-region points
        this_box_points = points[point_mask[i, :], :]

        # extract appropriate gt info
        if withgt_mask[i]:
            this_box_segmentation = gt_point_masks[i][point_mask[i, :]].reshape(-1, 1)
            npnts = np.sum(this_box_segmentation)
        else:
            this_box_segmentation = np.array([])
            npnts = this_box_points.shape[0]

        # record this box info
        box_points.append(this_box_points)
        box_segmentations.append(this_box_segmentation)
        num_pnts.append(int(npnts))

    # put together into our point object
    object_points = ObjectPoints(points=box_points,
                                 num_points=num_pnts,
                                 segmentation=box_segmentations,
                                 surrounding_context_factor=scale_factor)
    return True, object_points


def mmdet_boxes_point_assignments(boxes, points):
    boxes = torch.from_numpy(boxes)[:, :7]
    points = torch.from_numpy(points[:, :3]).contiguous()
    assignment_idxs = points_in_boxes_cpu(points, boxes)
    return assignment_idxs.data.numpy() > 0.5


def gpu_mmdet_boxes_point_assignments(boxes, points):
    boxes = torch.from_numpy(boxes).float().cuda().view(boxes.shape[0], 1, -1)[:, :, :7]
    points = torch.from_numpy(points[:, :3]).unsqueeze(dim=0).float().cuda().repeat(boxes.shape[0], 1, 1).contiguous()
    assignment_idxs = points_in_boxes_batch(points, boxes)  # returns B x N-pts x M-

    formatted_assignment_idxs = torch.zeros((boxes.size(0), points.size(1)), dtype=torch.bool).cuda()
    for i in range(len(boxes)):
        obj_point_mask = assignment_idxs[i, :, 0] == 1
        formatted_assignment_idxs[i, :] = obj_point_mask
    return formatted_assignment_idxs.cpu().data.numpy()


def bboxes2mmdetformat(boxes, gt=False):
    if gt:
        center_attr = "gt_center"
        yaw_attr = "gt_orientation"
        wlh_attr = "gt_wlh"
    else:
        center_attr = "center"
        yaw_attr = "orientation"
        wlh_attr = "wlh"

    if gt:
        mask = [box.with_gt_box for box in boxes]
        if sum(mask) == 0:
            return np.zeros((0, 7))
    else:
        mask = [True] * len(boxes)

    centers = np.concatenate([getattr(boxes[i], center_attr) for i in range(len(boxes)) if mask[i]], axis=1)
    yaws = np.concatenate([getattr(boxes[i], yaw_attr) for i in range(len(boxes)) if mask[i]], axis=1)
    lwhs = np.concatenate([getattr(boxes[i], wlh_attr) for i in range(len(boxes)) if mask[i]], axis=1)
    mmdet_boxes = np.transpose(np.concatenate([centers, lwhs, yaws], axis=0), axes=(1,0))

    # change coordinate systems to Kitti
    mmdet_boxes[:, 2] -= mmdet_boxes[:, 5] / 2
    mmdet_boxes[:, [3, 4]] = mmdet_boxes[:, [4, 3]]
    mmdet_boxes[:, 6] = (-mmdet_boxes[:, 6] - np.pi/2) % (2*np.pi)
    mmdet_boxes[:, 6][mmdet_boxes[:, 6] > np.pi] -= 2*np.pi
    return mmdet_boxes