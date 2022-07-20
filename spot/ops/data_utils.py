
import argparse
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
import torch

# raw world point cloud sequences will be given timestamps from 0 to this
DEFAULT_MAX_TIMESTAMP = 5.0
# numbser of steps in each sequence
DEFAULT_EXPECTED_SEQ_LEN = 10
# number of points at each tep
DEFAULT_EXPECTED_NUM_PTS = 4096
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# "train-crop", "train-shift-augment", "crop-wlh-pct", "crop-center-pct", "crop-yaw-rad", "crop-frame-wlh-pct", "crop-frame-center-pct", "crop-frame-yaw-rad", "scale-noise", "rotation-noise"
# {'wlh_pct': [0.45, 0.75], 'center_pct': [-0.09, 0.09], 'yaw_rad': [-np.pi/8, np.pi/8],
#                              'frame_wlh_pct': [-0.05, 0.05], 'frame_center_pct': [-0.025, 0.025], 'frame_yaw_rad': [-np.pi/20, np.pi/20]}


def kitti2nusc(kitti_boxes):
    nusc_boxes = kitti_boxes.clone()
    nusc_boxes[:, 2] += nusc_boxes[:, 5] / 2
    nusc_boxes[:, [3, 4]] = nusc_boxes[:, [4, 3]]
    nusc_boxes[:, 6] = (-nusc_boxes[:, 6] - np.pi/2) % (2*np.pi)
    nusc_boxes[:, 6][nusc_boxes[:, 6] > np.pi] -= 2*np.pi
    return nusc_boxes


def nusc2kitti(nusc_boxes):
    kitti_boxes = nusc_boxes.clone()
    kitti_boxes[:, 2] -= kitti_boxes[:, 5] / 2
    kitti_boxes[:, [3, 4]] = kitti_boxes[:, [4, 3]]
    kitti_boxes[:, 6] = (-kitti_boxes[:, 6] - np.pi/2) % (2*np.pi)
    kitti_boxes[:, 6][kitti_boxes[:, 6] > np.pi] -= 2*np.pi
    return kitti_boxes


def pclist_to_pctensor(pclist, num_pts, min_pts=1, sampling_idxs=None):
    """
    Converts a list of T point clouds with variable numbers of points into an array of size (T, num_pts, pnt_dims).
    To each num_pts, point clouds with more points are randomly downsampled and point clouds with fewer points are
    randomly duplicated.
    **Note:
    Args:
        pclist (list[np.ndarray]): list of point clouds
        num_pts (int): number of points to sample each point cloud to
        min_pts (int): if a point cloud has less than this number of points, we ignore it. Must be >= 1.
        sampling_idxs (list[np.ndarray]): Specifies which point-indices to use for up or downsampling.

    Returns:
        batch_point_cloud (np.ndarray): combined point cloud of size (T', num_pts, pnt_dims). T' is the number of
                                        point clouds with >= min_pts.
        sampled_idxs (list[np.ndarray] or None): list of indices used to up or down-sample each point cloud.
                                         len(sampled_idxs) = T'
        min_pts_mask (np.ndarray): boolean mask of size (t,) which has True if pclist[i].shape[0] >= min_pts and False
                                   otherwise
    """
    if len(pclist) == 0:
        raise RuntimeError("Input list of tensors is empty!")

    sampled_pcs = []
    sampled_idxs = []
    min_pts_mask = []
    pc_means = []
    last_dim = pclist[0].shape[1]
    for i, pc in enumerate(pclist):
        pc_npoints = pc.shape[0]
        if pc_npoints < min_pts:
            min_pts_mask.append(False)
            sampled_pcs.append(np.zeros((num_pts, last_dim)))
            sampled_idxs.append(np.zeros(num_pts))
            pc_means.append(np.zeros(3))
            continue

        pc_mean = np.mean(pc, axis=0)
        if pc_npoints < num_pts:
            if sampling_idxs is not None:
                sampled_pc, cur_sampled_idxs = buffer_pc(pc, num_pts, sampling_idxs[i])
            else:
                sampled_pc, cur_sampled_idxs = buffer_pc(pc, num_pts, None)
        elif pc_npoints > num_pts:
            if sampling_idxs is not None:
                cur_sampled_idxs = sampling_idxs[i]
            else:
                cur_sampled_idxs = np.random.choice(np.arange(pc_npoints), num_pts, replace=False)
            sampled_pc = pc[cur_sampled_idxs, :]
        else:
            cur_sampled_idxs = np.arange(pc_npoints)
            sampled_pc = pc

        sampled_pcs.append(sampled_pc)
        sampled_idxs.append(cur_sampled_idxs)
        min_pts_mask.append(True)
        pc_means.append(pc_mean)

    sampled_tensors = np.stack(sampled_pcs, axis=0)
    return sampled_tensors, sampled_idxs, np.array(min_pts_mask, dtype=bool), pc_means


def buffer_pc(pc, num_pts, indices=None):
    """
    Given a point cloud with less than a desired num_pts, repeats and randomly samples the points to buffer
    it to num_pts size.
    """
    num_repeats = num_pts // pc.shape[0]
    buffered_pc = pc.repeat(num_repeats, axis=0)
    if indices is not None:
        sampled_idxs = indices
    else:
        sampled_idxs = np.random.choice(np.arange(pc.shape[0]), num_pts % pc.shape[0], replace=False)
    buffered_pc = np.concatenate((buffered_pc, buffered_pc[sampled_idxs, :]), axis=0)
    return buffered_pc, sampled_idxs


def get_points_in_box(points, center, dxdydz, yaw):
    """
    *Note: It is best to use a CUDA and batched version of this method.
    Given points and a single bounding box, computes a mask of which points lie inside the bounding box.
    """
    if type(yaw) == float:
        quat = Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw)
    elif type(yaw) == Quaternion:
        quat = yaw
    else:
        raise RuntimeError("Yaw must be a float or a quaternion.")
    unit_points = points - center.reshape(1, 3)
    unit_points = (quat.rotation_matrix.T @ unit_points.T).T
    unit_points /= dxdydz.reshape(1, 3)
    in_box_mask = np.all(np.abs(unit_points) <= 0.5, axis=1)
    return in_box_mask


def get_random_noise(dxdydz, wlh_noise_range, center_noise_range, yaw_noise_range, per_frame_pct, only_per_frame=False):
    frame_wlh_noise = np.random.uniform(1 + wlh_noise_range[0] * per_frame_pct, 1 + wlh_noise_range[1] * per_frame_pct, size=(1, 3))
    frame_center_noise = np.random.uniform(center_noise_range[0] * per_frame_pct, center_noise_range[1] * per_frame_pct, size=(1, 3)) * dxdydz.reshape(1, 3)
    frame_yaw_noise = np.random.uniform(yaw_noise_range[0] * per_frame_pct, yaw_noise_range[1] * per_frame_pct) % (2 * np.pi)
    frame_yaw_noise = Quaternion(axis=(0.0, 0.0, 1.0), radians=frame_yaw_noise)

    if not only_per_frame:
        wlh_noise = np.random.uniform(1 + wlh_noise_range[0], 1 + wlh_noise_range[1], size=(1, 3))
        center_noise = np.random.uniform(center_noise_range[0], center_noise_range[1], size=(1, 3))
        yaw_noise = np.random.uniform(yaw_noise_range[0], yaw_noise_range[1]) % (2 * np.pi)
        yaw_noise = Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw_noise)
    else:
        wlh_noise, center_noise, yaw_noise = None, None, None

    return (wlh_noise, center_noise, yaw_noise), (frame_wlh_noise, frame_center_noise, frame_yaw_noise)


def get_nocs(points, center, sizes, yaw):
    if type(yaw) == float:
        quat = Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw)
    elif type(yaw) == Quaternion:
        quat = yaw
    else:
        raise RuntimeError("Yaw must be a float or a quaternion.")
    nocs_points = points - center.reshape(1, 3)
    nocs_points = (quat.rotation_matrix.T @ nocs_points.T).T
    nocs_points /= np.max(sizes)
    return nocs_points


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def rotmat_yaw(rotmat: np.ndarray) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(rotmat, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def get_glob_nusc_boxes(sample_id):
    nusc = NuScenes(version="v1.0-trainval", dataroot="/mnt/fsx/scratch/colton.stearns/pi/tri-nuscenes")
    sample = nusc.get('sample', sample_id)
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    boxes = nusc.get_boxes(sample_data['token'])

    # format into Caspr-format
    caspr_boxes = []
    for box in boxes:
        formatted_box = np.concatenate([box.center, box.wlh, np.array([quaternion_yaw(Quaternion(box.orientation))])])
        caspr_boxes.append(formatted_box)
    caspr_boxes = np.stack(caspr_boxes)
    caspr_boxes[:, [3, 4]] = caspr_boxes[:, [4, 3]]

    return caspr_boxes

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def bboxes_tpointnet2global(boxes, frame2batch, origin2ego_trans, glob2ego_affine):
    boxes_global = boxes.clone()
    for k in range(boxes.size(0)):
        local2global = origin2ego_trans[k].view(1, 3)
        boxes_global[k, :, :3] += local2global
        ego2glob_affine = torch.inverse(glob2ego_affine[k])  # 4 x 4 tensor
        formatted_box = affine_transform_boxes(boxes_global[k, :, :], ego2glob_affine)
        boxes_global[k:k+1] = formatted_box
    return boxes_global


def affine_transform_boxes(boxes, affine):
    boxes_transformed = boxes.clone()
    boxes_transformed[:, :3] = (boxes_transformed[:, :3].double() @ affine[:3, :3].T)
    boxes_transformed[:, :3] += affine[:3, 3:4].T

    # get updated yaw
    v = affine[:3, :3].float() @ torch.tensor([1, 0, 0]).view(3, 1).to(boxes)
    affine_yaw = torch.atan2(v[1], v[0])  # size (1,)
    yaws = boxes[:, 6] + affine_yaw
    yaws = yaws % (2*np.pi)
    yaws[yaws > np.pi] -= 2 * np.pi
    boxes_transformed[:, 6] = yaws

    return boxes_transformed

def points_tpointnet2global(points, origin2ego_trans, glob2ego_affine):
    B, T, N, _ = points.size()
    global_points = points.clone()
    global_points[:, :, :, :3] += origin2ego_trans.view(B, 1, 1, 3)
    for k in range(points.size(0)):
        frame_affine = torch.inverse(glob2ego_affine[k])  # 4 x 4 tensor
        global_points[k, :, :, :3] = (global_points.double()[k, :, :, :3] @ frame_affine[:3, :3].T) + frame_affine[:3, 3].view(1, 1, 3)

    return global_points

def vis_boxes(boxes, boxes2=None, name="boxes_viz.eps"):
    fig, ax = plt.subplots()
    ax.scatter(boxes[:, 0].to('cpu').data.numpy(),
               boxes[:, 1].to('cpu').data.numpy())
    for i in range(boxes.size(0)):
        box_corner = boxes[i, :2] - boxes[i, 3:5]
        ax.add_patch(Rectangle((box_corner[0], box_corner[1]),
                               boxes[i, 3], boxes[i, 4],
                               fc='none',
                               color='yellow',
                               linewidth=1,
                               linestyle="dotted"))
    if boxes2 is not None and boxes2.size(0) > 0:
        for i in range(boxes2.size(0)):
            box_corner = boxes2[i, :2] - boxes2[i, 3:5]
            ax.add_patch(Rectangle((box_corner[0], box_corner[1]),
                                   boxes2[i, 3], boxes2[i, 4],
                                   fc='none',
                                   color='red',
                                   linewidth=1,
                                   linestyle="dotted"))

    plt.xlabel("X-AXIS")
    plt.ylabel("Y-AXIS")
    plt.title("PLOT-2")
    plt.savefig(name, format='eps')