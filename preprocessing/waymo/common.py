import numpy as np
import numba
import pickle
import re
import open3d as o3d
from collections import defaultdict
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import torch


# import torch

def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path, 'rb')
    return pickle.load(file)


def save_pkl(obj, path):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def register_odometry_np(src, T_src_glo, T_tgt_glo, normals=False):
    M = np.linalg.inv(T_tgt_glo) @ T_src_glo

    # if the input are normals transform with the transpose of the inverse trans matrix
    if normals:
        M = np.linalg.inv(M).transpose()

    R, t = M[:3, :3], M[:3, 3][:, None]

    src = (R @ src.T + t).T

    return src


def global_vel_to_ref(vel, global_from_ref_rotation):
    # inverse means ref_from_global, rotation_matrix for normalization
    vel = [vel[0], vel[1], 0]
    ref = np.dot(global_from_ref_rotation.transpose(), vel)
    ref = [ref[0], ref[1], 0.0]

    return ref


def extract_camera_labels(frame):
    cam_labels = defaultdict(list)

    for camera in sorted(frame.camera_labels, key=lambda i: i.name):

        for label in camera.labels:
            cam_labels['cam_{}'.format(camera.name)].append({
                'name': label.id,
                'label': label.type,
                '2D_bbox': np.array([label.box.center_x, label.box.center_y,
                                     label.box.length, label.box.width], dtype=np.float32)
            })

    return cam_labels


def extract_projected_labels(frame):
    name_map = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}

    proj_camera_labels = defaultdict(list)

    for camera in sorted(frame.projected_lidar_labels, key=lambda i: i.name):
        for label in camera.labels:
            proj_camera_labels['cam_{}'.format(camera.name)].append({
                'name': label.id[0:label.id.find(name_map[camera.name]) - 1],
                'label': label.type,
                '3D_bbox_proj': np.array([label.box.center_x, label.box.center_y,
                                          label.box.length, label.box.width], dtype=np.float32)
            })

    return proj_camera_labels


def kabsch_transformation_estimation(x1, x2, weights=None, eps=1e-7):
    """
    numpy differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimated rotation matrix it then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.
    Args:
        x1            (numpy array): points of the first point cloud [b,n,3]
        x2            (numpy array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (numpy array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]

    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]

    """

    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
        x2 = torch.from_numpy(x2)

    if x1.ndim == 2:
        x1 = x1.unsqueeze(0)

    if x2.ndim == 2:
        x2 = x2.unsqueeze(0)

    if weights is None:
        weights = torch.ones(x1.shape[0], x1.shape[1]).type_as(x1).to(x1.device)

    elif isinstance(weights, np.ndarray):
        x1 = torch.from_numpy(weights).type_as(x1).to(x1.device)

    weights = weights.unsqueeze(2)

    x1_mean = torch.matmul(weights.transpose(1, 2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1, 2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           (x2_centered * weights))

    u, s, v = torch.svd(cov_mat)

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(
        torch.cat((torch.ones((tm_determinant.shape[0], 2), device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v, torch.matmul(determinant_matrix, u.transpose(1, 2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1, 2) - torch.matmul(rotation_matrix, x1_mean.transpose(1, 2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters

    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res


def transform_point_cloud(x1, R, t):
    """
    Transforms the point cloud using the giver transformation paramaters

    Args:
        x1  (np array): points of the point cloud [b,n,3]
        R   (np array): estimated rotation matrice [b,3,3]
        t   (np array): estimated translation vectors [b,3,1]
    Returns:
        x1_t (np array): points of the transformed point clouds [b,n,3]
    """

    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if isinstance(R, np.ndarray):
        R = torch.from_numpy(R)

    R = R.type_as(x1)
    t = t.type_as(x1)

    if len(R.shape) != 3:
        R = R.unsqueeze(0)
    if len(t.shape) != 3:
        t = t.unsqueeze(0)
    if len(x1.shape) != 3:
        x1 = x1.unsqueeze(0)

    x1_t = (torch.matmul(R, x1.transpose(2, 1)) + t).transpose(2, 1)

    return x1_t


def get_3d_bbox_coords(bbox3d):
    x, y, z = bbox3d[0], bbox3d[1], bbox3d[2]
    length, width, height = bbox3d[3], bbox3d[4], bbox3d[5]
    heading = bbox3d[-1]

    # Computes the coordinates of the bbox corners
    l2 = length / 2
    w2 = width / 2
    h2 = height / 2

    translation = np.array([x, y, z]).reshape(1, 3)

    P1 = np.array([-l2, -w2, -h2])
    P2 = np.array([-l2, -w2, h2])
    P3 = np.array([-l2, w2, h2])
    P4 = np.array([-l2, w2, -h2])
    P5 = np.array([l2, -w2, -h2])
    P6 = np.array([l2, -w2, h2])
    P7 = np.array([l2, w2, h2])
    P8 = np.array([l2, w2, -h2])

    # Get the rotation matrix from the heading angle
    yaw_rotation = Rotation.from_rotvec(heading * np.array([0, 0, 1])).as_matrix()

    corners = np.stack([P1, P2, P3, P4, P5, P6, P7, P8], axis=0)

    corners = np.matmul(yaw_rotation, corners.transpose()).transpose() + translation

    return corners


def is_within_3d_bbox(points, box, normals=None, return_points_in_bbox_frame=False):
    """Checks whether a point is in a 3d box given a set of points and a box.
        Args:
            point: [N, 3] tensor. Inner dims are: [x, y, z].
            box: [7] tensor. Inner dims are: [center_x, center_y, center_z, length,
            width, height, heading].
            name: tf name scope.
        Returns:
            point_in_box; [N,] boolean array.
    """

    center = box[0:3]
    dim = box[3:6]
    heading = box[-1]

    # Get the rotation matrix from the heading angle
    rotation = Rotation.from_rotvec(heading * np.array([0, 0, 1])).as_matrix()

    # [4, 4]
    transform = rot_trans_to_se3(rotation, center)
    # [4, 4]
    transform = np.linalg.inv(transform)
    # [3, 3]
    rotation = transform[0:3, 0:3]
    # [3]
    translation = transform[0:3, 3]

    # [M, 3]
    points_in_box_frames = np.matmul(rotation, points.transpose()).transpose() + translation

    # [M, 3]
    point_in_box = np.logical_and(
        np.logical_and(points_in_box_frames <= dim * 0.5,
                       points_in_box_frames >= -dim * 0.5),
        np.all(np.not_equal(dim, 0), axis=-1, keepdims=True))

    # [N, M]
    point_in_box = np.prod(point_in_box, axis=-1).astype(bool)

    if not return_points_in_bbox_frame:
        return point_in_box
    else:
        if normals is not None:
            T_normals = np.linalg.inv(transform).transpose()

            normals_in_bbox_frame = np.matmul(T_normals[0:3, 0:3],
                                              normals[point_in_box, :].transpose()).transpose() + T_normals[0:3, 3]

            return points_in_box_frames[point_in_box, :], normals_in_bbox_frame / np.linalg.norm(normals_in_bbox_frame,
                                                                                                 axis=1, keepdims=True)
        else:
            return points_in_box_frames[point_in_box, :]


def points_in_bboxes(points, bboxes):
    """Checks whether a point is in any of the bboxes
        Args:
            points: [N, 3] tensor. Inner dims are: [x, y, z].
            boxes: [7] list of bboxes
        Returns:
            bbox_idx: [N,] idx of a bbox for each point (-1 denotes background points)
    """

    bbox_idx = -np.ones(points.shape[0])

    for crnt_ind, bbox in enumerate(bboxes):
        temp_bbox_ind = is_within_3d_bbox(points, bbox)
        bbox_idx[temp_bbox_ind] = crnt_ind

    return bbox_idx


def rot_trans_to_se3(rot, trans):
    if rot.ndim > 2:
        T = np.eye(4)
        T = np.tile(T, (rot.shape[0], 1, 1))
        T[:, 0:3, 0:3] = rot
        T[:, 0:3, 3] = trans.reshape(-1, 3, )

    else:
        T = np.eye(4)
        T[0:3, 0:3] = rot
        T[0:3, 3] = trans.reshape(3, )

    return T


def compute_depth_projections(points, cp, calibration, use_distance=False):
    # Columns of the data
    # Img_id1, u_1, v_1, Img_id2, u_2, v_2
    # At most 2 projections are considered
    # If point only maps to one image the other id is -1

    columns = [0, 1, 2, 4, 5, 6]
    cp_padded = np.zeros((cp.shape[0], cp.shape[1] + 2))
    cp_padded[:, columns] = cp

    # Convert points to homogeneous coordinates
    points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    # Compute the depth of all points by projecting them to the corresponding camera frame
    # Five cameras in total
    for i in range(5):
        ## FIRST PROJECTION
        # i + 1 as camera labels go from 1 to 5
        idx_1 = np.where(cp_padded[:, 0] == (i + 1))[0]

        extrinsic = np.linalg.inv(calibration['cam_{}'.format(i + 1)]['T_cam_sdc'])
        points_camera_frame = np.matmul(extrinsic, points[idx_1, :].transpose()).transpose()

        if use_distance:
            dists = np.linalg.norm(points_camera_frame[:, :3], axis=1)
            cp_padded[idx_1, 3] = dists
        else:
            depth = points_camera_frame[:, 0]
            cp_padded[idx_1, 3] = depth

        ## SECOND PROJECTION
        # i + 1 as camera labels go from 1 to 5
        idx_1 = np.where(cp_padded[:, 4] == (i + 1))[0]
        extrinsic = np.linalg.inv(calibration['cam_{}'.format(i + 1)]['T_cam_sdc'])
        points_camera_frame = np.matmul(extrinsic, points[idx_1, :].transpose()).transpose()

        if use_distance:
            dists = np.linalg.norm(points_camera_frame[:, :3], axis=1)
            cp_padded[idx_1, 7] = dists
        else:
            depth = points_camera_frame[:, 0]
            cp_padded[idx_1, 7] = depth

    return cp_padded


def reconstruct_dynamic_object(meta_bbox, pose, laser_data_list, bbox_ind_list):
    start_frame = meta_bbox['f_idx'][0]
    obj_idx = meta_bbox['unique_idx']
    target_bbox = get_3d_bbox_coords(meta_bbox['bbox'][0])

    # Extract the target point cloud
    target_pc = laser_data_list[start_frame]
    target_pc = target_pc[np.where(bbox_ind_list[start_frame] == obj_idx)[0], :]

    # test plot
    pcd_target_pc = o3d.geometry.PointCloud()
    pcd_source_pc = o3d.geometry.PointCloud()
    pcd_bbox = o3d.geometry.PointCloud()

    pcd_target_pc.points = o3d.utility.Vector3dVector(target_pc)
    pcd_bbox.points = o3d.utility.Vector3dVector(target_bbox)
    pcd_bbox.paint_uniform_color([1, 0, 0])
    pcd_target_pc.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd_target_pc, pcd_bbox])

    for idx, f_idx in enumerate(meta_bbox['f_idx'][1:]):
        source_pc = laser_data_list[f_idx]
        source_pc = source_pc[np.where(bbox_ind_list[f_idx] == obj_idx)[0], :]

        source_bbox = get_3d_bbox_coords(meta_bbox['bbox'][idx + 1])

        R_rel, t_rel, _ = kabsch_transformation_estimation(source_bbox, target_bbox)

        source_pc = transform_point_cloud(source_pc, R_rel, t_rel).squeeze(0).numpy()
        pcd_source_pc.points = o3d.utility.Vector3dVector(source_pc)
        pcd_source_pc.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([pcd_target_pc, pcd_source_pc, pcd_bbox])
        pcd_target_pc += pcd_source_pc


def multi_vis(pcds, names, render=True, width=960, height=540, shift=100):
    """
    Visulise point clouds in multiple windows, we allow at most 4 windows

    Input:
        pcds:   a list of pcds
        names:  a list of window names
    """
    assert len(pcds) == len(names)
    n_windows = len(pcds)

    window_corners = [
        [0, 0],
        [0, height + shift],
        [width, 0],
        [width + shift, height + shift]
    ]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0, 0, 0])

    # estimate normals for better visualisation
    if (render):
        for each_pcd in pcds:
            each_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    # initialise the windows
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window(window_name=names[0], width=width, height=height, left=window_corners[0][0],
                        top=window_corners[0][1])
    vis_1.add_geometry(pcds[0])
    vis_1.add_geometry(mesh_frame)

    if (n_windows >= 2):
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(window_name=names[1], width=width, height=height, left=window_corners[1][0],
                            top=window_corners[1][1])
        vis_2.add_geometry(pcds[1])
        vis_2.add_geometry(mesh_frame)
        if (n_windows >= 3):
            vis_3 = o3d.visualization.Visualizer()
            vis_3.create_window(window_name=names[2], width=width, height=height, left=window_corners[2][0],
                                top=window_corners[2][1])
            vis_3.add_geometry(pcds[2])
            vis_3.add_geometry(mesh_frame)
            if (n_windows >= 4):
                vis_4 = o3d.visualization.Visualizer()
                vis_4.create_window(window_name=names[3], width=width, height=height, left=window_corners[3][0],
                                    top=window_corners[3][1])
                vis_4.add_geometry(pcds[3])
                vis_4.add_geometry(mesh_frame)

    # start rendering
    while True:
        vis_1.update_geometry(pcds[0])
        vis_1.update_geometry(mesh_frame)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()

        if (n_windows >= 2):
            vis_2.update_geometry(pcds[1])
            vis_2.update_geometry(mesh_frame)
            if not vis_2.poll_events():
                break
            vis_2.update_renderer()

            cam = vis_1.get_view_control().convert_to_pinhole_camera_parameters()
            cam2 = vis_2.get_view_control().convert_to_pinhole_camera_parameters()
            cam2.extrinsic = cam.extrinsic
            vis_2.get_view_control().convert_from_pinhole_camera_parameters(cam2)

            if (n_windows >= 3):
                vis_3.update_geometry(pcds[2])
                vis_3.update_geometry(mesh_frame)
                if not vis_3.poll_events():
                    break
                vis_3.update_renderer()

                cam3 = vis_3.get_view_control().convert_to_pinhole_camera_parameters()
                cam3.extrinsic = cam.extrinsic
                vis_3.get_view_control().convert_from_pinhole_camera_parameters(cam3)

                if (n_windows >= 4):
                    vis_4.update_geometry(pcds[3])
                    vis_4.update_geometry(mesh_frame)
                    if not vis_4.poll_events():
                        break
                    vis_4.update_renderer()

                    cam4 = vis_4.get_view_control().convert_to_pinhole_camera_parameters()
                    cam4.extrinsic = cam.extrinsic
                    vis_4.get_view_control().convert_from_pinhole_camera_parameters(cam4)

    vis_1.destroy_window()
    if (n_windows >= 2):
        vis_2.destroy_window()

        if (n_windows >= 3):
            vis_3.destroy_window()

            if (n_windows >= 4):
                vis_4.destroy_window()


def plot_image_labels(image, labels):
    f = plt.figure(figsize=(20, 15))
    ax = f.add_subplot(111)

    for label in labels:
        # Draw the object bounding box.
        ax.add_patch(patches.Rectangle(
            xy=(label['box'][0] - 0.5 * label['box'][2],
                label['box'][1] - 0.5 * label['box'][3]),
            width=label['box'][2],
            height=label['box'][3],
            linewidth=1,
            edgecolor='red',
            facecolor='none'))

    # Show the camera image.
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.show()