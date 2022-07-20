import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import scipy
from nuscenes.utils.data_classes import LidarPointCloud
from matplotlib import rcParams
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image
import os.path as osp
from typing import Dict, Tuple

import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes









def merge_two_boxes(box1, box2):
    # get approximated union-box center and orientation
    center = (box1.center + box2.center) / 2
    orientation = Quaternion.slerp(q0=box1.orientation, q1=box2.orientation, amount=0.5)
    vel = (box1.velocity + box2.velocity) / 2

    # compute all corners of original two boxes
    box1_corners = box1.corners()
    box2_corners = box2.corners()
    corners = np.hstack((box1_corners, box2_corners))

    # transform into union-box coordinate frame
    all_corners = np.linalg.inv(orientation.rotation_matrix) @ (corners - center.reshape(3, 1))
    all_corners[[0, 1], :] = all_corners[[1, 0], :]  # correct for xyz --> wlh
    wlh = (np.max(np.abs(all_corners), axis=1)*2).tolist()

    return Box(center, wlh, orientation, label=box1.label, score=box1.score, velocity=vel, name=box1.name, token=box1.token)


def interpolate_box(box_prev, box_cur, interp_factor, category_name, token):
    interp_center = interp_factor * box_cur.center + (1 - interp_factor) * box_prev.center
    interp_vel = interp_factor * box_cur.velocity + (1 - interp_factor) * box_prev.velocity
    interp_rot = Quaternion.slerp(q0=box_prev.orientation, q1=box_cur.orientation, amount=interp_factor)
    interp_box = Box(interp_center, box_cur.wlh, interp_rot, label=box_cur.label, score=box_cur.score, velocity=interp_vel, name=category_name, token=token)
    return interp_box

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

def affine_transformation(point_cloud, transform):
    affine_extension = np.ones(point_cloud.shape[0]).reshape(point_cloud.shape[0], 1)
    return np.dot(transform, np.hstack((point_cloud, affine_extension)).T).T[:, :3]


def vis_cloud(point_cloud, mask=None):
    # Plot best fit plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud
    if mask is not None:
        c = np.where(mask, 'green', 'red')
    else:
        c = 'blue'
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], marker='.', alpha=0.5, color=c)
    ax.set_xlim(-.5, .5)
    ax.set_ylim(-.5, .5)
    ax.set_zlim(-.5, .5)
    plt.show()


def vis_boxes(bbox_locs, point_cloud):
    # Plot best fit plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(bbox_locs[:, 0], bbox_locs[:, 1], bbox_locs[:, 2], marker='o', color='red')

    # Plot point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], marker='.', alpha=0.05, color='green')

    plt.show()


def vis_plane(plane_pnt, plane_normal, point_cloud):
    # compute a grid of points on the plane
    plane_xs = np.arange(-20, 20, 0.25)
    plane_ys = np.arange(-20, 20, 0.25)
    X, Y = np.meshgrid(plane_xs, plane_ys)
    Z = (-plane_normal[0] * X - plane_normal[1] * Y + np.dot(plane_normal, plane_pnt)) / plane_normal[2]

    # Plot best fit plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5)

    # Plot point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='green')

    # plot plane normal line
    norm_xs = np.arange(-20, 20, 0.25) * plane_normal[0]
    norm_ys = np.arange(-20, 20, 0.25) * plane_normal[1]
    norm_zs = np.arange(-20, 20, 0.25) * plane_normal[2]
    ax.plot(norm_xs, norm_ys, norm_zs, color='red')

    plt.show()


def fit_plane_to_points(points):
    """
    Computer best-fit plane of points, and returns a point on the plane + the unit normal vector of the plane
    :param points:
    :return:
    """
    n = points.shape[0]
    A = np.hstack((points[:, :2], np.ones(n).reshape(n, 1)))
    C, _, _, _ = scipy.linalg.lstsq(A, points[:, 2])

    z_int, y_int, x_int = np.array([0, 0, C[2]]), np.array([0, -C[2]/C[1], 0]), np.array([-C[2]/C[0], 0, 0])
    normal = np.cross(z_int - y_int, z_int - x_int)
    normal = normal / np.linalg.norm(normal)
    return z_int, normal


def compute_road_mask(plane_pnt, plane_normal, points, plane_threshold=0.1):
    plane_offsets = points - plane_pnt.reshape(1, 3)
    dist_to_plane = np.abs(np.dot(plane_offsets, plane_normal.reshape(3, 1)))

    road_idxs = np.where(dist_to_plane > plane_threshold, 1, 0)
    return road_idxs


def compute_road_mask_v2(plane_pnt, plane_normal, points, plane_threshold=0.1):
    plane_offsets = points - plane_pnt.reshape(1, 3)
    dist_to_plane = np.abs(np.dot(plane_offsets, plane_normal.reshape(3, 1)))

    road_idxs = np.where(dist_to_plane > plane_threshold, True, False)
    return road_idxs


def getGreedyPerm(seed_cloud, K):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : 1D Array ((N-1) + (N-2) + ...); 1D array
        A compact distance matrix
    N : number of points
    K : number of points to sample
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    """
    detection_threshold = np.linalg.norm(seed_cloud, axis=1) < 50  # outside detection range
    height_threshold = np.where(seed_cloud[:, 2] < 10, True, False)  # above 10 meters in the air ... most likely noise
    mask = detection_threshold & height_threshold

    idx = 0
    ds = np.linalg.norm(seed_cloud[:, :2] - seed_cloud[idx, :2], axis=1) * mask  # only want x-y spread
    point_idxs = np.zeros(K, dtype=np.int64)
    for i in range(1, K):
        idx = np.argmax(ds)
        point_idxs[i] = idx
        next_distances = np.linalg.norm(seed_cloud[:, :2] - seed_cloud[idx, :2], axis=1) * mask
        ds = np.minimum(ds, next_distances)
    return point_idxs


def compact_to_row(D, n, idx):  # start as D, N, 0
    dense_distance_row = []
    for i in range(n):
        if i != idx:
            dense_distance_row.append(D[i+idx])
        else:
            dense_distance_row.append(0)
    return np.array(dense_distance_row)


def render_annotation(nusc_explorer,
                      anntoken: str,
                      margin: float = 10,
                      view: np.ndarray = np.eye(4),
                      box_vis_level: BoxVisibility = BoxVisibility.ANY,
                      out_path: str = None,
                      extra_info: bool = False,
                      extra_boxes = None) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc_explorer.nusc.get('sample_annotation', anntoken)
    sample_record = nusc_explorer.nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    for cam in cams:
        _, boxes, _ = nusc_explorer.nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                selected_anntokens=[anntoken])
        if len(boxes) > 0:
            break  # We found an image that matches. Let's abort.
    assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
                           'Try using e.g. BoxVisibility.ANY.'
    assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    cam = sample_record['data'][cam]

    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc_explorer.nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    # LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    sample_data = nusc_explorer.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    pc, _ = LidarPointCloud.from_file_multisweep(nusc_explorer.nusc, sample_record, sample_data['channel'], 'LIDAR_TOP', nsweeps=10)
    pc.render_height(axes[0], view=view)


    if extra_boxes is not None:
        boxes = boxes + extra_boxes
    for box in boxes:
        c = np.array(nusc_explorer.get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    data_path, boxes, camera_intrinsic = nusc_explorer.nusc.get_sample_data(cam, selected_anntokens=[anntoken])
    im = Image.open(data_path)
    axes[1].imshow(im)
    axes[1].set_title(nusc_explorer.nusc.get('sample_data', cam)['channel'])
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    for box in boxes:
        c = np.array(nusc_explorer.get_color(box.name)) / 255.0
        box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c))

    # Print extra information about the annotation below the camera view.
    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc_explorer.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc_explorer.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)


def from_file_multisweep(nusc: 'NuScenes',
                         sample_rec: Dict,
                         chan: str,
                         ref_chan: str,
                         nsweeps: int = 5,
                         min_distance: float = 1.0) -> Tuple['PointCloud', np.ndarray, list]:
    """
    MODIFIED FROM NUSCENES DEVKIT
    Return a point cloud that aggregates multiple sweeps.
    UNLIKE IN NUSCENES DEVKIT, WE DO NOT map the coordinates to a single reference frame.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
    :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
    :param nsweeps: Number of sweeps to aggregated.
    :param min_distance: Distance below which points are discarded.
    :return: (all_pc, all_times, close_points_masks). The aggregated point cloud and timestamps.
    """
    # Init.
    cls = LidarPointCloud
    points = np.zeros((cls.nbr_dims(), 0), dtype=np.float32 if cls == LidarPointCloud else np.float64)
    all_pc = cls(points)
    all_times = np.zeros((1, 0))
    close_points_masks = []

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data'][chan]
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        close_points_mask = get_close_point_mask(current_pc, min_distance)
        current_pc.points = current_pc.points[:, close_points_mask]

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        times = time_lag * np.ones((1, current_pc.nbr_points()))
        all_times = np.hstack((all_times, times))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # Record close-points-mask
        close_points_masks.append(close_points_mask)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return all_pc, all_times, close_points_masks


def load_gt_local(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    if eval_split == 'test':
        for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):
            all_annotations.add_boxes(sample_token, [])
        return all_annotations

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                this_box = box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name
                    )
                this_box.instance_token = sample_annotation['instance_token']
                this_box.token = sample_annotation['token']
                this_box.name = sample_annotation['category_name']
                this_box.acceleration = box_acceleration(nusc, sample_annotation['token'])[:2]
                sample_boxes.append(this_box)

            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations

def box_angular_velocity(nusc, sample_annotation_token: str, max_time_diff: float = 1.5):
    """
    Estimate the angular velocity for an annotation.
    If possible, we compute the difference between the previous and next frame.
    Otherwise we use the difference between the current and previous/next frame.
    If the angular velocity cannot be estimated, we return np.nan.
    :param sample_annotation_token: Unique sample_annotation identifier.
    :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
    :return: float. Angular velocity in counter-clockwise radians.
    """
    current = nusc.get('sample_annotation', sample_annotation_token)
    has_prev = current['prev'] != ''
    has_next = current['next'] != ''

    # Cannot estimate velocity for a single annotation.
    if not has_prev and not has_next:
        return np.nan

    if has_prev:
        first = nusc.get('sample_annotation', current['prev'])
    else:
        first = current

    if has_next:
        last = nusc.get('sample_annotation', current['next'])
    else:
        last = current

    pos_last = last['rotation']
    pos_first = first['rotation']
    angular_diff = (quaternion_yaw(pos_last) - quaternion_yaw(pos_first) + np.pi) % (2*np.pi) - np.pi
    time_last = 1e-6 * nusc.get('sample', last['sample_token'])['timestamp']
    time_first = 1e-6 * nusc.get('sample', first['sample_token'])['timestamp']
    time_diff = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    if time_diff > max_time_diff:
        # If time_diff is too big, don't return an estimate.
        return np.nan
    else:
        return angular_diff / time_diff


def box_acceleration(nusc, sample_annotation_token: str, max_time_diff: float = 1.5):
    current = nusc.get('sample_annotation', sample_annotation_token)
    has_prev = current['prev'] != ''
    has_next = current['next'] != ''
    has_next_next, has_prev_prev = False, False
    if has_prev:
        has_prev_prev = nusc.get('sample_annotation', current['prev'])['prev'] != ''
    if has_next:
        has_next_next = nusc.get('sample_annotation', current['next'])['next'] != ''

    # to get a valid acceleration estimate, we need two unique velocities
    cond1 = has_next and has_next_next
    cond2 = has_prev and has_prev_prev
    cond3 = has_next and has_prev
    if not (cond1 or cond2 or cond3):
        return np.array([0, 0, 0])

    if has_prev:
        first = nusc.get('sample_annotation', current['prev'])
    else:
        first = current

    if has_next:
        last = nusc.get('sample_annotation', current['next'])
    else:
        last = current

    vel_last = nusc.box_velocity(last['token'])
    vel_first = nusc.box_velocity(first['token'])
    vel_diff = vel_last - vel_first

    time_last = 1e-6 * nusc.get('sample', last['sample_token'])['timestamp']
    time_first = 1e-6 * nusc.get('sample', first['sample_token'])['timestamp']
    time_diff = time_last - time_first

    if has_next and has_prev:
        # If doing centered difference, allow for up to double the max_time_diff.
        max_time_diff *= 2

    if time_diff > max_time_diff:
        # If time_diff is too big, don't return an estimate.
        return np.array([0, 0, 0])
    else:
        return vel_diff / time_diff


def get_close_point_mask(pc, radius: float):
    """
    Function adapted from nuscenes/utils/data_classes.py
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(pc.points[0, :]) < radius
    y_filt = np.abs(pc.points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return not_close


def bboxes_glob2sensor(boxes: list, ego_pose, sensor):
    """
    Moves NuScenes boxes in the global coordinate frame into the specified sensor coordinate frame.
    Args:
        boxes :list['Box']: List of different objects' bounding boxes
        ego_pose :dict: NuScenes ego_pose attribute
        sensor :dict: NuScenes calibrated_sensor attribute

    Returns (:list['Box'], np.ndarray): Object bounding boxes transformed into sensor reference frame, as well as the affine
    matrix that performed the transformation.

    """
    # transform boxes from global coordinate frame to sensor's coordinate frame
    transformed_boxes = []

    # create affine transform from global to ego
    affine_glob2ego_rot = np.eye(4)
    affine_glob2ego_rot[:3, :3] = Quaternion(ego_pose['rotation']).inverse.rotation_matrix
    affine_glob2ego_trans = np.eye(4)
    affine_glob2ego_trans[:3, 3] = -np.array(ego_pose['translation'])

    # create affine transform from ego to sensor
    affine_ego2sensor_rot = np.eye(4)
    affine_ego2sensor_rot[:3, :3] = Quaternion(sensor['rotation']).inverse.rotation_matrix
    affine_ego2sensor_trans = np.eye(4)
    affine_ego2sensor_trans[:3, 3] = -np.array(sensor['translation'])

    # create affine from global to sensor
    affine_glob2sensor = affine_ego2sensor_rot @ affine_ego2sensor_trans @ affine_glob2ego_rot @ affine_glob2ego_trans

    for box in boxes:
        box = box.copy()
        box.rotate(Quaternion(matrix=affine_glob2sensor[:3, :3]))
        box.translate(affine_glob2sensor[:3, 3:4])
        transformed_boxes.append(box)
    return transformed_boxes, affine_glob2sensor

