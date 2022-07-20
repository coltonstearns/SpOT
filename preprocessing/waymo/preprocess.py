from spot.data.loading.scene import SceneIO
import os
import glob
import argparse
import struct
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from preprocessing.common.bounding_box import Box3D, associate
from preprocessing.waymo.points_in_boxes import points_in_box_regions
from waymo_open_dataset.utils import range_image_utils, transform_utils
import time
from pyquaternion import Quaternion
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset

from preprocessing.waymo.load_objects import extract_lidar_labels, extract_predictions
import tqdm
from multiprocessing import Pool, Lock
import open3d as o3d

import json

os.environ["CUDA_VISIBLE_DEVICES"]="1"
global predictions

hash2idx = {}


class WaymoConverter(object):

    def __init__(self, load_dir, save_dir, results_file, num_proc, split, box_context_scale_factor, match_threshold, match_type):
        self.lidar_list = ['_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT', '_SIDE_LEFT']

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.num_proc = int(num_proc)
        self.match_threshold = match_threshold
        self.match_type = match_type
        assert split in ["train", "val", "test"]
        self.split = split
        self.box_context_scale_factor = box_context_scale_factor

        self.tfrecord_pathnames = sorted(glob.glob(os.path.join(self.load_dir, '*.tfrecord')))

        # load waymo tracking predictions that we're using
        try:  # if results file is a directory, use all files in it
            self.results_file_list = os.listdir(results_file)
            self.results_file_list = [os.path.join(results_file, filename) for filename in self.results_file_list]
        except NotADirectoryError:
            self.results_file_list = [results_file]


    def convert_parallel(self):
        global predictions
        predictions = extract_predictions(self.results_file_list)
        print("start converting ...")
        with Pool(self.num_proc) as p:
            r = list(tqdm.tqdm(p.imap(self.convert_one, range(len(self))), total=len(self)))
        print("\nfinished ...")

    def convert(self):
        global predictions
        predictions = extract_predictions(self.results_file_list)
        print("start converting ...")
        progress_bar = tqdm.tqdm(total=len(self), desc='Pre-processing...', dynamic_ncols=True)
        for idx in range(len(self)):
            self.convert_one(idx)
            progress_bar.update()
            break
        progress_bar.close()
        print("\nfinished ...")

    def convert_one(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        # Check that all frames in the dataset have the same sequence name
        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame_idx == 0:
                scene_name = frame.context.name
            if frame.context.name != scene_name:
                raise SystemExit("NOT ALL FRAMES BELONG TO THE SAME SEQUENCE. ABORTING THE CONVERSION!")

        # create SceneIO object
        scene_objects = SceneIO(scene_name, base_dir=os.path.join(self.save_dir, self.split))

        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            # # parse calibration files
            calibration = self.decode_calib(frame)

            # parse point clouds
            success, boxes, points = self.decode_lidar(frame, calibration)

            # write new frame in SceneIO
            timestamp = frame.timestamp_micros * 1e-6
            scene_objects.init_new_sample_if_not_exist(timestep=timestamp, keyframe_id=str(frame.timestamp_micros))

            # record info if have boxes and points
            if success:
                world2ego = np.linalg.inv(calibration['T_sdc_global'])
                keyframe_id = str(frame.timestamp_micros)
                scene_objects.add_frame(boxes, points, keyframe_id, world2ego, is_keyframe=True)

        # close SceneIO and dump its contents to json
        scene_objects.close_filesystem_pipe()

        # save scene metadata (dense object data already saved)
        serialized_scene_objects = scene_objects.dump_to_dict()
        with open(os.path.join(self.save_dir, self.split, f'scene_%s_objects.json' % scene_name), 'w') as f:
            json.dump(serialized_scene_objects, f)

    def decode_calib(self, frame):
        calibration_parameters = {}
        for lidar in frame.context.laser_calibrations:
            if lidar.name == 1:  # 1 equals to the top lidar
                T_sdc_lidar = np.linalg.inv(np.array(lidar.extrinsic.transform).reshape(4, 4))

        calibration_parameters['T_sdc_lidar'] = T_sdc_lidar

        T_sdc_global = np.reshape(np.array(frame.pose.transform), [4, 4])
        calibration_parameters['T_sdc_global'] = T_sdc_global

        return calibration_parameters

    def decode_lidar(self, frame, calibration):

        # Extract lidar data in form of rays in the world coordinate system
        range_images, range_image_top_pose = parse_range_image_and_camera_projection(frame)

        # First return
        points_0, intensity_0, elongation_0 = convert_range_image_to_point_cloud(
                frame,
                range_images,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)

        # Second return
        points_1, intensity_1, elongation_1 = convert_range_image_to_point_cloud(
                frame,
                range_images,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # Transform the points back to the SDV coordinate system
        T_global_sdc = np.linalg.inv(calibration['T_sdc_global'])
        points_sdc = (T_global_sdc[:3, :3] @ points[:, :3].transpose() + T_global_sdc[:3, 3:4]).transpose()

        # get prediction boxes and transform into sdc reference frame
        global predictions
        if frame.context.name not in predictions:
            print("Scene %s is not in these predictions!" % frame.context.name)
            return False, None, None
        if frame.timestamp_micros not in predictions[frame.context.name]:
            print("Scene %s Frame %s has no predictions." % (frame.context.name, str(frame.timestamp_micros)))
            return False, None, None

        frame_predictions = predictions[frame.context.name][frame.timestamp_micros]

        # get label boxes
        frame_labels = extract_lidar_labels(frame)

        # match predictions and labels
        associated_boxes = associate(frame_labels, frame_predictions, threshold=self.match_threshold, distance_type=self.match_type)

        # extract object points in boxes
        success, box_points = points_in_box_regions(associated_boxes, points, self.box_context_scale_factor)

        # visualize:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud_points = o3d.utility.Vector3dVector(points[:, :3])
        point_cloud.points = point_cloud_points
        point_cloud_colors = np.array([[0.6, 0.6, 0.6]]).repeat(points.shape[0], axis=0)

        # get 3d rotation matrix
        o3d_boxes = []
        global hash2idx
        colors = [(0.00784313725490196, 0.24313725490196078, 1.0), (1.0, 0.48627450980392156, 0.0), (0.10196078431372549, 0.788235294117647, 0.2196078431372549), (0.9098039215686274, 0.0, 0.043137254901960784), (0.5450980392156862, 0.16862745098039217, 0.8862745098039215), (0.9098039215686274, 0.0, 0.043137254901960784)]
        for i, pred in enumerate(frame_predictions):
            center = pred.center
            if str(pred.instance_id) not in hash2idx.keys():
                clr_idx = len(hash2idx)
                hash2idx[str(pred.instance_id)] = len(hash2idx)
            else:
                clr_idx = hash2idx[str(pred.instance_id)]

            clr = list(colors[clr_idx % len(colors)])
            orient = Quaternion(axis=[1, 0, 0], angle=pred.orientation.item()).rotation_matrix
            wlh = pred.wlh  # size (3,1)
            viz_box = o3d.geometry.OrientedBoundingBox(center=center, R=orient, extent=wlh)
            viz_box.color = np.array(clr)
            points_in_box_idxs = viz_box.get_point_indices_within_bounding_box(point_cloud_points)
            point_cloud_colors[np.array(points_in_box_idxs)] = np.array([clr]).repeat(len(points_in_box_idxs), axis=0)
            o3d_boxes.append(viz_box)
        point_cloud.colors = o3d.utility.Vector3dVector(point_cloud_colors)
        o3d.visualization.draw_geometries([point_cloud] + o3d_boxes)


        return success, associated_boxes, box_points

    def __len__(self):
        return len(self.tfrecord_pathnames)


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       range_image_top_pose,
                                       ri_index=0):
    """Convert range images to point cloud.
    Args:
        frame (:obj:`Frame`): Open dataset frame.
        range_images (dict): Mapping from laser_name to list of two
            range images corresponding with two returns.
        range_image_top_pose (:obj:`Transform`): Range image pixel pose for
            top lidar.
        ri_index (int): 0 for the first return, 1 for the second return.
            Default: 0.
    Returns:
        tuple[list[np.ndarray]]: (List of points with shape [N, 3],
            camera projections of points with shape [N, 6], intensity
            with shape [N, 1], elongation with shape [N, 1]). All the
            lists have the length of lidar numbers (5).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    intensity = []
    elongation = []

    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(tf.convert_to_tensor(value=range_image_top_pose.data), range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(range_image_top_pose_tensor[..., 0],
                                                                               range_image_top_pose_tensor[..., 1],
                                                                               range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(range_image_top_pose_tensor_rotation, range_image_top_pose_tensor_translation)
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(tf.constant([c.beam_inclination_min, c.beam_inclination_max]), height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data),
            range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == open_dataset.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_mask = range_image_tensor[..., 0] > 0

        # filter out points in no-label zone:
        nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
        range_image_mask = range_image_mask & nlz_mask

        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())

        intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                        tf.where(range_image_mask))
        intensity.append(intensity_tensor.numpy())

        elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                         tf.where(range_image_mask))
        elongation.append(elongation_tensor.numpy())

    return points, intensity, elongation


def parse_range_image_and_camera_projection(frame):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == open_dataset.LaserName.TOP:
        range_image_top_pose_str_tensor = tf.io.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = open_dataset.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

    if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
      range_image_str_tensor = tf.io.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

  return range_images, range_image_top_pose


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # , load_dir, save_dir, results_file, num_proc, split, box_context_scale_factor
    parser.add_argument('--load_dir', type=str, help='Directory to load Waymo Open Dataset tfrecords')
    parser.add_argument('--save_dir', type=str, help='Directory to save converted data')
    parser.add_argument('--results_file', type=str, help='Path to file containing original tracking results.')
    parser.add_argument('--split', type=str, help='One of [train, val, test]')
    parser.add_argument('--box-context-scale-factor', type=float, default=1.25, help='Crop points of this proportion around each predicted box.')
    parser.add_argument('--num_proc', default=1, help='Number of processes to spawn')
    parser.add_argument('--match_thresh', type=float, default=0.15, help='Within this threshold, we consider it a true-positive for confidence training.')
    parser.add_argument('--match_type', type=str, default="3D-IOU", help='Either 3D-IOU or L2.')
    args = parser.parse_args()

    converter = WaymoConverter(args.load_dir, args.save_dir, args.results_file, args.num_proc, args.split, args.box_context_scale_factor,
                               args.match_thresh, args.match_type)
    # converter.convert_parallel()
    converter.convert()

