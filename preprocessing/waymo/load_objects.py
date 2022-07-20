import numpy as np
from waymo_open_dataset.protos import metrics_pb2
from preprocessing.common.bounding_box import Box3D
from preprocessing.waymo.common import global_vel_to_ref
import tqdm
from spot.io.globals import WAYMO_CLASSPROTO2NAME


def extract_predictions(results_path):
    """
    Note: these are in global reference frame.
    Args:
        results_path (list<str>): List of all paths containing predictions. Handles multiple paths due to 2GB size limit
        on proto files.
    """
    # load in predictions
    if isinstance(results_path, str):
        results_path = [results_path]

    results = {}

    for path in results_path:
        all_predictions = metrics_pb2.Objects()
        with open(path, 'rb') as f:
            all_predictions.ParseFromString(f.read())

        # create formatted dictionary of all results
        for o in tqdm.tqdm(all_predictions.objects):
            context_name = o.context_name
            timestamp = o.frame_timestamp_micros
            if context_name not in results:
                results[context_name] = {}
            if timestamp not in results[context_name]:
                results[context_name][timestamp] = []

            # extract box
            box = o.object.box
            center = np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1)
            lwh = np.array([box.length, box.width, box.height]).reshape(3, 1)
            orientation = np.array([box.heading]).reshape([1, 1])

            # extract velocity + acceleration (if applicable)
            extra_info = o.object.metadata
            vel = np.array([extra_info.speed_x, extra_info.speed_y, 0]).reshape(3, 1)

            # get other info
            class_name = WAYMO_CLASSPROTO2NAME[o.object.type]
            score = o.score
            instance_id = o.object.id

            # create object
            formatted_pred = Box3D(center,
                                   orientation,
                                   lwh,
                                   vel,
                                   score,
                                   class_name,
                                   None,
                                   instance_id,
                                   timestamp*1e-6)

            results[context_name][timestamp].append(formatted_pred)

    return results


def extract_lidar_labels(frame):
    pose = np.reshape(np.array(frame.pose.transform), [4, 4])

    objects = []
    for object_id, label in enumerate(frame.laser_labels):
        class_name = WAYMO_CLASSPROTO2NAME[label.type]
        box = label.box

        # Speed and acceleration are given in the global coordinate frame
        speed = [label.metadata.speed_x, label.metadata.speed_y]
        accel = [label.metadata.accel_x, label.metadata.accel_y]
        num_lidar_points_in_box = label.num_lidar_points_in_box

        # Difficulty level is 0 if labeler did not say this was LEVEL_2.
        # Set difficulty level of "999" for boxes with no points in box.
        if num_lidar_points_in_box <= 0:
            combined_difficulty_level = 999
        if label.detection_difficulty_level == 0:
            # Use points in box to compute difficulty level.
            if num_lidar_points_in_box >= 5:
                combined_difficulty_level = 1
            else:
                combined_difficulty_level = 2
        else:
            combined_difficulty_level = label.detection_difficulty_level

        # Convert global velocity to the reference frame of the SDC
        ref_velocity = global_vel_to_ref(speed, pose[0:3, 0:3])
        ref_acceleration = global_vel_to_ref(accel, pose[0:3, 0:3])
        gt_ismoving = (np.linalg.norm(ref_velocity) > 0.05).item()

        formatted_label = Box3D(center=None,
                                orientation=None,
                                wlh=None,
                                velocity=None,
                                score=None,
                                class_name=class_name,
                                attribute=combined_difficulty_level,
                                instance_id=None,
                                timestep=frame.timestamp_micros*1e-6,
                                with_gt_box=True,
                                gt_instance_id=object_id,
                                gt_center=np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1),
                                gt_orientation=np.array([box.heading]).reshape([1, 1]),
                                gt_wlh=np.array([box.length, box.width, box.height]).reshape(3, 1),
                                gt_velocity=np.array(ref_velocity).reshape(3, 1),
                                gt_acceleration=np.array(ref_acceleration).reshape(3, 1),
                                gt_is_moving=gt_ismoving)

        objects.append(formatted_label)

    return objects