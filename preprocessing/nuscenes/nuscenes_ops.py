from preprocessing.common.bounding_box import Box3D
import numpy as np
from pyquaternion import Quaternion


def nusc_boxes2our_boxes(nusc_boxes, timestep, box_type="prediction"):
    converted_boxes = []
    for nusc_box in nusc_boxes:
        # note, prediction boxes are expected to be tracking-type boxes
        if box_type == "prediction":
            velocity_3d = np.zeros((3, 1))
            if not np.any(np.isnan(nusc_box.velocity)):
                velocity_3d[:2, 0] = np.array(nusc_box.velocity)
            box = Box3D(center=np.array(nusc_box.translation).reshape(3, 1),
                        orientation=Quaternion(nusc_box.rotation),
                        wlh=np.array(nusc_box.size).reshape(3, 1),
                        velocity=velocity_3d,
                        score=nusc_box.tracking_score,
                        class_name=nusc_box.tracking_name,
                        attribute=None,
                        instance_id=nusc_box.tracking_id,
                        timestep=timestep)

        # note, gt boxes are expecte to be detection-type boxes
        elif box_type == "ground-truth":
            velocity_3d = np.zeros((3, 1))
            if not np.any(np.isnan(nusc_box.velocity)):
                velocity_3d[:2, 0] = np.array(nusc_box.velocity)
            acceleration_3d = np.zeros((3, 1))
            if not np.any(np.isnan(nusc_box.acceleration)):
                acceleration_3d[:2, 0] = np.array(nusc_box.acceleration)
            gt_is_moving = attribute2motion(nusc_box.attribute_name)
            box = Box3D(center=None,
                        orientation=None,
                        wlh=None,
                        velocity=None,
                        score=None,
                        class_name=nusc_box.detection_name,
                        attribute=None,
                        instance_id=None,
                        timestep=timestep,
                        gt_instance_id=nusc_box.instance_token,
                        with_gt_box=True,
                        gt_center=np.array(nusc_box.translation).reshape(3, 1),
                        gt_orientation=Quaternion(nusc_box.rotation),
                        gt_wlh=np.array(nusc_box.size).reshape(3, 1),
                        gt_velocity=velocity_3d,
                        gt_acceleration=acceleration_3d,
                        gt_is_moving=gt_is_moving
                        )
        else:
            raise RuntimeError("Box Type must be one of ['prediction', 'ground-truth']")
        converted_boxes.append(box)
    return converted_boxes


def attribute2motion(attribute):
    if len(attribute) == 0:
        return False

    if attribute.startswith("vehicle"):
        is_moving = True if attribute == "vehicle.moving" else False
    elif attribute.startswith("pedestrian"):
        is_moving = True if attribute == "pedestrian.moving" else False
    elif attribute.startswith("cycle"):
        is_moving = True if attribute == "cycle.with_rider" else False
    else:
        is_moving = False
    return is_moving