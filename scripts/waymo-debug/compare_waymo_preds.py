import wandb
import numpy as np
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from preprocessing.common.bounding_box import Box3D
import tqdm
from spot.io.globals import WAYMO_CLASSPROTO2NAME
from spot.io.wandb_utils import get_wandb_boxes

WAYMO_NAME2CLASSPROTO = {
    'vehicle': label_pb2.Label.TYPE_VEHICLE,
    'pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
    'sign': label_pb2.Label.TYPE_SIGN,
    'cyclist': label_pb2.Label.TYPE_CYCLIST,
}


def load_results(results_path):
    results = {}

    all_predictions = metrics_pb2.Objects()
    with open(results_path, 'rb') as f:
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
                               timestamp * 1e-6)

        results[context_name][timestamp].append(formatted_pred)

    return results


def predictions_to_output(predictions, outfile):
    """
    Note: these are in global reference frame.
    """
    # load in predictions
    all_predictions = metrics_pb2.Objects()

    for context_name in predictions:
        for timestep in predictions[context_name]:
            for box in predictions[context_name][timestep]:
                center = box.center.flatten()
                yaw = box.orientation.flatten()
                wlh = box.wlh.flatten()
                velocity = box.velocity.flatten()
                class_name = box.class_name
                id = box.instance_id
                score = box.score

                box = label_pb2.Label.Box()
                box.center_x = center[0]
                box.center_y = center[1]
                box.center_z = center[2]
                box.length = wlh[0]
                box.width = wlh[1]
                box.height = wlh[2]
                if box.width < 0 or box.length < 0 or box.height < 0:
                    print("Size Error!!!")
                box.heading = yaw[0]

                o = metrics_pb2.Object()
                o.object.box.CopyFrom(box)
                o.object.type = WAYMO_NAME2CLASSPROTO[class_name]
                o.score = score
                o.context_name = context_name
                o.frame_timestamp_micros = timestep
                all_predictions.objects.append(o)

    with open(outfile, 'wb') as f:
        f.write(all_predictions.SerializeToString())


def convert2wandbboxes(loaded_boxes, color):
    centers_1 = np.stack([box.center.flatten() for box in loaded_boxes if box.class_name == "vehicle"], axis=0)
    yaws_1 = np.stack([box.orientation.flatten() for box in loaded_boxes if box.class_name == "vehicle"], axis=0)
    sizes_1 = np.stack([box.wlh.flatten() for box in loaded_boxes if box.class_name == "vehicle"], axis=0)
    boxes_1 = np.hstack([centers_1, sizes_1, yaws_1])
    colors_1 = color.repeat(boxes_1.shape[0], axis=0)
    wandb_boxes_1 = get_wandb_boxes(boxes_1, colors_1)
    return wandb_boxes_1


def convert_all_confidences_to_1(results_path, outfile):
    results = load_results(results_path)
    # for context_name in results:
    #     for timestep in results[context_name]:
    #         for box in results[context_name][timestep]:
    #             box.score = 1
    #
    # predictions_to_output(results, outfile)


def compare_detections_wandb_vis(results_path_1, results_path_2):
    wandb.init(project="waymo-compare", config={})
    loaded_boxes_1 = load_results(results_path_1)
    loaded_boxes_2 = load_results(results_path_2)

    for scene_id in loaded_boxes_1.keys():
        for timestamp_id in loaded_boxes_1[scene_id].keys():
            wandb_boxes_1 = convert2wandbboxes(loaded_boxes_1[scene_id][timestamp_id], np.array([[255, 0, 0]]))
            wandb_boxes_2 = convert2wandbboxes(loaded_boxes_2[scene_id][timestamp_id], np.array([[0, 255, 0]]))

            wandb_boxes = np.array(wandb_boxes_1.tolist() + wandb_boxes_2.tolist())
            wandb.log({"Compare Predictions": wandb.Object3D({"type": "lidar/beta", "points": np.zeros((1, 6)), "boxes": wandb_boxes})})

if __name__ == "__main__":
    # results_path = "/home/colton/Documents/SpOT/vivid_sweep_5_good_tracking.bin"
    # outpath = "/home/colton/Documents/SpOT/normalized_confs_vivid_sweep_5_good_tracking.bin"
    results_path = "/home/colton/Documents/SpOT/stilted_sweep_2_bad_tracking.bin"
    outpath = "/home/colton/Documents/SpOT/normalized_confs_stilted_sweep_2_bad_tracking.bin"
    convert_all_confidences_to_1(results_path, outpath)


    # results_path_1 = "/home/colton/Documents/vehicle_tracking_results.bin"
    # results_path_2 = "/home/colton/Documents/CenterPointResults/waymo/val/tracking-results-thresh0.60.bin"
    # compare_detections_wandb_vis(results_path_1, results_path_2)


