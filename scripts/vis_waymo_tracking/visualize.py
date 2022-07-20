import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import scripts.vis_waymo_tracking.mot_3d.visualization as visualization, mot_3d.utils as utils
from scripts.vis_waymo_tracking.mot_3d.data_protos import BBox, Validity
from scripts.vis_waymo_tracking.waymo_loader import WaymoLoader
from scripts.vis_waymo_tracking.mot_3d.utils.geometry import iou3d, iou2d
import matplotlib.pyplot as plt


def load_gt_bboxes(gt_folder, data_folder, segment_name, type_token):
    gt_info = np.load(os.path.join(gt_folder, '{:}.npz'.format(segment_name)),
                      allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
                       allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info['types']
    gt_ids, gt_bboxes = utils.inst_filter(ids, bboxes, inst_types, type_field=[type_token], id_trans=True)

    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    # gt_bboxes = gt_bbox2world(gt_bboxes, egos)
    return gt_bboxes, gt_ids


def gt_bbox2world(bboxes, egos):
    frame_num = len(egos)
    for i in range(frame_num):
        ego = egos[i]
        bbox_num = len(bboxes[i])
        for j in range(bbox_num):
            bboxes[i][j] = BBox.bbox2world(ego, bboxes[i][j])
    return bboxes


def frame_visualization(bboxes, ids, bboxes2, ids2, gt_bboxes=None, gt_ids=None, pc=None, out_folder='', name=''):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))

    if pc is not None:
        visualizer.handler_pc(pc)
    for _, bbox in enumerate(gt_bboxes):
        visualizer.handler_box(bbox, message='', color='black')

    for _, (bbox, id) in enumerate(zip(bboxes, ids)):
        visualizer.handler_box(bbox, message='', color='light_blue')

    for _, (bbox, id) in enumerate(zip(bboxes2, ids2)):
        visualizer.handler_box(bbox, message='', color='red', linestyle='solid')

    # visualizer.show()
    visualizer.save(os.path.join(out_folder, '{:}.png'.format(name)))
    visualizer.close()


def frame_comparison(bboxes, ids, bboxes2, ids2, gt_bboxes=None, gt_ids=None, pc=None, out_folder='', name=''):
    # get bounding box differences
    box1_gains, box2_losses, outliers1 = [], [], []
    box2_gains, box1_losses, outliers2 = [], [], []
    for gt_box in gt_bboxes:
        max_iou_1 = np.argmax([iou3d(gt_box, bbox)[1] for bbox in bboxes])
        match1_score, iou2d_1 = iou3d(gt_box, bboxes[max_iou_1])
        match1 = match1_score >= 0.7

        max_iou_2 = np.argmax([iou3d(gt_box, bbox)[1] for bbox in bboxes2])
        match2_score, iou2d_2 = iou3d(gt_box, bboxes2[max_iou_2])
        match2 = match2_score >= 0.7

        if match1 and not match2:
            box1_gains.append(bboxes[max_iou_1])
            box2_losses.append(bboxes2[max_iou_2])
            outliers1.append(match2_score == 0)

        elif match2 and not match1:
            box2_gains.append(bboxes2[max_iou_2])
            box1_losses.append(bboxes[max_iou_1])
            outliers2.append(match2_score == 0)


    if len(box1_gains) == 0 and len(box2_gains) == 0:
        return


    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    if pc is not None:
        visualizer.handler_pc(pc)
    for _, bbox in enumerate(gt_bboxes):
        visualizer.handler_box(bbox, message='', color='black')
    for i, (bbox1, bbox2) in enumerate(zip(box1_gains, box2_losses)):
        message = "Outlier %s!" % i if outliers1[i] else ""
        visualizer.handler_box(bbox1, message=message, color='green')
        message = "Outlier %s!" % i if outliers1[i] else ""
        visualizer.handler_box(bbox2, message=message, color='red', linestyle='dashed')

    for i, (bbox1, bbox2) in enumerate(zip(box1_losses, box2_gains)):
        message = "Outlier %s!" % i if outliers2[i] else ""
        visualizer.handler_box(bbox2, message=message, color='purple')
        message = "Outlier %s!" % i if outliers2[i] else ""
        visualizer.handler_box(bbox1, message=message, color='dark_green', linestyle='dashed')

    # visualizer.show()
    visualizer.save(os.path.join(out_folder, '{:}.png'.format(name)))
    visualizer.close()


def get_diff_hist(bboxes, bboxes2):
    bboxes_arr = np.stack([BBox.bbox2array(box) for box in bboxes])  # n x 8
    bboxes_arr2 = np.stack([BBox.bbox2array(box) for box in bboxes2])  # m x 8
    diffs = np.abs(bboxes_arr.reshape((1, -1, 8)) - bboxes_arr2.reshape((-1, 1, 8)))  # m x n x 8
    diffs = np.min(diffs, axis=0)  # n x 8
    diff_xyz = np.sqrt(np.sum(diffs[:, :3]**2, axis=1)).tolist()
    diff_wlh = np.sqrt(np.sum(diffs[:, 3:6]**2, axis=1)).tolist()
    diff_yaw = diffs[:, 6].tolist()
    return diff_xyz, diff_wlh, diff_yaw



def visualize_mot(data_loader: WaymoLoader, data_loader2: WaymoLoader, sequence_id, gt_bboxes=None, gt_ids=None, out_folder=""):
    frame_num = len(data_loader)
    num_det1, num_dets2 = 0, 0
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(data_loader.type_token, sequence_id + 1, frame_index + 1,
                                                        frame_num))
        # input data
        frame_data = next(data_loader)
        frame_data2 = next(data_loader2)

        # visualize first predictions
        results_pred_bboxes = frame_data['dets']
        results_pred_ids = frame_data['det_ids']
        pc = frame_data['pc']

        # visualize second predictions
        results_pred_bboxes2 = frame_data2['dets']
        results_pred_ids2 = frame_data2['det_ids']

        num_det1 += len(results_pred_bboxes)
        num_dets2 += len(results_pred_bboxes2)

        # get GT visualization
        frame_visualization(results_pred_bboxes, results_pred_ids, results_pred_bboxes2, results_pred_ids2,
                            gt_bboxes[frame_index], gt_ids[frame_index], pc, out_folder,
                            name='{:}_{:}'.format(sequence_id, frame_index))
        # frame_visualization(results_pred_bboxes, results_pred_ids, results_pred_bboxes2, results_pred_ids2,
        #                     gt_bboxes[frame_index], gt_ids[frame_index], pc, out_folder,
        #                     name='{:}_{:}'.format(sequence_id, frame_index))

    print("SEQ {:}".format(sequence_id+1))
    print("Num Dets 1:")
    print(num_det1)
    print("Num Dets 2:")
    print(num_dets2)





def main(obj_type, data_folder, preds_folder1, preds_folder2, preds1_name, preds2_name, gt_folder, out_folder, start_scene=0, start_frame=0, vis_pc=True):
    # simply knowing about all the segments
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4

    for file_index, file_name in enumerate(file_names[:]):
        print(file_index)
        if file_index < start_scene:
            continue
        print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1, len(file_names)))
        segment_name = file_name.split('.')[0]
        data_loader = WaymoLoader([type_token], segment_name, data_folder, preds_folder1, start_frame, use_pc=vis_pc)
        data_loader2 = WaymoLoader([type_token], segment_name, data_folder, preds_folder2, start_frame, use_pc=vis_pc)
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder, segment_name, type_token)

        # visualize here
        visualize_mot(data_loader, data_loader2, file_index, gt_bboxes, gt_ids, out_folder)


    # stats_file = os.path.join(out_folder, "stats.npz")
    # np.savez(stats_file, xyz=np.array(pos_dist), wlh=np.array(size_dist), yaw=np.array(yaw_dist))
    #
    # n, bins, patches = plt.hist(pos_dist, 10000, density=True, facecolor='g', alpha=0.75)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_seq', type=int, default=0, help='start at a middle sequence for debug')
    parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
    parser.add_argument('--vis_pc', action='store_true', help='Also visualize point cloud data.')
    parser.add_argument('--obj_type', type=str, default='vehicle', choices=['vehicle', 'pedestrian', 'cyclist'])
    parser.add_argument('--data_folder', type=str, help='Path to SimpleTrack preprocessed data')
    parser.add_argument('--preds_folder1', type=str,
                        help='Path to folder with our tracking predictions (in SimpleTrack data format.')
    parser.add_argument('--preds_folder2', type=str,
                        help='Path to folder with our second tracking predictions (in SimpleTrack data format.')
    parser.add_argument('--preds1_name', type=str, default="Good")
    parser.add_argument('--preds2_name', type=str, default="Buggy")

    parser.add_argument('--gt_folder', type=str, help='Path to folder with GT boxes (in SimpleTrack data format).')
    parser.add_argument('--out_folder', type=str, help='Path to output visualizations.')


    args = parser.parse_args()

    out_folder = args.out_folder
    out_folder = os.path.join(out_folder, args.obj_type)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    main(args.obj_type, args.data_folder, args.preds_folder1, args.preds_folder2, args.preds1_name, args.preds2_name,
         args.gt_folder, out_folder, args.start_seq, args.start_frame, args.vis_pc)

