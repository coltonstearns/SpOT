import torch
import torch_geometric.nn as geo_nn
from collections import OrderedDict

# from spot.models.deprecated.orig.spot import CaSPR
from spot.models.tpointnet2 import TPointNet2
from spot.io.globals import NUSCENES_BINNING_CLUSTER_SIZES, NUSCENES_CLASSES, WAYMO_CLASSES, WAYMO_BINNING_CLUSTER_SIZES
import numpy as np
from spot.data.box3d import LidarBoundingBoxes


def load_models(backbone_config, loss_config, dataset_source, object_class, model_in_path, device, parallel_train=False):
    if dataset_source == "nuscenes":
        class_idx = NUSCENES_CLASSES[object_class]
        size_bins = np.array(NUSCENES_BINNING_CLUSTER_SIZES[class_idx])
    else:  # waymo
        class_idx = WAYMO_CLASSES[object_class]
        size_bins = np.array(WAYMO_BINNING_CLUSTER_SIZES[class_idx])

    # create spot model
    tpointnet_encoder = TPointNet2(cfg=backbone_config,
                                   loss_params=loss_config,
                                   size_clusters=size_bins,
                                   object_class=object_class,
                                   parallel=parallel_train)

    if model_in_path != '':
        print('Loading model weights from %s...' % (model_in_path))
        loaded_state_dict = torch.load(model_in_path)
        for network in loaded_state_dict:
            state_dict = loaded_state_dict[network]
            for k, v in state_dict.items():
                if k.split('.')[0] == 'module':
                    # then it was trained with Data parallel
                    print('Loading weights trained with DataParallel...')
                    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if
                                  k.split('.')[0] == 'module'}
                    break

            loaded_state_dict[network] = state_dict

        # UPDATING OLD WEIGHT FIELDS
        loaded_state_dict['tpointnet_state'].pop('bbox_regressor.binning_loss.weight', None)

        # load model
        tpointnet_encoder.load_state_dict(loaded_state_dict['tpointnet_state'], strict=True)

    if parallel_train:
        tpointnet_encoder = geo_nn.DataParallel(tpointnet_encoder)


    tpointnet_encoder.to(device)
    return tpointnet_encoder


def split_inputs(data_list, max_pnts_per_batch):
    # get rolling point counts
    count = torch.tensor([data.num_nodes for data in data_list])
    cumsum = count.cumsum(0)
    cumsum = torch.cat([cumsum.new_zeros(1), cumsum], dim=0)

    # compute how many divisions we need
    num_divisions, prev_split_idx = 1, 0
    cumsum_cp = cumsum.clone()
    for i, cumcount in enumerate(cumsum_cp):
        if cumcount > max_pnts_per_batch:
            if i == prev_split_idx:
                raise RuntimeError("Error: A single object sequence exceeds our max_pnts_per_batch.")
            cumsum_cp[i:] -= cumcount.clone()
            num_divisions += 1
            prev_split_idx = i

    # compute optimal split locations
    split_id = (num_divisions) * cumsum.to(torch.float) / cumsum[-1].item()
    split_id = (split_id[:-1] + split_id[1:]) / 2.0
    split_id = split_id.to(torch.long)  # round.
    split = split_id.bincount().cumsum(0)
    split = torch.cat([split.new_zeros(1), split], dim=0)
    split = torch.unique(split, sorted=True)
    split = split.tolist()

    # split each data-list
    split_data_list = []
    for i in range(len(split)-1):
        data_list_segment = data_list[split[i]:split[i + 1]]
        split_data_list.append(data_list_segment)
    return split_data_list

def update_dict(dictionary, update):
    for k, val in update.items():
        if k not in dictionary:
            dictionary[k] = val
        elif k in dictionary and torch.is_tensor(val):
            dictionary[k] = torch.cat([dictionary[k], val])
        elif k in dictionary and isinstance(val, LidarBoundingBoxes):
            dictionary[k] = LidarBoundingBoxes.concatenate([dictionary[k], val])
        else:
            pred_list = [torch.cat([dictionary[k][j], val[j]]) for j in range(len(val))]
            dictionary[k] = tuple(pred_list)


def parse_waymo_detection_resultstring(ret_texts, object_class):
    ap_dict = OrderedDict()
    ap_keys = [
        'Vehicle/L1 mAP',
        'Vehicle/L1 mAPH',
        'Vehicle/L2 mAP',
        'Vehicle/L2 mAPH',
        'Pedestrian/L1 mAP',
        'Pedestrian/L1 mAPH',
        'Pedestrian/L2 mAP',
        'Pedestrian/L2 mAPH',
        'Sign/L1 mAP',
        'Sign/L1 mAPH',
        'Sign/L2 mAP',
        'Sign/L2 mAPH',
        'Cyclist/L1 mAP',
        'Cyclist/L1 mAPH',
        'Cyclist/L2 mAP',
        'Cyclist/L2 mAPH',
        'Overall/L1 mAP',
        'Overall/L1 mAPH',
        'Overall/L2 mAP',
        'Overall/L2 mAPH'
    ]
    for k in ap_keys:
        ap_dict[k] = 0.0

    mAP_splits = ret_texts.split('mAP ')
    mAPH_splits = ret_texts.split('mAPH ')
    for idx, key in enumerate(ap_dict.keys()):
        split_idx = int(idx // 2) + 1
        if idx % 2 == 0:  # mAP
            ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
        else:  # mAPH
            ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])

    # only keep our class type
    orig_keys = list(ap_dict.keys())
    for k in orig_keys:
        if object_class not in k.lower():
            del ap_dict[k]
    return ap_dict

def parse_waymo_tracking_resultstring(ret_texts, object_class):
    mota_dict = OrderedDict()
    mota_keys = [
        'Vehicle/L1 MOTA',
        'Vehicle/L1 MOTP',
        'Vehicle/L1 Miss',
        'Vehicle/L1 Mismatch',
        'Vehicle/L1 FP',
        'Vehicle/L2 MOTA',
        'Vehicle/L2 MOTP',
        'Vehicle/L2 Miss',
        'Vehicle/L2 Mismatch',
        'Vehicle/L2 FP',
        'Pedestrian/L1 MOTA',
        'Pedestrian/L1 MOTP',
        'Pedestrian/L1 Miss',
        'Pedestrian/L1 Mismatch',
        'Pedestrian/L1 FP',
        'Pedestrian/L2 MOTA',
        'Pedestrian/L2 MOTP',
        'Pedestrian/L2 Miss',
        'Pedestrian/L2 Mismatch',
        'Pedestrian/L2 FP',
        'Sign/L1 MOTA',
        'Sign/L1 MOTP',
        'Sign/L1 Miss',
        'Sign/L1 Mismatch',
        'Sign/L1 FP',
        'Sign/L2 MOTA',
        'Sign/L2 MOTP',
        'Sign/L2 Miss',
        'Sign/L2 Mismatch',
        'Sign/L2 FP',
        'Cyclist/L1 MOTA',
        'Cyclist/L1 MOTP',
        'Cyclist/L1 Miss',
        'Cyclist/L1 Mismatch',
        'Cyclist/L1 FP',
        'Cyclist/L2 MOTA',
        'Cyclist/L2 MOTP',
        'Cyclist/L2 Miss',
        'Cyclist/L2 Mismatch',
        'Cyclist/L2 FP',
    ]
    for k in mota_keys:
        mota_dict[k] = 0.0

    mota_splits = ret_texts.split('MOTA ')
    motp_splits = ret_texts.split('MOTP ')
    miss_splits = ret_texts.split('Miss ')
    mismatch_splits = ret_texts.split('Mismatch ')
    fp_splits = ret_texts.split('FP ')

    for idx, key in enumerate(mota_dict.keys()):
        split_idx = int(idx // 5) + 1
        if idx % 5 == 0:  # MOTA
            mota_dict[key] = float(mota_splits[split_idx].split(']')[0])
        elif idx % 5 == 1:  # MOTP
            mota_dict[key] = float(motp_splits[split_idx].split(']')[0])
        elif idx % 5 == 2:  # Miss
            mota_dict[key] = float(miss_splits[split_idx].split(']')[0])
        elif idx % 5 == 3:  # Mismatch
            mota_dict[key] = float(mismatch_splits[split_idx].split(']')[0])
        else:  # FP
            mota_dict[key] = float(fp_splits[split_idx].split(']')[0])

    # only keep our class type
    orig_keys = list(mota_dict.keys())
    for k in orig_keys:
        if object_class not in k.lower():
            del mota_dict[k]
    return mota_dict

