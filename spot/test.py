'''

This script can be used to test the CaSPR model. Use:

python test.py --help

'''
import os
import json

import torch

from spot.ops.torch_utils import get_device, count_params
from spot.io.logging_utils import log
from spot.io.run_utils import load_models, parse_waymo_detection_resultstring, parse_waymo_tracking_resultstring
from spot.data.sequences_dataset import load_dataset
import wandb
from spot.utils.test_utils import TrackAndEvaluateNuscenes, TrackAndEvaluateWaymo
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from torch_geometric.loader import DataListLoader
from spot.tracking_utils.evaluate_nuscenes import compute_motas
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["WANDB_MODE"] = "online"


def main(args):
    # get each configuration dictionary
    backbone_config = args["backbone"]
    trk_dataloading_config = args["testing_dataloading"]
    data_aug_config = args["data_augmentation"]
    losses_config = args["losses"]
    scheduling_config = args["scheduling"]
    general_config = args["general"]
    trajectory_config = args["track_motion_model"]
    trk_association_config = args["track_association"]
    trk_manage_config = args["track_management"]
    assert trk_dataloading_config["dataset-properties"]["tracking-mode"] == True

    # General options
    dataset_source = general_config['dataset_name']
    assert dataset_source in ['nuscenes', 'waymo']
    data_root = general_config['data_root']
    out_dir = general_config['out']
    parallel = general_config['parallel']
    max_pnts_in_batch = general_config['max_pnts_per_batch']
    object_class = general_config['object_class']
    weights = general_config['weights']
    assert dataset_source in ['nuscenes', 'waymo']

    # set up wandb logging
    wandb.init(project="SpOT-test", config=args)

    if "sweep" in out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.join(out_dir, wandb.run.name)
    if not os.path.exists(out_dir):  os.makedirs(out_dir)
    log_out = os.path.join(out_dir, 'test_log.txt')
    log(log_out, args)

    # data_root, object_class, dataloading_cfg, dataaug_cfg, split='train', load_baseline=False, debug=False
    val_dataset, val_ignored_dataset = load_dataset(data_root, dataset_source, object_class, trk_dataloading_config, data_aug_config, split='val', dataset_type="track-iter")
    num_workers = scheduling_config['data-loading']['num-workers']

    # load our model
    device = get_device()
    tpointnet_encoder = load_models(backbone_config, losses_config, dataset_source, object_class, weights, device, parallel_train=parallel)
    params = count_params(tpointnet_encoder)
    log(log_out, 'Num model params: ' + str(params))

    test(val_dataset, val_ignored_dataset, num_workers, tpointnet_encoder, device, log_out, object_class, parallel,
         max_pnts_in_batch, out_dir, dataset_source, general_config['evaluation'], trk_manage_config, trajectory_config, trk_association_config)


def test(dataset, ignored_dataset, num_workers, model, device, log_out, object_class, parallel, max_pnts_in_batch, out_dir,
         dataset_source, eval_config, trk_manage_config, trajectory_config, trk_association_config):
    model.eval()
    if dataset_source == "nuscenes":
        num_workers = min(num_workers, 16)
    else:
        num_workers = min(num_workers, 32)  # waymo has many more objects per frame

    persistent_workers = num_workers > 0
    loader = DataListLoader(dataset, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers) # note: keeping persistent workers is crucial for efficiency!
    ignored_loader = DataListLoader(ignored_dataset, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)

    # Run validation
    with torch.no_grad():
        if dataset_source == "nuscenes":
            nuscenes_version = eval_config['nuscenes-version']
            nuscenes_path = eval_config['nuscenes-path']
            evaluator = TrackAndEvaluateNuscenes(loader, ignored_loader, object_class, nuscenes_version, nuscenes_path, trk_manage_config, trajectory_config, trk_association_config, out_dir)
            _, tracking_submission = evaluator.track_and_evaluate(model, device, parallel, max_pnts_in_batch)
            official_evaluate_nuscenes(tracking_submission, nuscenes_version, nuscenes_path, object_class, out_dir)
        else:
            evaluator = TrackAndEvaluateWaymo(loader, ignored_loader, object_class, trk_manage_config, trajectory_config, trk_association_config, out_dir)
            _, tracking_submission = evaluator.track_and_evaluate(model, device, parallel, max_pnts_in_batch)

            track_executable_path = eval_config['waymo-tracking-eval-executable']
            waymo_gt_path = eval_config['waymo-gt-file']
            official_evaluate_waymo(tracking_submission, track_executable_path, waymo_gt_path, out_dir, object_class, log_out)



def official_evaluate_nuscenes(tracking_submission,
                              nuscenes_version,
                              nuscenes_path,
                              object_class,
                              out_dir):

    # run official NuScenes Tracking Evaluation
    track_results_outpath = os.path.join(out_dir, "tracking_results.json")
    with open(track_results_outpath, "w") as f:
        json.dump(tracking_submission, f)
    track_cfg = config_factory('tracking_nips_2019')

    if nuscenes_version != "v1.0-test":
        # otherwise pandas dumps a ton of FutureWarnings from nuScenes Devkit
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Run NuScenes Evaluation
        track_eval = TrackingEval(config=track_cfg, result_path=track_results_outpath, eval_set="val",
                                  output_dir=os.path.join(out_dir, "tracking"), nusc_version=nuscenes_version,
                                  nusc_dataroot=nuscenes_path, verbose=True, render_classes=[object_class])
        tracking_summary = track_eval.main(render_curves=True)
        amota = tracking_summary['label_metrics']['amota'][object_class]
        amotp = tracking_summary['label_metrics']['amotp'][object_class]
        wandb.log({"AMOTA": amota, "AMOTP": amotp})

        # evaluate and log MOTA values
        metric_details_path = os.path.join(out_dir, "tracking", "metrics_details.json")
        compute_motas(metric_details_path, os.path.join(out_dir, "tracking"))


def official_evaluate_waymo(tracking_submission,
                            track_executable_path,
                            gt_path,
                            out_dir,
                            object_class,
                            log_out):

    track_results_outpath = os.path.join(out_dir, "tracking_results.bin")
    with open(track_results_outpath, "wb") as f:
        f.write(tracking_submission.SerializeToString())

    # Run official waymo on tracking
    import subprocess
    track_ret_bytes = subprocess.check_output(f'{track_executable_path}' + ' ' +  f'{track_results_outpath}' + ' ' + f'{gt_path}', shell=True)
    track_ret_texts = track_ret_bytes.decode('utf-8')
    log(log_out, track_ret_texts)
    mota_dict = parse_waymo_tracking_resultstring(track_ret_texts, object_class)
    wandb.log(mota_dict)


if __name__=='__main__':
    from spot.io.config_utils import get_general_options
    config = get_general_options()
    main(config)