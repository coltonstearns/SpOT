'''

This script can be used to train the CaSPR model. Use:

python train.py --help

'''
import os
import gc

import numpy as np

import torch
import torch.optim as optim
from spot.io.wandb_utils import sample_and_viz_bboxes_and_nocs

from spot.ops.torch_utils import get_device, count_params
from spot.utils.train_utils import run_one_epoch
from spot.io.logging_utils import log, plot_val_metrics
from spot.io.run_utils import load_models

from spot.data.sequences_dataset import load_dataset
import wandb
from torch_geometric.loader import DataListLoader
from spot.test import test

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# from multiprocessing import set_start_method
# set_start_method("spawn")

os.environ["WANDB_MODE"] = "online"
# np.seterr(all='raise')


def train(args):
    torch.autograd.set_detect_anomaly(True)

    # get each configuration dictionary
    backbone_config = args["backbone"]
    dataloading_config = args['training_dataloading']
    trk_dataloading_config = args["testing_dataloading"]
    data_aug_config = args["data_augmentation"]
    losses_config = args["losses"]
    scheduling_config = args["scheduling"]
    general_config = args["general"]
    trajectory_config = args["track_motion_model"]
    trk_association_config = args["track_association"]
    trk_manage_config = args["track_management"]
    assert trk_dataloading_config["dataset-properties"]["tracking-mode"] == True

    # Get general arguments
    dataset_source = general_config['dataset_name']
    assert dataset_source in ['nuscenes', 'waymo']
    data_root = general_config['data_root']
    out_dir = general_config['out']
    parallel = general_config['parallel']
    batch_size = general_config['batch_size']
    lr = general_config['lr']
    object_class = general_config['object_class']
    weights = general_config['weights']
    max_pnts_per_batch = general_config['max_pnts_per_batch']
    if isinstance(weights, dict) and "sweep" in out_dir:
        weights = weights[object_class]
        # weights = weights[args.dataloading_config][args.backbone_config]
    general_config['weights'] = weights

    only_validate = general_config['only_val']
    if only_validate:
        num_epochs = 1
    else:
        num_epochs = scheduling_config['optimizer']['epochs']

    if "only_viz" in general_config:
        only_viz = general_config['only_viz']
    else:
        only_viz = False

    # set up wandb logging
    wandb.init(project="SpOT-train", config=args)

    # prepare output
    if "sweep" in out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_dir = os.path.join(out_dir, wandb.run.id)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    log_out = os.path.join(out_dir, 'train_log.txt')
    log(log_out, args)

    # instantiate pytorch-geometric dataloaders
    num_workers = scheduling_config['data-loading']['num-workers']

    # build train and validation datasets
    # data_root, object_class, dataloading_cfg, dataaug_cfg, split='train', load_baseline=False, debug=False
    train_dataset, _ = load_dataset(data_root, dataset_source, object_class, dataloading_config, data_aug_config, split='train', batch_size=batch_size, shuffle=True, drop_last=True, dataset_type="default")
    val_dataset, _ = load_dataset(data_root, dataset_source, object_class, dataloading_config, data_aug_config, split='val', batch_size=batch_size, shuffle=True, drop_last=False, dataset_type="default")
    track_dataset, track_ignored_dataset = load_dataset(data_root, dataset_source, object_class, trk_dataloading_config, data_aug_config, split='val', dataset_type="track-iter")


    # Load model_weights
    if parallel:
        log(log_out, 'Attempting to use all available GPUs for parallel training...')
    device = get_device()
    tpointnet_encoder = load_models(backbone_config, losses_config, dataset_source, object_class, weights, device, parallel_train=parallel)

    if scheduling_config["optimizer"]["type"] == "adam":
        optimizer = optim.Adam(list(tpointnet_encoder.parameters()), lr=lr, betas=scheduling_config['optimizer']['betas'], eps=scheduling_config['optimizer']['eps'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduling_config['optimizer']['decay'])
    else:
        raise RuntimeError("Scheduler must be adam or adamw.")

    # Log out info
    params = count_params(tpointnet_encoder)
    log(log_out, 'Num model params: ' + str(params))

    # Begin training loop
    persistent_workers = True if num_workers > 0 else 0
    best_val_loss = np.inf
    val_every, track_val_every, save_every, print_every = scheduling_config['logging']['val-every'], scheduling_config['logging']['track-every'], scheduling_config['logging']['save-every'], scheduling_config['logging']['print-every']
    for epoch in range(num_epochs):
        # train
        if not (only_validate or only_viz):
            train_dataset.reset()
            train_loader = DataListLoader(train_dataset, num_workers=num_workers, pin_memory=True,  persistent_workers=persistent_workers, timeout=60, worker_init_fn=lambda _: np.random.seed())  # get around numpy RNG seed bug
            succeeded, iteration = False, 0
            while not succeeded:
                succeeded, iteration, _, _ = run_one_epoch(tpointnet_encoder, train_loader, device, optimizer, epoch, backbone_config,
                                                log_out, mode='train', max_pnts_per_batch=max_pnts_per_batch, print_stats_every=print_every,
                                                           autoregressive_training=scheduling_config['other']['train-autoregressive'], dataset=train_dataset, num_workers=num_workers)
            scheduler.step()
            del train_loader
            gc.collect()

        # validate
        if (epoch % val_every == 0 or only_validate) and (not only_viz):
            # run validation metrics
            val_dataset.reset()
            val_loader = DataListLoader(val_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=persistent_workers)  # note: keeping persistent workers is crucial for efficiency!
            with torch.no_grad(): # must do this to avoid running out of memory
                succeeded = False
                while not succeeded:
                    succeeded, _, val_loss, val_metrics = run_one_epoch(tpointnet_encoder, val_loader, device, None,
                                                epoch, backbone_config, log_out, mode='val', max_pnts_per_batch=max_pnts_per_batch, print_stats_every=print_every,
                                                                autoregressive_training=False, dataset=val_dataset, num_workers=num_workers)
                if not np.isfinite(val_loss):
                    continue
                if val_loss < best_val_loss:
                    log(log_out, 'BEST Val loss so far! Saving checkpoint...')
                    save_name = 'BEST_time_model.pth'
                    save_file = os.path.join(out_dir, save_name)
                    model_state = {'tpointnet_state': tpointnet_encoder.state_dict()}
                    torch.save(model_state, save_file)
                    plot_val_metrics(val_metrics)
            del val_loader
            gc.collect()

        if epoch % val_every == 0 or only_validate or only_viz:
            # randomly select NOCS objects to visualize
            val_dataset.reset()
            viz_loader = DataListLoader(val_dataset, num_workers=num_workers, pin_memory=True, timeout=60, persistent_workers=persistent_workers)
            with torch.no_grad(): # must do this to avoid running out of memory
                sample_and_viz_bboxes_and_nocs(tpointnet_encoder, viz_loader, device,max_pnts_per_batch=max_pnts_per_batch, nsamples=5, parallel=parallel, outdir=out_dir)
            del viz_loader
            gc.collect()
            if only_viz:
                return

        # tracking validation
        if (epoch+1) % track_val_every == 0:
            track_dataset.reset()
            track_ignored_dataset.reset()
            with torch.no_grad():
                if parallel:
                    tpointnet_single_machine = tpointnet_encoder.module
                    num_devices = torch.cuda.device_count()
                    single_dev_pnts_per_batch = max_pnts_per_batch // num_devices
                else:
                    tpointnet_single_machine = tpointnet_encoder
                    single_dev_pnts_per_batch = max_pnts_per_batch

            test(track_dataset, track_ignored_dataset, num_workers, tpointnet_single_machine, device, log_out,
                 object_class, False, single_dev_pnts_per_batch, out_dir, dataset_source, general_config['evaluation'],
                 trk_manage_config, trajectory_config, trk_association_config)
            gc.collect()

        if epoch % save_every == 0 or (epoch+1) % track_val_every == 0:
            # save model parameters
            save_name = 'time_model_%d.pth' % (epoch)
            save_file = os.path.join(out_dir, save_name)
            model_state = {'tpointnet_state': tpointnet_encoder.state_dict()}
            torch.save(model_state, save_file)


if __name__=='__main__':
    from spot.io.config_utils import get_general_options
    config = get_general_options()
    train(config)