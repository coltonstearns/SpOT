import matplotlib
matplotlib.use('Agg')

import numpy as np
import wandb


def plot_val_metrics(val_metrics):
    for k, val in val_metrics.items():
        if k == 'tnocs_l1':
            spatial_l1s = np.linalg.norm(np.vstack(val), axis=1)
            data = spatial_l1s.reshape(-1, 1).tolist()
            table = wandb.Table(data=data, columns=["mean-l1-loss"])
            wandb.log({'Best Val NOCS L1 Distribution': wandb.plot.histogram(table, "mean-l1-loss")})

        elif k in ['yaw-l1', "scale-iou", "center-l1", 'proc-center-l1', 'proc-yaw-l1', 'proc-scale-1Diou', 'kabsch-center-l1', 'kabsch-yaw-l1', 'min-yaw-l1', 'min-center-l1', 'inputcrop-center-l1']:
            data = np.concatenate(val).reshape(-1, 1).tolist()
            table = wandb.Table(data=data, columns=[k])
            wandb.log({'Best Val %s Distribution' % k: wandb.plot.histogram(table, k)})

        elif k == 'segmentation_mIoU':
            continue
        else:
            continue


def log_runtime_stats(runtimes):
    hist_data = np.array(runtimes).reshape(-1, 1).tolist()
    table = wandb.Table(data=hist_data, columns=["Runtime"])
    wandb.log({'Runtime Distribution': wandb.plot.histogram(table, "Runtime")})
    print("=========Runtime Statistics=========")
    print("Mean Runtime: %s" % np.mean(runtimes))
    print("Median Runtime: %s" % np.median(runtimes))
    print("====================================")
    wandb.log({"Mean Runtime": np.mean(runtimes)})


def log(log_out, write_str):
    with open(log_out, 'a') as f:
        f.write(str(write_str) + '\n')
    print(write_str)


def print_batch_stats(log_out, epoch, cur_batch, num_batches, losses, metrics, type_id='TRAIN'):
    log(log_out, '-------------------------')
    log(log_out, '[Epoch %d: Batch %d/%d] %s' % (epoch, cur_batch, num_batches, type_id))
    for k, loss in losses.items():
        log(log_out, '             %s: %f' % (k, loss))
    for k, metric in metrics.items():
        log(log_out, '             %s: %f' % (k, metric))


def print_epoch_stats(log_out, epoch, losses, metrics, type_id='TRAIN'):
    log(log_out, '==================================================================')
    log(log_out, '[Epoch %d: Summary] %s' % (epoch, type_id))
    for k, loss in losses.items():
        log(log_out, '             %s: %f' % (k, loss))
    for k, metric in metrics.items():
        log(log_out, '             %s: %f' % (k, metric))
    log(log_out, '==================================================================')


def log_batch_stats(losses, metrics, epoch, batch_idx, num_batches, mode, log_out):
    losses_formatted, metrics_formatted = format_stats(losses, metrics, mode)
    print_batch_stats(log_out, epoch, batch_idx, num_batches, losses_formatted, metrics_formatted, mode)
    if mode == "train":
        wandb.log({**losses_formatted, **metrics_formatted})


def log_epoch_stats(losses, metrics, epoch, mode, log_out):
    losses_formatted, metrics_formatted = format_stats(losses, metrics, mode)
    wandb.log({**losses_formatted, **metrics_formatted})
    print_epoch_stats(log_out, epoch, losses_formatted, metrics_formatted, mode)
    wandb.log({**losses_formatted, **metrics_formatted})


def format_stats(losses, metrics, mode):
    losses_formatted = {}
    for k, loss in losses.items():
        if k in ['kabsch-reg']:
            continue
        loss_formatted = np.vstack(loss)
        losses_formatted["%s: loss %s" % (mode, k)] = np.mean(loss_formatted)

    metrics_formatted = format_metrics(metrics, mode)
    return losses_formatted, metrics_formatted


def format_metrics(metrics, mode):
    metrics_formatted = {}
    for k, metric in metrics.items():

        if k in ["tnocs_l1", "inputcrop-tnocs-l1"]:
            tnocs_l1_err = np.vstack(metric).reshape((-1, 4))  # will be ~(n_iters*B, 4)
            spatial_err = np.mean(np.linalg.norm(tnocs_l1_err[:, :3], axis=1))
            spatial_err_med = np.median(np.linalg.norm(tnocs_l1_err[:, :3], axis=1))
            temporal_err = np.mean(tnocs_l1_err[:, 3])
            metrics_formatted['%s: metric %s_spatial' % (mode, k)] = spatial_err
            # metrics_formatted['%s: metric %s_temporal' % (mode, k)] = temporal_err
            # metrics_formatted['%s: metric %s_spatial_median' % (mode, k)] = spatial_err_med

        elif k in ['class_predictions']:
            continue
        elif k in ['yaw-l1', 'kabsch-yaw-l1', 'scale-iou', 'inputcrop-yaw-l1', 'inputcrop-scale-iou', 'center-l1', 'kabsch-center-l1', 'inputcrop-center-l1', 'min-yaw-l1', 'min-center-l1', 'velocity-l1', 'inputcrop-velocity-l1']:
            try:
                metric_formatted = np.concatenate(metric)
            except:
                print(k)
                if len(metric) > 0:
                    print(metric[0].shape)
                metric_formatted = np.concatenate(metric)
            metric_formatted = metric_formatted[~np.any(np.isinf(metric_formatted))]
            # metrics_formatted["%s: metric %s median" % (mode, k)] = np.median(metric_formatted)
            metrics_formatted["%s: metric %s mean" % (mode, k)] = np.mean(metric_formatted)
        else:
            try:
                metric_formatted = np.vstack(metric)
            except:
                print(k)
                if len(metric) > 0:
                    print(metric[0].shape)
                metric_formatted = np.vstack([batch_metric.reshape(-1, *batch_metric.shape[-2:]) for batch_metric in metric])
            metric_formatted = metric_formatted[~np.any(np.isinf(metric_formatted), axis=1)]
            metrics_formatted["%s: metric %s" % (mode, k)] = np.mean(metric_formatted)
    return metrics_formatted




