
import json
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

"""
This script Renders provided AMOTA curves, normalizing for one of the AMOTAS.
"""

SAMPLED_RECALLS = {
    'car': [0.90, 0.85, 0.8],
    'pedestrian': [0.92, 0.87, 0.8],
    'bus': [0.89, 0.85, 0.8],
    'motorcycle': [0.87, 0.8, 0.75],
    'trailer': [0.71, 0.65, 0.6],
    'truck': [0.85, 0.80, 0.72],
    'bicycle': [0.85, 0.7, 0.55]
}
NUSC_CLASSES = SAMPLED_RECALLS.keys()
METRIC_KEYS = ['ids', 'fp', 'fn', 'mota']

def load_info(filepath):
    mota_metrics = {}
    with open(filepath, "r") as f:
        data = json.load(f)
        for cls in NUSC_CLASSES:
            recalls = data[cls]['recall']
            cls_eval_recalls = np.array(SAMPLED_RECALLS[cls])
            mota_metrics[cls] = {'recall': cls_eval_recalls.tolist()}
            for k in METRIC_KEYS:
                metric_per_recthresh = data[cls][k]

                interp_metric = np.interp(cls_eval_recalls, np.array(recalls)[::-1], np.array(metric_per_recthresh)[::-1])
                if k in ['ids', 'fp', 'fn']:
                    interp_metric /= data[cls]['gt'][-1]
                mota_metrics[cls][k] = interp_metric.tolist()

    return mota_metrics

def compute_motas(metric_details_path, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    mota_vals = load_info(metric_details_path)
    with open(os.path.join(outdir, "mota_samples.json"), "w") as f:
        json.dump(mota_vals, f)
    print("MOTA info:")
    print(mota_vals)


def render_amota_comparisons(savepath, path1, path2, path3, name1='SpOT', name2='SpOT-No-SSR', name3='CenterPoint', offset=5):
    """
    This function Renders provided AMOTA curves, normalizing for one of the AMOTAS.

    Eg.
    path1 = "/home/colton/Documents/nuscenes-ped-visuals/ours/tracking/metrics_details.json"
    path2 = "/home/colton/Documents/nuscenes-ped-visuals/nms-baseline/tracking/metrics_details.json"
    path3 = "/home/colton/Documents/nuscenes-ped-visuals/cp-baseline/metrics_details.json"
    """


    def load_recalls_and_motas(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            mota = data['pedestrian']['mota']
            recalls = data['pedestrian']['recall']
        return recalls, mota

    recalls1, motas1 = load_recalls_and_motas(path1)
    recalls2, motas2 = load_recalls_and_motas(path2)
    recalls3, motas3 = load_recalls_and_motas(path3)

    # put everything into the same reference frame
    motas2 = np.interp(np.array(recalls1)[::-1], np.array(recalls2)[::-1], np.array(motas2)[::-1])[::-1]
    recalls2 = np.interp(np.array(recalls1)[::-1], np.array(recalls2)[::-1], np.array(recalls2)[::-1])[::-1]
    motas3 = np.interp(np.array(recalls1)[::-1], np.array(recalls3)[::-1], np.array(motas3)[::-1])[::-1]
    recalls3 = np.interp(np.array(recalls1)[::-1], np.array(recalls3)[::-1], np.array(recalls3)[::-1])[::-1]

    # take difference from CP
    motas1 -= motas3
    motas2 -= motas3
    motas3 -= motas3
    motas1 *= 100
    motas2 *= 100

    data = pd.DataFrame({name1: motas1[offset:], name2: motas2[offset:], name3: motas3[offset:],
                        'Recall': recalls1[offset:]})

    axis1 = sns.lineplot(data=data, x='Recall', y=name1, palette="bright")
    axis2 = sns.lineplot(data=data, x='Recall', y=name2, palette="bright")
    axis3 = sns.lineplot(data=data, x='Recall', y=name3, palette="bright")
    axis1.fill_between(data['Recall'], data['SpOT'], data['SpOT-No-SSR'], color="blue", alpha=0.1)
    axis2.fill_between(data['Recall'], data['SpOT-No-SSR'], data['CenterPoint'], color="orange", alpha=0.1)
    axis1.set_ylabel("MOTA Diff.")
    axis1.set_title("nuScenes Pedestrian Tracking")
    plt.legend(labels=['SpOT', 'SpOT-No-SSR'])
    plt.savefig(savepath)
