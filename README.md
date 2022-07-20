# SpOT: Spatiotemporal Modeling for 3D Object Tracking

This is the official implementation for the ECCV 2022 oral presentation [SpOT](https://arxiv.org/pdf/2207.05856.pdf).

![SpOT Teaser](media/teaser.png)


#### **CURRENTLY BEING UPDATED. PLEASE BE PATIENT.**

## Quick Installation

#### Python Environment
> Note: the code in this repo has been tested on Ubuntu 20.04 with Python 3.8, CUDA 11.1, and PyTorch 1.9.0. It may work for other setups, but has not been tested.

For quick installation, install [anaconda](https://www.anaconda.com/) and run `bash setup.sh ANACONDA_PATH ENV_NAME`. For example,
`bash setup.sh /home/colton/anaconda3 spot_env`. For step-by-step installation, see below.

Note that we use we [Weights and Biases](https://wandb.ai/) for visualizations and run metrics. In order to access any program outputs, please create a Weights and Biases account.

#### Waymo Open Dataset Evaluation
The Waymo Open Dataset uses a separate evaluation pipeline written that must be compiled with [Bazel](https://bazel.build/). In order to evaluate on Waymo, clone the [waymo toolkit](https://bazel.build/) into `third_party/` and compile all Bazel scripts as instructed in their toolkit. Make sure that the compiled tracking-evaluation script is at:
* `./third_party/waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_tracking_metrics_main`


## Downloads
#### Preprocessed Datasets
We provide preprocessed versions of the nuScenes and Waymo datasets. Note that these datasets are quite large. To download them run:
* `cd data`
* `bash download_data.sh`


#### Pretrained Weights
We provide downloads for pretrained models of SpOT. To download them, visit this [link](https://drive.google.com/drive/folders/1S7r7vR43GzDVRLXnK9RVleV2YbMMzDKp?usp=sharing).

#### nuScenes Dataset
> Note: the nuScenes dataset must be downloaded for any nuScenes tracking evaluation

Please refer to the official [nuScenes website](https://www.nuscenes.org/). To verify reported results in the paper, download the **full dataset (v1.0)** for both **trainval** and **test** splits. To additionally run our preprocessing, also download the **nuScenes-lidarseg** annotations. Simlink the dataset folder to `./data/nuScenes`.

#### Waymo Open Dataset
> Note: the Waymo Open dataset only needs to be downloaded if you do NOT wish to use our preprocessed format (i.e. you want to preprocess the dataset from scratch).

Please refer to the official [Waymo website](https://waymo.com/open/data/perception/). To verify reported results in the paper, download the **Perception Dataset v1.2**. Follow the structure of the GCP bucket, i.e. the base folder contains `training`, `validation`, and `testing`, each with many `.tfrecord` files. Simlink the dataset folder to `./data/Waymo`.

## Evaluation with Pretrained Models
The [`spot/test.py`] script is used to run the evaluations on a trained SpOT model. Here are examples running various evaluations with the provided pretrained network weights.

#### nuScenes Dataset
To reproduce nuScenes results on the `cars` class run:
```
python spot/test.py --config=configs/reported_eval/nusc_reported_car.yaml --general.out=./out/nusc_car
```

To reproduce nuScenes results on the `pedestrian` class run:
```
python spot/test.py --config=configs/reported_eval/nusc_reported_pedestrian.yaml --general.out=../out/nusc_ped
```

#### Waymo Dataset
To reproduce nuScenes results on the `vehicles` class run:
```
python spot/test.py --config=configs/reported_eval/waymo_reported_vehicle.yaml --general.out=../out/waymo_vehicle
```

To reproduce nuScenes results on the `vehicles` class run:
```
python spot/test.py --config=configs/reported_eval/waymo_reported_pedestrian.yaml --general.out=../out/waymo_ped
```


## Training
Coming soon...

## Installation Step by Step
Coming soon...

## Citation
If you found this code or paper useful, please consider citing:
```
@inproceedings{stearns2022spot,
	author={Stearns, Colton and Rempe, Davis and Li, Jie and Ambrus, Rares and Guizilini, Vitor and Zakharov, Sergey and Yang, Yanchao and Guibas, Leonidas J.},
	title={SpOT: Spatiotemporal Modeling for 3D Object Tracking},
	booktitle={European Conference on Computer Vision (ECCV)},
	year={2022}
}
```

## Questions?
If you run into any problems or have questions, please create an issue or contact Colton (`coltongs@stanford.edu`).