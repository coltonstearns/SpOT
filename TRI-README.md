# CaSPR: Learning Canonical Spatiotemporal Point Cloud Representations

This is a fork from the official implementation for the NeurIPS 2020 spotlight paper. The original project website can be found at [project webpage](https://geometry.stanford.edu/projects/caspr/).
This fork acts as a sandbox environment for testing + modifying the CaSPR object model on object tracks in self-driving datasets (currently only NuScenes is supported).

![CaSPR Teaser](caspr.png)


# TRI Environment Setup
I have not yet made a dockerfile for this repo. This section has instructions for setting up an appropriate virtual environment.

Remove mmdet voxelization code in ops to make worth with python 3.8

First set up and activate a virtual environment with Python 3.6 and install some initial dependencies, e.g. using conda:
* `conda create -n caspr_env python=3.6`
* `conda activate caspr_env`

Next install PyTorch 1.7.0 and Torchvision 0.8.0; for TRI-Ouroboros P3 instance with CUDA 10.1 use the following (for other combinations see [this page](https://pytorch.org/get-started/previous-versions/)), then requirements file:
* `pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
* `pip install -r requirements.txt`

Next install [tk3dv](https://github.com/drsrinathsridhar/tk3dv), which is used for visualization, and the Neural ODE library [torchdiffeq](https://github.com/rtqichen/torchdiffeq):
* `pip install git+https://github.com/drsrinathsridhar/tk3dv.git`
* `pip install torchdiffeq==0.0.1` (note the version is important)

Next we must build and install [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) which is used for PointNet++. However,
Kaolin has a bug for PyTorch > 1.4; therefore, we need to copy a bug-fix from the `setup` folder:
* `mkdir external`
* `cd external`
* `git clone https://github.com/NVIDIAGameWorks/kaolin.git`
* `cd kaolin`
* `git checkout v0.1`
* `cp ../setup/kaolin_fix.py ./kaolin/datasets/base.py`
* `python setup.py build_ext --inplace`
* `python setup.py install`
* Verify in python with `import kaolin as kal` then `print(kal.__version__)`. The error "No module names nuscenes" can be ignored.

Next, install MMDetection library:
* `pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/101/torch1.7.1/index.html`
* `pip install mmdet==2.11.0`

Now, we also need to install pytorch geometric and pytorch3d:
conda install pyg -c pyg -c conda-forge

Finally, compile all cython scripts
* `python setup.py develop`


# SAIL V100 Environment
Because SAIL V100s use CUDA 9.2 (sneeky, but have to check /usr/local/cuda/bin/nvcc), we have to install
`conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=9.2 -c pytorch`
`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu92/torch1.7.0/index.html`
`pip install mmdet==2.11.0`
Also, needs to be installed with
`pip install -v -e .`

modify AT_CHECK to be TORCH_CHECK
# Docker workflow

Run:

```bash
make docker-build
make docker-start-interactive
```

At this point you should be in a terminal inside the docker container. To test that kaolin was built properly run `import kaolin as kal` then `print(kal.__version__)`. The error "No module names nuscenes" can be ignored.

# Running on Existing Data
In it's current development stage, running the Caspr pipeline is done entirely through the `train.py` script. To only run on the validation set, use `--only-val` flag. Also due to being in-development,
there are multiple config files and keyword arguments that need to be passed in. 

An example run command on NuScenes data is:
```
python train.py --data-cfg ../data/configs/nuscenes.cfg --out ../train_out --batch-size 18 --model-config ../data/configs/train_hyperparams.json --dataset-source nuscenes --num-pts 256 --cluttered --seq-len 20 --val-every 2 --save-every 2 --num-attention-heads 1 --group-norm
```
For complete definition of the flags, refer to the signatures in `--help`. However, note that for non-shapenet data, the `--cluttered` flag must be provided.
Also, `--data-cfg` is a text file that provides any data-loader-specific parameters, and `--model-config` is a json file that
specifies model losses to use, as well as whether to use PointNet++ backbone or the simple 3D per-frame baseline.

### NuScenes Data
Caspr cannot take the raw NuScenes dataset as input. Instead, it must first be preprocessed in the `nuscenes_preprocessing` 
directory, or can be downloaded from TRI-s3 at `s3://tri-ml-datasets/scratch/colton.stearns/tri-nuscenes-processed-2.0`. Follow the 
readme in `nuscenes_preprocessing` section for more information.


# Running on New Data
Currently, there is no code supporting a streamlined way to incorporate new datasets into the Caspr Repo. To 
work with new data, one must (1) create a new pytorch dataset loader in `./caspr/data/` and (2) modify `train.py` and `utils/config_utils.py`
to incorporate the new pytorch dataset.

Regarding the pytorch dataset, the `.__getitem__(idx)` call must output a dictionary with the following fields:
```
output = {'in_point_sequence': torch.Tensor (size=(T_in,N,4) dtype=float32),
          'out_nocs_sequence': torch.Tensor (size=(T_out,N,4) dtype=float32),
          'out_points_mask': torch.Tensor (size=(T_out,N) dtype=float32),
          'out_clean_nocs_sequence': torch.Tensor (size=(T_out,N') dtype=float32),
          'out_frames_mask': torch.Tensor (size=(T_in,) dtype=bool),
          'out_bboxes': torch.Tensor (size=(T_out, 3+3+1+12+3), dtype=float32),
          'out_nocs_bboxes': torch.Tensor(),
          'out_velocities': torch.Tensor(),
          'out_velocities_mask': torch.Tensor(),
          'object_class': torch.Tensor()}
```
Furthermore, the dataloader must follow the initialization convention of `./caspr/data/nuscenes_dataset`, and load different 
data splits based on the `split=['train', 'val', 'test']` parameter.

### T_in, T_out, and N
* `T_in` - the length of the input sequence. If we choose this to be 20 sweeps of the NuScenes dataset, then this is `T_in = 20`.
* `T_out` - the number of labeled frames in the sequence; because Nuscenes is labeled every 10 sweeps, if our sequence length is 20 sweeps, `T_out = 2.`
* `N` - the number of points we sample per frame. For NuScenes we use `N=256`. If there are not N points for a certain frame, we the tensor contains repeated points.

### Output Values
* `in_point_sequence` - the input point cloud sequence to Caspr. Should be mean-centered at the first frame.
* `out_nocs_sequence` - the entire input point cloud sequence transformed into the NOCS coordinate frame. That is, `out_nocs_sequence = s * R @ in_point_sequence + t`, given NOCS scale, rotation, and translation.
* `out_points_mask` - a binary mask indicating if a point belongs to the object or belongs to background
* `out_clean_nocs_sequence` - `out_nocs_sequence` filtered with `out_points_mask`. Thus, every point is in the NOCS reference frame AND belongs to the object. Points are repeated to match the correct tensor size.
* `out_frames_mask` - a binary mask indicating if an input frame has ground truth label.
* `out_bboxes` - the ground truth bounding boxes for each labeled frame. The bounding box is represented as `[center, dxdydz_residual, yaw_residual, yaw-onehot-bin, dxdydz-onehot-bin]`, creating a 
  22-dimensional vector per box. Yaw bins are 30 degree splits from 0 to 360. Size bins, in meters, are `[[4.8, 1.8, 1.5], [10.0, 2.6, 3.2], [2.0, 1.0, 1.6]]`
  for cars and `[[0.9, 0.9, 1.7]]` for pedestrians.
  
The last four values are empty tensors as they currently are not supported.