CONDA_PATH=$1
CONDA_ENV_NAME=$2

# set up conda env
source ${CONDA_PATH}/bin/activate
conda create -n ${CONDA_ENV_NAME} python=3.8 -y
source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME}

# install pytorch and other requirements
yes | pip install numpy
#conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
yes | pip install -r requirements.txt

# install MMCV as base-package for some CUDA kernels in third_party/
yes | pip install cython==0.29.33
yes | pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
yes | pip install mmdet==2.11.0

# install pytorch-geometric and pytorch3d
#conda install pyg==2.0.3 -c pyg -c conda-forge -y
yes | pip install torch-scatter==2.0.2 -f https://data.pyg.org/whl/torch-1.9.1+cu111
yes | pip install torch-cluster==1.6.3
yes | pip install torch-sparse==0.6.12
pip install torch-geometric==2.0.3 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
cd third_party
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout v0.6.2
pip install .
cd ../..
#conda install pytorch3d==0.6.2 -c pytorch3d -y
#pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

# compile PointNet++ CUDA kernels
cd third_party/pointnet2
CUDA_VISIBLE_DEVICES=0 python setup.py install
cd ../..

# compile general CUDA kernels
CUDA_VISIBLE_DEVICES=0 python setup.py develop
#yes | pip install -v -e .

# for waymo evaluation
# conda install -c anaconda libprotobuf -y
conda install protobuf==3.14.0 -y
pip install waymo-open-dataset-tf-2-3-0==1.3.1
yes | pip uninstall protobuf
yes | pip install protobuf==3.14.0 -y

# reinstall to older versions (or else NuScenes devkit will not work)
pip install numpy==1.19.2 --no-cache-dir

