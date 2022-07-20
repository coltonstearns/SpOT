CONDA_PATH=$1
CONDA_ENV_NAME=$2

# set up conda env
source ${CONDA_PATH}/bin/activate
conda create -n ${CONDA_ENV_NAME} python=3.8 -y
source ${CONDA_PATH}/bin/activate ${CONDA_ENV_NAME}

# install pytorch and other requirements
yes | pip install numpy
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
yes | pip install -r requirements.txt

# install MMCV as base-package for some CUDA kernels in third_party/
yes | pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
yes | pip install mmdet==2.11.0

# install pytorch-geometric and pytorch3d
conda install pyg==2.0.3 -c pyg -c conda-forge -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d==0.6.2 -c pytorch3d -y

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

# reinstall to older versions (or else NuScenes devkit will not work)
pip install numpy==1.19.2 --no-cache-dir

