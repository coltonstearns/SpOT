conda create -n waymo-preprocess-2 python=3.7 -y
source /home/colton/anaconda3/bin/activate waymo-preprocess-2
echo $CONDA_PREFIX

# start with tensorflow
conda install -c anaconda libprotobuf -y
pip install waymo-open-dataset-tf-2-3-0==1.3.1

# install pytorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# install select necessary requirements
yes | pip install -r requirements.txt


# install mmcv
yes | pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
conda install -c conda-forge pycocotools -y
yes | pip install mmdet==2.11.0


conda install pyg -c pyg -c conda-forge -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y

CUDA_VISIBLE_DEVICES=0 python setup.py develop
