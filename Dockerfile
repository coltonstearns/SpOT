
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV PROJECT=spot
ENV PYTORCH_VERSION=1.9.0
ENV TORCHVISION_VERSION=0.10.0
ENV CUDNN_VERSION=8.9.5.30-1+cuda11.7
ENV NCCL_VERSION=2.12.12-1+cuda11.7
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG python=3.8
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
#    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn8=8.0.5.39-1+cuda11.1\
    libcudnn8-dev=8.0.5.39-1+cuda11.1\
    libnccl2=2.8.4-1+cuda11.1 \
    libnccl-dev=2.8.4-1+cuda11.1 \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg

# Install Open MPI
#RUN mkdir /tmp/openmpi && \
#    cd /tmp/openmpi && \
#    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
#    tar zxf openmpi-4.0.0.tar.gz && \
#    cd openmpi-4.0.0 && \
#    ./configure --enable-orterun-prefix-by-default && \
#    make -j $(nproc) all && \
#    make install && \
#    ldconfig && \
#    rm -rf /tmp/openmpi
#
## Install OpenSSH for MPI to communicate between containers
#RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
#    mkdir -p /var/run/sshd
#
## Allow OpenSSH to talk to containers without asking for confirmation
#RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
#    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
#    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install Pydata and other deps
RUN pip install cython

# Install PyTorch
RUN pip install torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} && ldconfig
ENV TORCH_CUDA_ARCH_LIST=Volta;Turing;Kepler+Tesla

# Override DGP wandb with required version
RUN pip install wandb==0.8.21 pyquaternion xarray diskcache tenacity pycocotools

# create project workspace dir
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/${PROJECT}
WORKDIR /workspace/${PROJECT}

# Copy project source last (to avoid cache busting)
WORKDIR /workspace/${PROJECT}
COPY . /workspace/${PROJECT}
#COPY /mnt/ssd2/nuscenes-centerpoint /workspace/${PROJECT}/data/nuscenes-centerpoint-processed
ENV PYTHONPATH="/workspace/${PROJECT}:$PYTHONPATH"

# ===========================================================================
# install pytorch and other requirements
RUN yes | pip install numpy
#conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN yes | pip install -r requirements.txt

# install MMCV as base-package for some CUDA kernels in third_party/
RUN yes | pip install cython==0.29.33
RUN yes | pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN yes | pip install mmdet==2.11.0

# ==========================================================================
# Everything after this has to be done manually!
# install pytorch-geometric and pytorch3d
RUN pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
RUN pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
RUN pip install torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.9.1+cu111.html
RUN pip install torch-geometric==2.0.3

# Install Pytorch 3D
WORKDIR /workspace
RUN git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd /workspace/pytorch3d && \
    git checkout v0.6.2 &&\
    pip install .

# compile PointNet++ CUDA kernels
WORKDIR /workspace/${PROJECT}/third_party/pointnet2
RUN pip install .

# compile general CUDA kernels
WORKDIR /workspace/${PROJECT}
#RUN python setup.py develop

# for waymo evaluation
# conda install -c anaconda libprotobuf -y
RUN pip install protobuf==3.14.0
RUN pip install waymo-open-dataset-tf-2-3-0==1.3.1
RUN yes | pip uninstall protobuf
RUN yes | pip install protobuf==3.14.0

# reinstall to older versions (or else NuScenes devkit will not work)
RUN #pip install numpy==1.19.2 --no-cache-dir


