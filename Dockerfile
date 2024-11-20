#FROM nvcr.io/nvidia/pytorch:20.06-py3
#FROM continuumio/miniconda3
#FROM gpuci/miniconda-cuda:11.0-devel-ubuntu20.04
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

#COPY environment.yml .
#RUN conda update --force-reinstall conda
#RUN conda env update --name base --file environment.yml --prune

ENV MY_ROOT=/workspace \
    PKG_PATH=/yolo_src \
    NUMPROC=4 \
    PYTHON_VER=3.8 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=. \
    DEBIAN_FRONTEND=noninteractive 

WORKDIR $PKG_PATH

RUN apt-get update && apt-get install -y apt-utils && apt-get -y upgrade && \
    apt-get install -y git libsnappy-dev libopencv-dev libhdf5-serial-dev libboost-all-dev libatlas-base-dev \
        libgflags-dev libgoogle-glog-dev liblmdb-dev curl unzip\
        python${PYTHON_VER}-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VER} get-pip.py && \
    rm get-pip.py && \
    # Clean UP
    apt upgrade -y && \
    apt clean && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*  # cleanup to reduce image size

RUN ln -s /usr/bin/python${PYTHON_VER} /usr/bin/python

RUN apt install unzip
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR $MY_ROOT
# We have to install mish-cuda from source due to an issue with one of the header files
ADD https://github.com/thomasbrandon/mish-cuda/archive/master.zip $MY_ROOT/mish-cuda.zip
RUN unzip mish-cuda.zip
WORKDIR $MY_ROOT/mish-cuda-master
RUN cp external/CUDAApplyUtils.cuh csrc/
RUN python setup.py build install
WORKDIR $PKG_PATH
# ADD https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing /weights/yolov4-csp.weights
ADD requirements.txt $PKG_PATH/requirements.txt
RUN pip install -r $PKG_PATH/requirements.txt
ADD yolo $PKG_PATH/yolo
ADD train.py $PKG_PATH/train.py
ADD test.py $PKG_PATH/test.py
ADD setup.py $PKG_PATH/setup.py
ADD data $PKG_PATH/data
RUN pip install .

