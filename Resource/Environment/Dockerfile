FROM nvidia/cuda:10.0-base-ubuntu18.04

# Disable interactive configuration.
ENV DEBIAN_FRONTEND=noninteractive

# Set proxies
# ENV http_proxy http://A.B.C.D.E:FG/
# ENV https_proxy http://A.B.C.D.E:FG/

# Apt-get installation
RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       cuda-command-line-tools-10-0 \
       cuda-cublas-10-0 \
       cuda-cufft-10-0 \
       cuda-curand-10-0 \
       cuda-cusolver-10-0 \
       cuda-cusparse-10-0 \
       graphviz \
       libcudnn7=7.6.0.64-1+cuda10.0 \
       libfreetype6-dev \
       libhdf5-serial-dev \
       libsm6 \
       libzmq3-dev \
       pkg-config \
       python3-tk \
       software-properties-common \
       unzip \
       vim \
       && \
   apt-get clean && \
   rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda-10.0/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}

ARG USE_PYTHON_3_NOT_2=True
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
   ${PYTHON} \
   ${PYTHON}-pip

RUN ${PIP} install --upgrade \
   pip \
   setuptools

RUN ${PIP} install \
    cython \
    flask \
    flask_restful \
    graphviz \
    idx2numpy \
    imutils \
    ipython \
    joblib \
    jupyterlab \
    matplotlib \
    metaflow \
    numpy \
    pandas \
    Pillow \
    pydot \
    pytest \
    requests \
    scipy \
    torch==1.6.0 \
    torchvision==0.7.0 \
    torchviz \
    tqdm

# RUN mkdir "/mnist_classifier"
# RUN 
