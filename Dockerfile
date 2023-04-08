ARG PYTORCH="1.12.1"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel as python-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM python-base as initial
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update && apt-get install -y curl git build-essential cmake ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG OPENMPI_VERSION="4.1.3"
ARG OPENMPI_CONFIGURE_OPTIONS="--enable-orterun-prefix-by-default --with-sge"

# Download, build, and install OPENMPI
RUN mkdir /tmp/openmpi-src \
    && cd /tmp/openmpi-src \
    && wget https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION%.*}/openmpi-${OPENMPI_VERSION}.tar.gz \
    && tar xfz openmpi-${OPENMPI_VERSION}.tar.gz \
    && cd openmpi-${OPENMPI_VERSION} \
    && ./configure ${OPENMPI_CONFIGURE_OPTIONS} \
    && make -j$(nproc) all \
    && make install \
    && ldconfig \
    && cd \
    && rm -rf /tmp/openmpi-src

WORKDIR /workspace

FROM initial as development

# Install MMCV
RUN conda clean --all
ENV MMCV_WITH_OPS="1"
ENV FORCE_CUDA="1"
RUN pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt --no-cache-dir
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/jiafatom/apex.git@01802f623c9b54199566871b49f94b2d07c3f047