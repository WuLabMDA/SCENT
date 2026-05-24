FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV NUMBA_CACHE_DIR=/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgeos-dev \
    libvips-tools \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libatlas-base-dev \
    gfortran \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    sudo \
    curl \
    wget \
    htop \
    vim \
    ca-certificates \
    python3-openslide \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip setuptools wheel && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN pip install \
    gpustat==0.6.0 \
    setuptools==61.2.0 \
    pytz==2021.3 \
    termcolor==1.1.0 \
    joblib==1.2.0 \
    tqdm==4.64.1 \
    docopt==0.6.2

RUN pip install \
    openslide-python==1.2.0 \
    opencv-python==4.7.0.68 \
    scikit-image==0.18.0 \
    Pillow==9.5.0 \
    imgaug==0.4.0 \
    mahotas==1.4.12 \
    pandas==2.0.2 \
    openpyxl==3.1.2

RUN pip install \
    deepdish==0.3.6 \
    numpy==1.24.1 \
    scipy==1.10.1 \
    matplotlib==3.7.1 \
    seaborn==0.12.1 \
    statsmodels==0.14.0 \
    scikit-learn==1.2.2 \
    xgboost==1.7.6 \
    gdown==4.7.0 \
    lmdb==1.4.0

RUN pip install \
    tensorboard==2.12.3 \
    einops==0.6.0 \
    albumentations==1.3.0 \
    itk==5.3.0 \
    SimpleITK==2.2.0 \
    nibabel==5.0.1 \
    pynrrd==1.0.0 \
    pyradiomics==3.1.0 \
    trimesh==3.22.5

RUN pip install \
    pycox==0.2.2 \
    lifelines==0.27.6 \
    scikit-survival==0.22.0

RUN pip install \
    torch==2.0.0+cu117 \
    torchvision==0.15.1+cu117 \
    --index-url https://download.pytorch.org/whl/cu117

RUN apt-get update && \
    apt-get remove -y python3-blinker && \
    rm -rf /var/lib/apt/lists/*

RUN pip install \
    torchinfo==1.7.2 \
    lightning==2.1.0 \
    transformers==4.30.2 \
    pytorch-ignite==0.4.12 \
    mlflow==2.4.2 \
    lungmask==0.2.15

RUN pip install monai==1.3.0

RUN pip install \
    ipython==8.12.1 \
    jupyterlab==3.6.1 \
    notebook==6.5.2 \
    jedi==0.18.0 \
    chardet==5.0.0 \
    cchardet==2.1.6 \
    ipywidgets==8.0.6 \
    glances==3.4.0.3 \
    traitlets==5.9.0

WORKDIR /.dgl
RUN chmod 777 /.dgl

WORKDIR /.local
RUN chmod 777 /.local

WORKDIR /.cache
RUN chmod -R 777 /.cache

WORKDIR /Data
RUN chmod 777 /Data

WORKDIR /Code
RUN chmod 777 /Code

WORKDIR /App

CMD ["/bin/bash"]
