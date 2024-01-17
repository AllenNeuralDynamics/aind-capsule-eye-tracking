# hash:sha256:540bba9692d81cf7b2af9738fc42bbfbb968db4c8899b957e3b46cd92bc180f5
FROM registry.codeocean.allenneuraldynamics.org/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
        libcublas-12-0=12.0.2.224-1 \
        libcudnn8=8.8.1.3-1+cuda12.0 \
        libsm6=2:1.2.3-1build2 \
        libxext6=2:1.3.4-1build1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        ruamel.yaml==0.16.12 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    deeplabcut[tf]==2.3.6 \
    moviepy==1.0.3 \
    npc-session==0.1.32
