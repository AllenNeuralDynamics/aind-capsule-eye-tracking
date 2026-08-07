# hash:sha256:cd7f588b6e49ff89013320fdc8cb5e7440991b5194ebb9f1b4976bfc2b9734de
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

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
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.2.3 \
    deeplabcut[tf]==2.3.6 \
    moviepy==1.0.3 \
    npc-session==0.1.32

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
