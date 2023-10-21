# hash:sha256:a9ca342f588e55347c58363226d257a5be9e3273e6ddb0ab0be78e7622cd3c66
FROM registry.codeocean.allenneuraldynamics.org/codeocean/miniconda3:4.9.2-cuda11.7.0-cudnn8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=7:4.2.7-0ubuntu0.1 \
        libsm6=2:1.2.3-1 \
        libxext6=2:1.3.4-0ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    deeplabcut[tf]==2.3.6 \
    moviepy==1.0.3
