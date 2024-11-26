# hash:sha256:42e869cad49f9b23f18361559e395377cf17e3a8b267186af8020c3fe27e2225
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

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
    aind-data-schema==1.2.0 \
    aind-qcportal-schema==0.2.3 \
    deeplabcut[tf]==2.3.6 \
    moviepy==1.0.3 \
    npc-session==0.1.32

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
