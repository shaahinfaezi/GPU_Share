ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.5.1
ARG TORCH_CUDA=cu121
ARG DEEPSPEED_VERSION=0.16.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash build-essential ca-certificates curl git libaio-dev libaio1 \
      ninja-build python3 python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/${TORCH_CUDA} \
      torch==${TORCH_VERSION}+${TORCH_CUDA} && \
    DS_BUILD_AIO=1 python3 -m pip install deepspeed==${DEEPSPEED_VERSION}

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install -r /app/requirements.txt
COPY app/model.py /app/model.py
COPY config/zero-nvme.json /app/zero-nvme.json

EXPOSE 8080
ENTRYPOINT ["python3", "/app/model.py"]
