# Base CUDA image compatible with 3080 Ti and CUDA 11.5
FROM nvidia/cuda:11.5.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python + PyTorch (CPU/GPU minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ca-certificates && \
    pip3 install --no-cache-dir torch==1.13.1+cu115 torchvision==0.14.1+cu115 \
        --extra-index-url https://download.pytorch.org/whl/cu115 && \
    pip3 install --no-cache-dir flask transformers==4.33.2 pynvml && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 5000
ENV NVIDIA_VISIBLE_DEVICES=all
CMD ["python3", "server.py"]
