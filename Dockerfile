# Dockerfile - minimal, uses CUDA runtime (no compiling)
FROM nvidia/cuda:11.5.2-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools curl ca-certificates && \
    pip3 install --no-cache-dir flask numpy pynvml requests && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 5000
ENV NVIDIA_VISIBLE_DEVICES=all
CMD ["python3", "server.py"]
