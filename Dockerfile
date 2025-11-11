# Smallest possible CUDA-enabled base (no compiler)
FROM nvidia/cuda:11.5.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python + minimal libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools python3-dev curl ca-certificates && \
    pip3 install --no-cache-dir flask pycuda pynvml requests && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 5000
ENV NVIDIA_VISIBLE_DEVICES=all
CMD ["python3", "server.py"]
