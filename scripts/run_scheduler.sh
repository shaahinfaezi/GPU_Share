#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f ../configs/.env ]; then
    export $(cat ../configs/.env | xargs)
fi

# Set default values for parameters if not provided
IMAGE=${IMAGE:-"llm-distilgpt2:cuda115"}
INSTANCES=${NUM_CONTAINERS:-4}
VRAM_SLICE_MB=${VRAM_SLICE_MB:-1500}
START_PORT=${START_PORT:-6000}

# Run the GPU scheduler
python3 ../src/gpu_scheduler.py --image "$IMAGE" --instances "$INSTANCES" --vram_slice_mb "$VRAM_SLICE_MB" --start_port "$START_PORT"