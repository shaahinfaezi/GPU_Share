#!/bin/bash

set -e

# Load environment variables from .env file if it exists
if [ -f /app/configs/.env ]; then
    export $(cat /app/configs/.env | xargs)
fi

# Start NVIDIA MPS if not already running
if ! pgrep -x "nvidia-mps" > /dev/null; then
    /app/scripts/start_mps.sh
fi

# Run the GPU scheduler application
python3 /app/src/gpu_scheduler.py "$@"