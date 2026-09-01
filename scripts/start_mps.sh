#!/bin/bash

# Start the NVIDIA Multi-Process Service (MPS)
sudo systemctl start nvidia-mps

# Check the status of the MPS service
sudo systemctl status nvidia-mps