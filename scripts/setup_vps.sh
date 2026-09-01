#!/bin/bash

# Update package list and install necessary packages
sudo apt update
sudo apt install -y \
    build-essential \
    docker.io \
    docker-compose \
    nvidia-driver-470 \
    nvidia-container-toolkit \
    python3-pip

# Enable and start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Install Python dependencies
pip3 install -r requirements.txt

# Add current user to the docker group
sudo usermod -aG docker $USER

# Print completion message
echo "VPS setup completed. Please log out and log back in for Docker group changes to take effect."