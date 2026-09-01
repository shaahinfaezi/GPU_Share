#!/bin/bash

# Stop and remove all running containers related to the GPU scheduler
docker ps -q --filter "name=llm_tenant" | xargs -r docker stop
docker ps -aq --filter "name=llm_tenant" | xargs -r docker rm

# Optionally remove any associated Docker images
# docker rmi -f llm-distilgpt2:cuda115

# Clean up any unused Docker resources
docker system prune -f

echo "Teardown complete. All related containers have been stopped and removed."