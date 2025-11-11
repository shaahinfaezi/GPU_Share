# GPU Scheduler

## Overview
The GPU Scheduler is a Python application designed to manage GPU resources efficiently by launching and monitoring Docker containers that utilize NVIDIA GPUs. This application is particularly useful for running machine learning models in a multi-tenant environment.

## Features
- Dynamic allocation of GPU resources based on available memory.
- Launching and managing multiple Docker containers.
- Monitoring of GPU usage and container performance.
- Integration with NVIDIA Multi-Process Service (MPS) for improved resource management.

## Prerequisites
- NVIDIA GPUs (e.g., 3070 Ti) with the latest drivers installed.
- Docker and Docker Compose installed on the VPS.
- Python 3.7 or higher.

## Directory Structure
```
gpu-scheduler
├── src
│   ├── gpu_scheduler.py        # Main logic for the GPU scheduler
│   └── __init__.py             # Package initialization
├── configs
│   ├── .env.example             # Example environment variable configuration
│   ├── scheduler.example.yaml    # Example YAML configuration for the scheduler
│   └── tenants.example.json      # Example JSON configuration for tenant settings
├── docker
│   ├── Dockerfile.scheduler      # Docker image definition for the GPU scheduler
│   └── entrypoint.sh            # Entrypoint script for the Docker container
├── scripts
│   ├── setup_vps.sh             # Script to set up the VPS environment
│   ├── start_mps.sh             # Script to start NVIDIA MPS
│   ├── run_scheduler.sh          # Script to run the GPU scheduler
│   └── teardown.sh               # Script to clean up the environment
├── tests
│   └── test_smoke.py            # Smoke tests for the GPU scheduler
├── .gitignore                    # Files and directories to ignore by Git
├── docker-compose.yml            # Docker Compose configuration
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gpu-scheduler.git
   cd gpu-scheduler
   ```

2. Copy the example environment configuration:
   ```
   cp configs/.env.example .env
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Build the Docker image:
   ```
   docker-compose build
   ```

## Usage
1. Start the NVIDIA MPS service:
   ```
   ./scripts/start_mps.sh
   ```

2. Run the GPU scheduler:
   ```
   ./scripts/run_scheduler.sh
   ```

3. Monitor the logs and performance metrics.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.