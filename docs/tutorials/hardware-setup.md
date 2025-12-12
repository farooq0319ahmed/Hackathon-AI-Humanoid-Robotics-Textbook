---
sidebar_position: 2
---

# Hardware Setup Guide: Workstation and Jetson Configuration

## Overview

This guide provides detailed instructions for setting up both workstation and Jetson platforms for humanoid robotics development. The NVIDIA Jetson platform is essential for edge deployment of AI models, while the workstation provides the development environment and simulation capabilities needed for testing and training.

## Workstation Setup

### System Requirements

- **CPU**: Intel i7 or AMD Ryzen 7 with 8+ cores (16+ recommended)
- **RAM**: 32GB minimum, 64GB+ recommended
- **GPU**: NVIDIA RTX 3070/3080/3090 or RTX 4070/4080/4090 (RTX 4090 recommended for training)
- **Storage**: 1TB+ SSD for models and datasets
- **OS**: Ubuntu 22.04 LTS

### NVIDIA GPU Setup

#### 1. Install NVIDIA Drivers

```bash
# Remove existing drivers
sudo apt remove --autoremove nvidia-* nvidia-*

# Add graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install latest NVIDIA driver (check nvidia.com for latest version)
sudo apt install nvidia-driver-535  # Or latest available version
sudo reboot
```

#### 2. Verify GPU Installation

```bash
# Check GPU detection
nvidia-smi

# Should show your NVIDIA GPU with driver version and CUDA version
```

#### 3. Install CUDA Toolkit

```bash
# Download CUDA toolkit (for Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run

# Make installer executable
chmod +x cuda_12.3.0_545.23.06_linux.run

# Run installer (uncheck driver installation if driver is already installed)
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add CUDA to PATH in ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Install cuDNN (if needed for deep learning)

```bash
# Download cuDNN from NVIDIA Developer website
# Extract and install:
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.X-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Workstation Optimization

#### 1. Swap Configuration for Large Models

```bash
# Create larger swap file for training large models
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 2. System Performance Tuning

```bash
# Install performance tools
sudo apt install htop nvidia-utils-535

# Configure power management for performance
sudo nvidia-smi -ac 5000,1590  # Set max GPU clock (adjust for your GPU)

# Configure CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Jetson Platform Setup

### Jetson AGX Orin Setup

#### 1. Initial Setup

The NVIDIA Jetson AGX Orin is the recommended platform for edge AI in humanoid robotics:

- **SoC**: 12-core NVIDIA ARM v8.2 64-bit CPU
- **GPU**: 2048-core NVIDIA Ampere architecture GPU
- **Memory**: 32GB 256-bit LPDDR5 ECC
- **Storage**: 64GB eMMC 5.1

#### 2. Flash Jetson OS

```bash
# Using NVIDIA SDK Manager (recommended)
# 1. Download SDK Manager from NVIDIA Developer website
# 2. Connect Jetson to host computer via USB-C
# 3. Put Jetson in recovery mode (hold RECOVERY button, press RESET)
# 4. Run SDK Manager to flash OS

# Alternative: Use jetson-flash tool
pip3 install jetson-flash
jetson-flash --device agx-orin-devkit --image /path/to/jetpack/image
```

#### 3. Post-Installation Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Jetson-specific packages
sudo apt install -y nvidia-jetpack nvidia-jetpack-cuda

# Verify Jetson status
sudo /usr/bin/jetson_clocks.sh  # Enable max clocks
jetson_release -v  # Check Jetson info
```

#### 4. Jetson Performance Mode

```bash
# Set to max performance mode
sudo nvpmodel -m 0  # Maximum performance mode
sudo jetson_clocks  # Enable max clocks permanently
```

### Jetson Container Setup

#### 1. Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
sudo apt install -y docker.io
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 2. Pull Isaac ROS Containers

```bash
# Pull Isaac ROS containers
docker pull nvcr.io/nvidia/isaac_ros/isaac_ros_dev:latest
docker pull nvcr.io/nvidia/isaac_ros/isaac_ros_visual_slam:latest
docker pull nvcr.io/nvidia/isaac_ros/isaac_ros_detectnet:latest
```

## Network Configuration

### 1. Robot-Workstation Communication

For reliable communication between workstation and robot:

```bash
# On workstation, create network bridge for robot communication
sudo apt install bridge-utils

# Create bridge interface
sudo brctl addbr br0
sudo ip addr add 192.168.100.1/24 dev br0
sudo ip link set br0 up

# Configure firewall for robot communication
sudo ufw allow from 192.168.100.0/24
```

### 2. Time Synchronization

```bash
# Install NTP for time synchronization
sudo apt install chrony

# Configure chrony to sync with robot
echo "server 192.168.100.10 iburst minpoll 0 maxpoll 4" | sudo tee -a /etc/chrony/chrony.conf
sudo systemctl restart chronyd
```

## Sensor Integration Setup

### 1. Camera Configuration

```bash
# Install camera support
sudo apt install v4l-utils

# Test camera
v4l2-ctl --list-devices
ffmpeg -f v4l2 -list_formats all -i /dev/video0
```

### 2. LiDAR Setup (Example: Velodyne VLP-16)

```bash
# Install LiDAR drivers
sudo apt install ros-humble-velodyne-*

# Configure network for LiDAR
sudo ip addr add 10.0.0.1/24 dev eth0
```

### 3. IMU Integration

```bash
# Install IMU packages
sudo apt install ros-humble-imu-tools

# Test IMU
roslaunch imu_tools imu_test.launch
```

## Power Management

### Workstation Power Profile

```bash
# Install power management tools
sudo apt install powertop tlp

# Configure for performance
sudo tlp start
echo 'CPU_SCALING_GOVERNOR_ON_AC=performance' | sudo tee -a /etc/default/tlp
echo 'CPU_SCALING_GOVERNOR_ON_BAT=performance' | sudo tee -a /etc/default/tlp
```

### Jetson Power Management

```bash
# Monitor Jetson power consumption
sudo tegrastats  # Real-time power monitoring

# Set power mode based on workload
# Mode 0: Max performance
# Mode 1: Balanced
# Mode 2: Low power
sudo nvpmodel -m 0  # Set to max performance
```

## Storage Optimization

### Workstation SSD Setup

```bash
# Create separate mount for datasets and models
sudo mkfs.ext4 /dev/nvme1n1  # Replace with your SSD
sudo mkdir /datasets
sudo mount /dev/nvme1n1 /datasets
echo '/dev/nvme1n1 /datasets ext4 defaults 0 0' | sudo tee -a /etc/fstab
```

### Jetson Storage Management

```bash
# Monitor storage usage
df -h

# Create swap on external storage if needed
sudo fallocate -l 8G /external/swapfile
sudo mkswap /external/swapfile
sudo swapon /external/swapfile
```

## Testing Hardware Setup

### 1. GPU Compute Test

```bash
# Test CUDA installation
nvidia-smi -q -d COMPUTE

# Run simple CUDA test
cat << EOF > test_cuda.cu
#include <stdio.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device 0: %s\n", prop.name);
    }
    return 0;
}
EOF

nvcc test_cuda.cu -o test_cuda
./test_cuda
```

### 2. Jetson Performance Test

```bash
# Run Jetson performance benchmark
sudo /opt/jetson-benchmark/run_benchmarks.sh

# Check thermal performance
sudo tegrastats -t 30  # Monitor for 30 seconds
```

### 3. Network Communication Test

```bash
# Test network throughput to robot
iperf3 -s &  # On robot
iperf3 -c <robot-ip>  # On workstation
```

## Troubleshooting

### Common Hardware Issues

1. **GPU Not Detected**
   - Check if NVIDIA drivers are properly installed: `nvidia-smi`
   - Reinstall drivers if needed
   - Check for kernel compatibility issues

2. **Jetson Thermal Throttling**
   - Ensure adequate cooling is provided
   - Check thermal paste on heatsink
   - Monitor with `sudo tegrastats`

3. **Camera Not Working**
   - Check camera permissions: `sudo usermod -a -G video $USER`
   - Verify camera is properly connected
   - Test with `v4l2-ctl --list-devices`

4. **LiDAR Network Issues**
   - Verify static IP configuration
   - Check firewall settings
   - Test with `ping <lidar-ip>`

5. **Memory Issues During Training**
   - Increase swap space
   - Reduce batch sizes in training
   - Monitor memory usage with `htop`

### Performance Optimization

- Use SSD storage for datasets and models
- Ensure adequate cooling for sustained performance
- Configure power settings for maximum performance
- Use appropriate CUDA compute capabilities for your GPU

## Next Steps

With your hardware properly configured, you're now ready to:
1. Install ROS 2 and the Isaac ROS ecosystem
2. Set up your development environment
3. Begin working through the modules in the Physical AI & Humanoid Robotics Book

The workstation setup enables development, simulation, and training of AI models, while the Jetson platform provides the edge computing power needed for real-time AI inference on the humanoid robot. Proper hardware configuration is essential for the complex AI and real-time processing requirements of humanoid robotics.