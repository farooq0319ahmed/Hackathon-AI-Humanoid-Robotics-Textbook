# Quickstart Guide: Physical AI & Humanoid Robotics Book

**Date**: 2025-12-12
**Feature**: 001-humanoid-robotics-book
**Status**: Complete

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or compatible Linux distribution
- **Hardware**:
  - Workstation: 8+ cores CPU, 32GB+ RAM, NVIDIA GPU with CUDA support (RTX 3070 or equivalent)
  - Alternative: NVIDIA Jetson Orin AGX/NX for edge deployment
- **Software**: Git, Python 3.8+, Docker, Docker Compose

### Required Accounts
- NVIDIA Developer Account (for Isaac Sim/ROS packages)
- GitHub Account (for repository access)

## Setup Environment

### 1. Clone the Repository
```bash
git clone <repository-url>
cd humanoid-robotics-book
```

### 2. Install ROS 2 (Humble Hawksbill)
```bash
# Add ROS 2 repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop ros-humble-ros-base
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo rosdep init
rosdep update

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. Install Gazebo
```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### 4. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install Python dependencies
pip install rclpy transforms3d numpy matplotlib openai
```

### 5. Install Additional Tools
```bash
# Install Docusaurus for book development
npm install -g @docusaurus/cli

# Install Unity Hub (optional, for Unity simulation)
# Download from https://unity.com/download
```

## Running the Examples

### 1. Basic ROS 2 Node Example
```bash
# Navigate to ROS 2 examples directory
cd examples/ros2_basics

# Build the workspace
colcon build --packages-select my_robot_controller

# Source the workspace
source install/setup.bash

# Run the example
ros2 run my_robot_controller publisher_node
```

### 2. URDF Robot Model Loading
```bash
# Launch Gazebo with a humanoid robot model
ros2 launch my_robot_gazebo humanoid_world.launch.py
```

### 3. Simulation Environment
```bash
# Start the simulation environment
ros2 launch my_robot_gazebo navigation_world.launch.py

# In another terminal, send navigation commands
ros2 run nav2_example_scripts simple_navigation.py
```

## Book Module Structure

### Module 1: The Robotic Nervous System (ROS 2)
- **Duration**: 4-6 hours
- **Focus**: ROS 2 architecture, nodes, topics, services, actions
- **Hands-on**: Create custom ROS 2 packages, implement publisher-subscriber patterns
- **Outcome**: Understanding of robotic communication systems

### Module 2: The Digital Twin (Gazebo & Unity)
- **Duration**: 6-8 hours
- **Focus**: Physics simulation, sensor simulation, visualization
- **Hands-on**: Set up simulation environments, configure sensors
- **Outcome**: Ability to test robot behaviors in virtual environments

### Module 3: The AI-Robot Brain (NVIDIA Isaac)
- **Duration**: 8-10 hours
- **Focus**: Perception, navigation, control pipelines
- **Hands-on**: Implement VSLAM, configure Nav2 for humanoid robots
- **Outcome**: Understanding of AI-driven robot capabilities

### Module 4: Vision-Language-Action (VLA) & Capstone
- **Duration**: 10-12 hours
- **Focus**: Multi-modal interaction, cognitive planning
- **Hands-on**: Voice-to-action implementation, end-to-end project
- **Outcome**: Complete autonomous humanoid system

## Validation Commands

### 1. Verify ROS 2 Installation
```bash
ros2 topic list
# Should show system topics like /parameter_events, /rosout
```

### 2. Test Basic Publisher-Subscriber
```bash
# Terminal 1: Start publisher
ros2 run demo_nodes_cpp talker

# Terminal 2: Start subscriber
ros2 run demo_nodes_py listener
# Should see messages passing between nodes
```

### 3. Check Gazebo Installation
```bash
gazebo --version
# Should show Gazebo version information
```

### 4. Validate Simulation
```bash
# Launch simple simulation
ros2 launch gazebo_ros empty_world.launch.py
# Should open Gazebo with an empty world
```

## Troubleshooting

### Common Issues

1. **ROS 2 Commands Not Found**
   - Ensure you've sourced the ROS 2 installation: `source /opt/ros/humble/setup.bash`

2. **Gazebo Not Launching**
   - Check NVIDIA GPU drivers are properly installed
   - Verify X11 forwarding if running in Docker

3. **Python Import Errors**
   - Activate your Python virtual environment: `source venv/bin/activate`

4. **Simulation Performance Issues**
   - Ensure sufficient RAM (32GB+ recommended)
   - Check GPU drivers and CUDA installation

### Getting Help
- Check the reference materials in the book for detailed troubleshooting
- Visit the ROS 2 documentation: https://docs.ros.org/
- Join the ROS community forums for support