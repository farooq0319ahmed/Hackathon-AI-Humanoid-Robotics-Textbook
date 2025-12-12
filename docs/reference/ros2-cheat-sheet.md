---
sidebar_position: 1
---

# ROS 2 Cheat Sheet: Common Commands and Patterns

## Overview

This cheat sheet provides quick reference for the most commonly used ROS 2 commands, patterns, and best practices for humanoid robotics development. This is your go-to reference for ROS 2 development throughout the Physical AI & Humanoid Robotics Book.

## Core Commands

### System Commands

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Source workspace
source ~/humanoid_ws/install/setup.bash

# Check ROS 2 version
ros2 --version

# List available commands
ros2 --list-extensions
```

### Node Commands

```bash
# List active nodes
ros2 node list

# Get information about a specific node
ros2 node info <node_name>

# Echo node logs
ros2 launch <package> <launch_file>.py
```

### Topic Commands

```bash
# List all topics
ros2 topic list

# Get information about a topic
ros2 topic info /topic_name

# Echo topic data
ros2 topic echo /topic_name

# Echo topic with specific number of messages
ros2 topic echo /topic_name --field data -n 10

# Print topic type
ros2 topic type /topic_name

# Publish to a topic once
ros2 topic pub /topic_name std_msgs/String "data: 'Hello World'"

# Get topic statistics
ros2 topic hz /topic_name

# Get topic bandwidth
ros2 topic bw /topic_name
```

### Service Commands

```bash
# List all services
ros2 service list

# Get service type
ros2 service type <service_name>

# Call a service
ros2 service call /service_name std_srvs/srv/Empty

# Call with parameters
ros2 service call /set_parameters rcl_interfaces/srv/SetParameters "parameters: [{name: 'param_name', value: {string_value: 'value'}}]"
```

### Action Commands

```bash
# List all actions
ros2 action list

# Get action type
ros2 action type <action_name>

# Send action goal
ros2 action send_goal /action_name action_package/action/ActionName "goal_field: value"

# Send goal with feedback
ros2 action send_goal /action_name action_package/action/ActionName "goal_field: value" --feedback
```

### Parameter Commands

```bash
# List parameters of a node
ros2 param list <node_name>

# Get parameter value
ros2 param get <node_name> <param_name>

# Set parameter
ros2 param set <node_name> <param_name> <value>

# Load parameters from file
ros2 param load <node_name> params.yaml
```

## Package Management

### Creating Packages

```bash
# Create a new ROS 2 package (C++)
ros2 pkg create --build-type ament_cmake my_package

# Create a new ROS 2 package (Python)
ros2 pkg create --build-type ament_python my_package

# Create with dependencies
ros2 pkg create --build-type ament_python my_robot_controller --dependencies rclpy std_msgs geometry_msgs
```

### Building Packages

```bash
# Build all packages in workspace
colcon build

# Build specific package
colcon build --packages-select my_package

# Build with symlinks (faster rebuilds)
colcon build --symlink-install

# Build with specific packages and dependencies
colcon build --packages-up-to my_package

# Build and run tests
colcon build --packages-select my_package && colcon test --packages-select my_package
```

## Launch Files

### Basic Launch File (Python)

```python
# launch/my_launch_file.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node_name',
            parameters=[
                {'param1': 'value1'},
                os.path.join(get_package_share_directory('my_package'), 'config', 'params.yaml')
            ],
            remappings=[
                ('/original_topic', '/new_topic')
            ],
            output='screen'
        )
    ])
```

### Launch File Commands

```bash
# Launch a file
ros2 launch my_package my_launch_file.py

# Launch with arguments
ros2 launch my_package my_launch_file.py arg_name:=value

# Launch with verbose output
ros2 launch -d my_package my_launch_file.py
```

## Common Message Types

### Standard Messages

```bash
# Standard messages
std_msgs/msg/Bool
std_msgs/msg/Int32
std_msgs/msg/Float64
std_msgs/msg/String
std_msgs/msg/Header

# Geometry messages
geometry_msgs/msg/Twist
geometry_msgs/msg/Pose
geometry_msgs/msg/Point
geometry_msgs/msg/Quaternion
geometry_msgs/msg/PoseStamped
geometry_msgs/msg/TwistStamped

# Sensor messages
sensor_msgs/msg/JointState
sensor_msgs/msg/Image
sensor_msgs/msg/LaserScan
sensor_msgs/msg/PointCloud2
sensor_msgs/msg/Imu
sensor_msgs/msg/CameraInfo

# Navigation messages
nav_msgs/msg/Odometry
nav_msgs/msg/Path
nav_msgs/msg/MapMetaData
```

## Python Node Template

### Basic Publisher Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Basic Subscriber Node

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Server

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

```python
#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClient()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## URDF Commands and Patterns

### URDF Validation

```bash
# Check URDF file
check_urdf /path/to/robot.urdf

# Parse and display URDF info
urdf_to_graphiz /path/to/robot.urdf
```

### Common URDF Elements

```xml
<!-- Robot definition -->
<robot name="humanoid_robot">
  <!-- Links -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints -->
  <joint name="joint_name" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
  </joint>
</robot>
```

## Navigation Commands

### Navigation 2

```bash
# Launch navigation stack
ros2 launch nav2_bringup navigation_launch.py

# Launch with simulation
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Launch with RViz
ros2 launch nav2_bringup view_navigation_launch.py

# Send navigation goal programmatically
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{pose: {pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}, header: {frame_id: 'map'}}}"
```

## Control Commands

### ROS 2 Control

```bash
# List controllers
ros2 control list_controllers

# List controller types
ros2 control list_controller_types

# Load controller
ros2 control load_controller <controller_name>

# Configure controller
ros2 control configure_controller <controller_name>

# Switch controllers
ros2 control switch_controllers --activate <controller_name>

# Get controller state
ros2 control get_controller_state <controller_name>
```

## Debugging Commands

### Process Monitoring

```bash
# Monitor all ROS 2 processes
ps aux | grep ros

# Check network connections
netstat -tulpn | grep :11311

# Monitor system resources
htop
iotop
```

### Logging

```bash
# Set logging level
export RCUTILS_LOGGING_SEVERITY_THRESHOLD=DEBUG

# View logs
ros2 topic echo /rosout

# Check launch logs
tail -f ~/.ros/log/latest/*.log
```

## Common Launch Arguments

```python
# In launch file, declare arguments
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        # ... rest of launch description
    ])

# Launch with argument
ros2 launch my_package my_launch_file.py use_sim_time:=true
```

## TF2 Commands

```bash
# View TF tree
ros2 run tf2_tools view_frames

# Echo TF transforms
ros2 run tf2_ros tf2_echo <source_frame> <target_frame>

# Check TF connectivity
ros2 run tf2_ros buffer_server --ros-args -p publish_rate:=10.0
```

## Performance Commands

```bash
# Monitor topic frequency
ros2 topic hz /topic_name

# Monitor topic bandwidth
ros2 topic bw /topic_name

# Profile node performance
ros2 run tracetools_trace trace -a

# Monitor CPU usage per node
ros2 run top top
```

## Common Error Solutions

### Fix common issues:

```bash
# If ROS_DOMAIN_ID conflicts
export ROS_DOMAIN_ID=0

# If multicast issues occur
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# If permission errors with serial devices
sudo usermod -a -G dialout $USER

# Rebuild if packages don't update
rm -rf build install log
colcon build
```

## Best Practices

1. **Use meaningful names**: Choose descriptive names for nodes, topics, and parameters
2. **Parameter files**: Use YAML files for complex parameter configurations
3. **Launch files**: Group related nodes in launch files for easier management
4. **Error handling**: Always check for errors in your nodes
5. **Resource cleanup**: Properly destroy nodes and shutdown ROS 2
6. **Simulation time**: Use `use_sim_time` parameter for simulation compatibility
7. **Logging**: Use appropriate log levels (info, warn, error, debug)
8. **Dependencies**: Properly declare dependencies in package.xml

## Quick Reference for Humanoid Robotics

### Joint Control
```bash
# Joint state topic
/joint_states

# Command topics for position control
/joint_group_position_controller/commands

# Velocity control
/joint_group_velocity_controller/commands
```

### Navigation
```bash
# Navigation topics
/goal_pose
/initialpose
/tf
/tf_static
/costmap_topic
/local_plan
/global_plan
```

### Sensors
```bash
# Common sensor topics
/camera/image_raw
/scan
/imu/data
/odom
```

This cheat sheet should serve as your quick reference throughout the Physical AI & Humanoid Robotics Book. Bookmark this page for easy access to common ROS 2 commands and patterns you'll use in your humanoid robotics projects.