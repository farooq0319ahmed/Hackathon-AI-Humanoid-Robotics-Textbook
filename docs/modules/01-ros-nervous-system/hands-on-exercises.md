---
sidebar_position: 5
---

# Hands-On Exercises: ROS 2 Fundamentals

## Exercise 1: Create Your First ROS 2 Node

### Goal
Create a simple publisher node that publishes "Hello, Robot!" messages to a topic and a subscriber node that receives and logs these messages.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Python 3.8+ environment set up
- Basic Python programming knowledge

### Steps

1. **Create a new ROS 2 package:**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ros2 pkg create --build-type ament_python hello_robot_pkg
   cd hello_robot_pkg
   ```

2. **Create the publisher script:**
   Create a file `hello_robot_pkg/hello_robot_pkg/publisher.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class HelloPublisher(Node):

       def __init__(self):
           super().__init__('hello_publisher')
           self.publisher_ = self.create_publisher(String, 'hello_topic', 10)
           timer_period = 1  # seconds
           self.timer = self.create_timer(timer_period, self.timer_callback)
           self.i = 0

       def timer_callback(self):
           msg = String()
           msg.data = f'Hello, Robot! Message #{self.i}'
           self.publisher_.publish(msg)
           self.get_logger().info(f'Publishing: "{msg.data}"')
           self.i += 1

   def main(args=None):
       rclpy.init(args=args)
       hello_publisher = HelloPublisher()
       rclpy.spin(hello_publisher)
       hello_publisher.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create the subscriber script:**
   Create a file `hello_robot_pkg/hello_robot_pkg/subscriber.py`:
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String

   class HelloSubscriber(Node):

       def __init__(self):
           super().__init__('hello_subscriber')
           self.subscription = self.create_subscription(
               String,
               'hello_topic',
               self.listener_callback,
               10)
           self.subscription  # prevent unused variable warning

       def listener_callback(self, msg):
           self.get_logger().info(f'I heard: "{msg.data}"')

   def main(args=None):
       rclpy.init(args=args)
       hello_subscriber = HelloSubscriber()
       rclpy.spin(hello_subscriber)
       hello_subscriber.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

4. **Update the setup.py file:**
   Modify `setup.py` to include entry points:
   ```python
   import os
   from glob import glob
   from setuptools import setup

   package_name = 'hello_robot_pkg'

   setup(
       name=package_name,
       version='0.0.0',
       packages=[package_name],
       data_files=[
           ('share/ament_index/resource_index/packages',
               ['resource/' + package_name]),
           ('share/' + package_name, ['package.xml']),
       ],
       install_requires=['setuptools'],
       zip_safe=True,
       maintainer='Your Name',
       maintainer_email='your.email@example.com',
       description='Simple ROS 2 publisher and subscriber example',
       license='Apache License 2.0',
       tests_require=['pytest'],
       entry_points={
           'console_scripts': [
               'publisher = hello_robot_pkg.publisher:main',
               'subscriber = hello_robot_pkg.subscriber:main',
           ],
       },
   )
   ```

5. **Build and run the nodes:**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select hello_robot_pkg
   source install/setup.bash

   # Terminal 1: Run the publisher
   ros2 run hello_robot_pkg publisher

   # Terminal 2: Run the subscriber (in a new terminal)
   ros2 run hello_robot_pkg subscriber
   ```

### Expected Outcome
- The publisher node should publish "Hello, Robot!" messages every second
- The subscriber node should receive and log these messages
- Both nodes should communicate over the 'hello_topic'

### Troubleshooting
- If nodes can't communicate, ensure both terminals have sourced the workspace
- Check that the topic names match exactly
- Use `ros2 topic list` to verify the topic exists

## Exercise 2: Publisher-Subscriber Communication with Custom Message

### Goal
Create a custom message type and use it in a publisher-subscriber pair to send robot position data.

### Steps

1. **Create a custom message:**
   Create directory `msg/` in your package and add `RobotPosition.msg`:
   ```
   float64 x
   float64 y
   float64 theta
   string robot_name
   ```

2. **Update package.xml to include message dependencies:**
   ```xml
   <depend>builtin_interfaces</depend>
   <depend>std_msgs</depend>
   <build_depend>rosidl_default_generators</build_depend>
   <exec_depend>rosidl_default_runtime</exec_depend>
   <member_of_group>rosidl_interface_packages</member_of_group>
   ```

3. **Update setup.py to include message generation:**
   ```python
   from setuptools import setup
   from glob import glob
   import os

   package_name = 'hello_robot_pkg'

   # Add this import
   from setuptools import find_packages

   setup(
       # ... other parameters ...
       data_files=[
           # ... other data files ...
           (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
       ],
       # ... rest of setup
   )
   ```

4. **Create publisher and subscriber using custom message**

### Expected Outcome
- Custom message type is properly defined and generated
- Publisher sends robot position data
- Subscriber receives and processes position data

## Exercise 3: URDF Robot Model Creation

### Goal
Create a simple URDF model of a wheeled robot with proper links, joints, and visual properties.

### Prerequisites
- ROS 2 with URDF packages installed
- Basic understanding of XML

### Steps

1. **Create a URDF file:**
   Create `hello_robot_pkg/urdf/simple_robot.urdf`:
   ```xml
   <?xml version="1.0"?>
   <robot name="simple_robot">
     <!-- Base link -->
     <link name="base_link">
       <visual>
         <geometry>
           <cylinder length="0.2" radius="0.2"/>
         </geometry>
         <material name="blue">
           <color rgba="0 0 1 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.2" radius="0.2"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="10"/>
         <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
       </inertial>
     </link>

     <!-- Left wheel -->
     <link name="left_wheel">
       <visual>
         <geometry>
           <cylinder length="0.05" radius="0.1"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.05" radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Right wheel -->
     <link name="right_wheel">
       <visual>
         <geometry>
           <cylinder length="0.05" radius="0.1"/>
         </geometry>
         <material name="black">
           <color rgba="0 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder length="0.05" radius="0.1"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="1"/>
         <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
       </inertial>
     </link>

     <!-- Joints -->
     <joint name="left_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="left_wheel"/>
       <origin xyz="0 0.2 -0.1" rpy="1.5707963267948966 0 0"/>
       <axis xyz="0 0 1"/>
     </joint>

     <joint name="right_wheel_joint" type="continuous">
       <parent link="base_link"/>
       <child link="right_wheel"/>
       <origin xyz="0 -0.2 -0.1" rpy="1.5707963267948966 0 0"/>
       <axis xyz="0 0 1"/>
     </joint>
   </robot>
   ```

2. **Visualize the robot:**
   ```bash
   # Install visualization tools if not already installed
   sudo apt install ros-humble-xacro ros-humble-joint-state-publisher-gui

   # Visualize the robot in RViz
   ros2 run rviz2 rviz2
   # In RViz, add a RobotModel display and set the description topic to your URDF
   ```

### Expected Outcome
- URDF file correctly defines a simple wheeled robot
- Robot model displays properly in RViz
- All links and joints are properly connected

### Validation Steps
- Verify the robot model appears correctly in RViz
- Check that TF tree shows proper relationships between links
- Ensure no warnings or errors in the URDF parser

## Exercise 4: Service Implementation

### Goal
Create a service that calculates the distance between two points in 2D space.

### Steps

1. **Create a service definition:**
   Create `srv/Distance.srv` in your package:
   ```
   float64 x1
   float64 y1
   float64 x2
   float64 y2
   ---
   float64 distance
   ```

2. **Implement service server and client**

3. **Test the service:**
   ```bash
   # Run the service server
   ros2 run hello_robot_pkg distance_server

   # In another terminal, call the service
   ros2 service call /calculate_distance hello_robot_pkg/srv/Distance "{x1: 0.0, y1: 0.0, x2: 3.0, y2: 4.0}"
   ```

### Expected Outcome
- Service correctly calculates Euclidean distance
- Client receives correct distance value (should be 5.0 for the example above)

## Summary

These hands-on exercises provide practical experience with the core concepts of ROS 2:
- Node creation and communication
- Topic-based publisher-subscriber patterns
- URDF robot modeling
- Service implementation

Completing these exercises will solidify your understanding of ROS 2 fundamentals and prepare you for more advanced topics in subsequent modules.