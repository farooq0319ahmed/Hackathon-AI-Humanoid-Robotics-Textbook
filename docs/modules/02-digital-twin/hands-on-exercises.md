---
sidebar_position: 6
---

# Hands-On Exercises: Robot Simulation and Environment Modeling

## Exercise 1: Gazebo Environment Setup and Configuration

### Goal
Set up a basic Gazebo environment with a humanoid robot model and configure physics properties for realistic simulation.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Gazebo (Garden or Harmonic) installed
- Basic understanding of URDF/SDF
- Completed Module 1: ROS 2 Fundamentals

### Requirements
- Ubuntu 22.04 LTS
- Computer with OpenGL 3.3+ capable graphics
- 8GB+ RAM recommended

### Steps

1. **Create a Gazebo world file:**
   Create `~/ros2_ws/src/my_robot_simulation/worlds/basic_environment.world`:

   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="basic_environment">
       <!-- Include default models -->
       <include>
         <uri>model://ground_plane</uri>
       </include>

       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Add a simple house model -->
       <model name="house">
         <pose>-3 0 0 0 0 0</pose>
         <link name="house_body">
           <collision name="collision">
             <geometry>
               <box>
                 <size>6 4 3</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>6 4 3</size>
               </box>
             </geometry>
             <material>
               <ambient>0.8 0.8 0.8 1</ambient>
               <diffuse>0.8 0.8 0.8 1</diffuse>
               <specular>0.1 0.1 0.1 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <!-- Add a cylindrical obstacle -->
       <model name="cylinder_obstacle">
         <pose>2 2 0.5 0 0 0</pose>
         <link name="cylinder_link">
           <collision name="collision">
             <geometry>
               <cylinder>
                 <radius>0.5</radius>
                 <length>1.0</length>
               </cylinder>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <cylinder>
                 <radius>0.5</radius>
                 <length>1.0</length>
               </cylinder>
             </geometry>
             <material>
               <ambient>0.5 0.5 1.0 1</ambient>
               <diffuse>0.5 0.5 1.0 1</diffuse>
             </material>
           </visual>
           <inertial>
             <mass>10.0</mass>
             <inertia>
               <ixx>0.625</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.625</iyy>
               <iyz>0</iyz>
               <izz>1.25</izz>
             </inertia>
           </inertial>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. **Launch Gazebo with your world:**
   ```bash
   gazebo ~/ros2_ws/src/my_robot_simulation/worlds/basic_environment.world
   ```

3. **Configure physics parameters:**
   Add physics configuration to your world file:

   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_factor>1</real_time_factor>
     <real_time_update_rate>1000</real_time_update_rate>
     <gravity>0 0 -9.8</gravity>
   </physics>
   ```

4. **Test physics behavior:**
   - Use Gazebo's GUI to spawn objects and observe physics interactions
   - Verify gravity is working correctly
   - Check collision detection between objects

### Expected Outcome
- Gazebo environment loads successfully with custom objects
- Objects respond correctly to gravity and collisions
- Physics simulation runs smoothly at real-time speed

### Validation Steps
- Verify all objects are visible and positioned correctly
- Test that objects fall under gravity
- Confirm collision detection works between different object types

## Exercise 2: Sensor Integration in Gazebo

### Goal
Integrate LiDAR, camera, and IMU sensors into a robot model and verify their functionality in Gazebo.

### Prerequisites
- Gazebo environment from Exercise 1
- Basic understanding of ROS 2 topics and messages

### Steps

1. **Create a robot URDF with sensors:**
   Create `~/ros2_ws/src/my_robot_simulation/urdf/sensor_robot.urdf`:

   ```xml
   <?xml version="1.0"?>
   <robot name="sensor_robot">
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
         <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
       </inertial>
     </link>

     <!-- LiDAR sensor -->
     <link name="lidar_link">
       <visual>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
         <material name="red">
           <color rgba="1 0 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <cylinder radius="0.05" length="0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.1"/>
         <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
       </inertial>
     </link>

     <joint name="lidar_joint" type="fixed">
       <parent link="base_link"/>
       <child link="lidar_link"/>
       <origin xyz="0 0 0.2" rpy="0 0 0"/>
     </joint>

     <!-- Camera sensor -->
     <link name="camera_link">
       <visual>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
         <material name="green">
           <color rgba="0 1 0 1"/>
         </material>
       </visual>
       <collision>
         <geometry>
           <box size="0.05 0.05 0.05"/>
         </geometry>
       </collision>
       <inertial>
         <mass value="0.05"/>
         <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
       </inertial>
     </link>

     <joint name="camera_joint" type="fixed">
       <parent link="base_link"/>
       <child link="camera_link"/>
       <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
     </joint>

     <!-- IMU sensor -->
     <link name="imu_link">
       <inertial>
         <mass value="0.01"/>
         <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
       </inertial>
     </link>

     <joint name="imu_joint" type="fixed">
       <parent link="base_link"/>
       <child link="imu_link"/>
       <origin xyz="0 0 0" rpy="0 0 0"/>
     </joint>

     <!-- Gazebo plugins for sensors -->
     <gazebo reference="lidar_link">
       <sensor type="ray" name="lidar_sensor">
         <pose>0 0 0 0 0 0</pose>
         <visualize>true</visualize>
         <update_rate>10</update_rate>
         <ray>
           <scan>
             <horizontal>
               <samples>360</samples>
               <resolution>1</resolution>
               <min_angle>-3.14159</min_angle>
               <max_angle>3.14159</max_angle>
             </horizontal>
           </scan>
           <range>
             <min>0.1</min>
             <max>30.0</max>
             <resolution>0.01</resolution>
           </range>
         </ray>
         <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
           <ros>
             <remapping>~/out:=scan</remapping>
           </ros>
           <output_type>sensor_msgs/LaserScan</output_type>
         </plugin>
       </sensor>
     </gazebo>

     <gazebo reference="camera_link">
       <sensor type="camera" name="camera_sensor">
         <update_rate>30</update_rate>
         <camera name="head_camera">
           <horizontal_fov>1.3962634</horizontal_fov>
           <image>
             <width>640</width>
             <height>480</height>
             <format>R8G8B8</format>
           </image>
           <clip>
             <near>0.1</near>
             <far>100</far>
           </clip>
         </camera>
         <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
           <frame_name>camera_link</frame_name>
         </plugin>
       </sensor>
     </gazebo>

     <gazebo reference="imu_link">
       <sensor name="imu_sensor" type="imu">
         <always_on>true</always_on>
         <update_rate>100</update_rate>
         <visualize>false</visualize>
         <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
           <ros>
             <remapping>~/out:=imu</remapping>
           </ros>
           <frame_name>imu_link</frame_name>
         </plugin>
       </sensor>
     </gazebo>
   </robot>
   ```

2. **Create a launch file for the robot in Gazebo:**
   Create `~/ros2_ws/src/my_robot_simulation/launch/sensor_robot.launch.py`:

   ```python
   import os
   from launch import LaunchDescription
   from launch.actions import DeclareLaunchArgument
   from launch.substitutions import LaunchConfiguration
   from launch_ros.actions import Node
   from launch.substitutions import PathJoinSubstitution
   from launch_ros.substitutions import FindPackageShare

   def generate_launch_description():
       # Declare launch arguments
       world_arg = DeclareLaunchArgument(
           'world',
           default_value='basic_environment.world',
           description='Choose one of the world files from `/my_robot_simulation/worlds`'
       )

       # Get URDF via xacro
       robot_description_content = open(
           os.path.join(
               FindPackageShare('my_robot_simulation').find('my_robot_simulation'),
               'urdf',
               'sensor_robot.urdf'
           )
       ).read()

       # Robot state publisher
       robot_state_publisher_node = Node(
           package='robot_state_publisher',
           executable='robot_state_publisher',
           name='robot_state_publisher',
           parameters=[{'robot_description': robot_description_content}]
       )

       # Spawn entity
       spawn_entity = Node(
           package='gazebo_ros',
           executable='spawn_entity.py',
           arguments=[
               '-topic', 'robot_description',
               '-entity', 'sensor_robot',
               '-x', '0',
               '-y', '0',
               '-z', '0.5'
           ],
           output='screen'
       )

       # Gazebo server
       gzserver = Node(
           package='gazebo_ros',
           executable='gzserver',
           arguments=[
               PathJoinSubstitution([
                   FindPackageShare('my_robot_simulation'),
                   'worlds',
                   LaunchConfiguration('world')
               ]),
               '-s', 'libgazebo_ros_init.so',
               '-s', 'libgazebo_ros_factory.so'
           ],
           output='screen'
       )

       # Gazebo client
       gzclient = Node(
           package='gazebo_ros',
           executable='gzclient',
           output='screen'
       )

       return LaunchDescription([
           world_arg,
           gzserver,
           gzclient,
           robot_state_publisher_node,
           spawn_entity
       ])
   ```

3. **Launch the simulation:**
   ```bash
   # Build your workspace
   cd ~/ros2_ws
   colcon build --packages-select my_robot_simulation
   source install/setup.bash

   # Launch the simulation
   ros2 launch my_robot_simulation sensor_robot.launch.py
   ```

4. **Test sensor functionality:**
   ```bash
   # Check if sensor topics are published
   ros2 topic list | grep -E "(scan|image|imu)"

   # Listen to LiDAR data
   ros2 topic echo /scan

   # Listen to camera data
   ros2 topic echo /camera/image_raw

   # Listen to IMU data
   ros2 topic echo /imu
   ```

### Expected Outcome
- Robot model spawns in Gazebo with sensors
- All sensors publish data to ROS topics
- Sensor data is accessible via ROS topics
- Simulation runs at real-time speed

### Validation Steps
- Verify all sensor topics are publishing data
- Check sensor data quality and range
- Confirm robot responds to physics in the environment

## Exercise 3: Unity Integration and Visualization

### Goal
Create a basic Unity scene that connects to ROS and visualizes robot sensor data.

### Prerequisites
- Unity Hub and Unity 2022.3 LTS installed
- Unity Robotics Package installed
- Completed previous exercises

### Steps

1. **Set up Unity project:**
   - Create new 3D project in Unity
   - Import Unity Robotics Package via Package Manager
   - Import Perception Package for synthetic data generation

2. **Create basic robot visualization:**
   Create a C# script `Assets/Scripts/RobotVisualizer.cs`:

   ```csharp
   using UnityEngine;
   using Unity.Robotics.ROSTCPConnector;
   using Unity.Robotics.ROSTCPConnector.MessageTypes.Sensor;
   using System.Collections.Generic;

   public class RobotVisualizer : MonoBehaviour
   {
       [Header("ROS Connection")]
       public string rosIP = "127.0.0.1";
       public int rosPort = 10000;

       [Header("Robot Parts")]
       public Transform baseLink;
       public Transform lidarLink;
       public Transform cameraLink;

       [Header("Visualization")]
       public GameObject lidarPointPrefab;
       public LineRenderer laserLineRenderer;

       private ROSConnection ros;
       private List<GameObject> lidarPoints = new List<GameObject>();

       void Start()
       {
           ros = ROSConnection.GetOrCreateInstance();
           ros.Initialize(rosIP, rosPort);

           ros.Subscribe<LaserScanMsg>("/scan", OnLaserScanReceived);
           ros.Subscribe<ImageMsg>("/camera/image_raw", OnCameraImageReceived);
       }

       void OnLaserScanReceived(LaserScanMsg scan)
       {
           // Clear previous lidar points
           foreach (GameObject point in lidarPoints)
           {
               DestroyImmediate(point);
           }
           lidarPoints.Clear();

           // Create visualization points for each laser reading
           for (int i = 0; i < scan.ranges.Length; i++)
           {
               float angle = scan.angle_min + i * scan.angle_increment;
               float distance = scan.ranges[i];

               if (!float.IsNaN(distance) && !float.IsInfinity(distance) && distance <= scan.range_max)
               {
                   Vector3 pointPos = new Vector3(
                       distance * Mathf.Cos(angle),
                       0.1f, // Height above ground
                       distance * Mathf.Sin(angle)
                   );

                   GameObject point = Instantiate(lidarPointPrefab, pointPos, Quaternion.identity, lidarLink);
                   lidarPoints.Add(point);
               }
           }
       }

       void OnCameraImageReceived(ImageMsg image)
       {
           // Process camera image for visualization
           // This would typically involve texture creation and assignment
       }
   }
   ```

3. **Create LiDAR visualization prefab:**
   - Create a sphere GameObject in Unity
   - Scale it down (e.g., 0.02 in all axes)
   - Set material to bright color (e.g., red)
   - Save as prefab

4. **Set up the Unity scene:**
   - Add RobotVisualizer script to Main Camera
   - Assign baseLink, lidarLink, cameraLink transforms
   - Assign the LiDAR point prefab
   - Configure ROS IP and port

5. **Test Unity-ROS connection:**
   - Run the Unity scene
   - Start the Gazebo simulation with the sensor robot
   - Verify LiDAR points appear in Unity based on Gazebo sensor data

### Expected Outcome
- Unity connects to ROS and receives sensor data
- LiDAR readings are visualized as points in Unity
- Robot visualization updates in real-time based on ROS data

### Validation Steps
- Confirm Unity connects to ROS
- Verify sensor data is received and processed
- Check that visualization updates correctly

## Exercise 4: Physics Validation and Tuning

### Goal
Validate and tune physics parameters to ensure realistic robot behavior in simulation.

### Prerequisites
- Working Gazebo simulation with robot model
- Basic understanding of physics concepts

### Steps

1. **Create a physics validation test:**
   Create `~/ros2_ws/src/my_robot_simulation/worlds/physics_test.world`:

   ```xml
   <?xml version="1.0" ?>
   <sdf version="1.7">
     <world name="physics_test">
       <include>
         <uri>model://ground_plane</uri>
       </include>
       <include>
         <uri>model://sun</uri>
       </include>

       <!-- Physics configuration -->
       <physics type="ode">
         <max_step_size>0.001</max_step_size>
         <real_time_factor>1</real_time_factor>
         <real_time_update_rate>1000</real_time_update_rate>
         <gravity>0 0 -9.8</gravity>
         <ode>
           <solver>
             <type>quick</type>
             <iters>10</iters>
             <sor>1.3</sor>
           </solver>
           <constraints>
             <cfm>0.0</cfm>
             <erp>0.2</erp>
             <contact_max_correcting_vel>100</contact_max_correcting_vel>
             <contact_surface_layer>0.001</contact_surface_layer>
           </constraints>
         </ode>
       </physics>

       <!-- Test objects with different properties -->
       <model name="steel_sphere">
         <pose>-2 0 2 0 0 0</pose>
         <link name="link">
           <inertial>
             <mass>7.6</mass> <!-- Steel density * volume -->
             <inertia>
               <ixx>0.0152</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.0152</iyy>
               <iyz>0</iyz>
               <izz>0.0152</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <sphere>
                 <radius>0.1</radius>
               </sphere>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <sphere>
                 <radius>0.1</radius>
               </sphere>
             </geometry>
             <material>
               <ambient>0.5 0.5 0.5 1</ambient>
               <diffuse>0.7 0.7 0.7 1</diffuse>
               <specular>0.9 0.9 0.9 1</specular>
             </material>
           </visual>
         </link>
       </model>

       <model name="wooden_box">
         <pose>0 0 2 0 0 0</pose>
         <link name="link">
           <inertial>
             <mass>2.4</mass> <!-- Wood density * volume -->
             <inertia>
               <ixx>0.08</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.08</iyy>
               <iyz>0</iyz>
               <izz>0.08</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <box>
                 <size>0.2 0.2 0.2</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>0.2 0.2 0.2</size>
               </box>
             </geometry>
             <material>
               <ambient>0.6 0.4 0.2 1</ambient>
               <diffuse>0.8 0.6 0.4 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <!-- Ramp for testing friction -->
       <model name="ramp">
         <pose>2 0 0 0 0.3 0</pose> <!-- 0.3 rad ≈ 17° angle -->
         <link name="link">
           <collision name="collision">
             <geometry>
               <box>
                 <size>2 1 0.1</size>
               </box>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <box>
                 <size>2 1 0.1</size>
               </box>
             </geometry>
             <material>
               <ambient>0.3 0.3 0.3 1</ambient>
               <diffuse>0.5 0.5 0.5 1</diffuse>
             </material>
           </visual>
         </link>
       </model>

       <!-- Objects to test on ramp -->
       <model name="high_friction_object">
         <pose>1.5 0 0.5 0 0 0</pose>
         <link name="link">
           <inertial>
             <mass>1.0</mass>
             <inertia>
               <ixx>0.01</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.01</iyy>
               <iyz>0</iyz>
               <izz>0.01</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <sphere>
                 <radius>0.05</radius>
               </sphere>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <sphere>
                 <radius>0.05</radius>
               </sphere>
             </geometry>
             <material>
               <ambient>0 1 0 1</ambient>
               <diffuse>0 1 0 1</diffuse>
             </material>
           </visual>
           <gazebo>
             <mu1>1.0</mu1> <!-- High friction -->
             <mu2>1.0</mu2>
           </gazebo>
         </link>
       </model>

       <model name="low_friction_object">
         <pose>1.8 0 0.5 0 0 0</pose>
         <link name="link">
           <inertial>
             <mass>1.0</mass>
             <inertia>
               <ixx>0.01</ixx>
               <ixy>0</ixy>
               <ixz>0</ixz>
               <iyy>0.01</iyy>
               <iyz>0</iyz>
               <izz>0.01</izz>
             </inertia>
           </inertial>
           <collision name="collision">
             <geometry>
               <sphere>
                 <radius>0.05</radius>
               </sphere>
             </geometry>
           </collision>
           <visual name="visual">
             <geometry>
               <sphere>
                 <radius>0.05</radius>
               </sphere>
             </geometry>
             <material>
               <ambient>1 0 0 1</ambient>
               <diffuse>1 0 0 1</diffuse>
             </material>
           </visual>
           <gazebo>
             <mu1>0.1</mu1> <!-- Low friction -->
             <mu2>0.1</mu2>
           </gazebo>
         </link>
       </model>
     </world>
   </sdf>
   ```

2. **Run the physics validation test:**
   ```bash
   gazebo ~/ros2_ws/src/my_robot_simulation/worlds/physics_test.world
   ```

3. **Observe and validate:**
   - Watch steel sphere fall and bounce with realistic physics
   - Observe wooden box behavior
   - Compare high vs low friction objects on the ramp
   - Note the differences in motion

4. **Tune parameters based on observations:**
   - Adjust time step if simulation is unstable
   - Modify friction coefficients for more realistic behavior
   - Fine-tune mass and inertia properties

### Expected Outcome
- Objects behave according to realistic physics properties
- High friction objects stay on ramp, low friction objects slide
- Bouncing objects exhibit proper energy loss
- Simulation remains stable and performant

### Validation Steps
- Verify objects fall at correct acceleration (9.8 m/s²)
- Confirm friction properties affect motion as expected
- Ensure simulation runs stably without oscillations or explosions
- Measure performance to ensure real-time operation

## Summary

These hands-on exercises provide practical experience with:
- Setting up Gazebo environments with custom models
- Integrating various sensor types into robot models
- Connecting Unity visualization with ROS/Gazebo
- Validating and tuning physics properties for realistic behavior

Completing these exercises will give you the skills needed to create sophisticated simulation environments for humanoid robots, enabling safe testing and validation before deployment on real hardware.