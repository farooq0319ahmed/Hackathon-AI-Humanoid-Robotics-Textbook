---
sidebar_position: 2
---

# URDF Specification: Detailed Format Documentation

## Overview

URDF (Unified Robot Description Format) is the standard XML format for representing robot models in ROS. This specification provides comprehensive documentation for creating robot descriptions, with particular focus on humanoid robots. Understanding URDF is crucial for simulation, visualization, and control of humanoid robots.

## URDF Structure

### Root Element

Every URDF file must have a single `<robot>` root element:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Robot content goes here -->
</robot>
```

### Required Attributes
- `name`: Unique identifier for the robot

### Optional Attributes
- `version`: URDF specification version (defaults to 1.0)

## Links

Links represent rigid bodies in the robot structure. Each link must have a unique name.

### Basic Link Structure

```xml
<link name="link_name">
  <!-- Optional elements (in order) -->
  <inertial>...</inertial>
  <visual>...</visual>
  <collision>...</collision>
  <self_collision_check>
    <radius>0.1</radius>
  </self_collision_check>
</link>
```

### Inertial Properties

The `<inertial>` element defines the inertial properties of a link:

```xml
<inertial>
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
  <mass value="1.0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

**Elements:**
- `<origin>`: Pose of inertial reference frame relative to link frame
  - `xyz`: Position (x, y, z) in meters
  - `rpy`: Orientation (roll, pitch, yaw) in radians
- `<mass>`: Mass in kilograms
- `<inertia>`: Inertia matrix values (symmetric, only 6 values needed)

### Visual Properties

The `<visual>` element defines how a link appears for visualization:

```xml
<visual name="visual_name">
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
  <geometry>
    <!-- Geometry type -->
  </geometry>
  <material name="material_name">
    <color rgba="0.8 0.2 0.2 1.0"/>
    <texture filename="package://my_robot/meshes/texture.png"/>
  </material>
</visual>
```

**Elements:**
- `<origin>`: Pose of visual geometry relative to link frame
- `<geometry>`: Shape of the visual element (see Geometry section)
- `<material>`: Material properties for visualization

### Collision Properties

The `<collision>` element defines collision properties:

```xml
<collision name="collision_name">
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
  <geometry>
    <!-- Geometry type -->
  </geometry>
</collision>
```

**Elements:**
- `<origin>`: Pose of collision geometry relative to link frame
- `<geometry>`: Shape of the collision element

## Geometry Types

URDF supports four basic geometry types:

### Box Geometry

```xml
<geometry>
  <box size="1.0 0.5 0.2"/>  <!-- width height depth -->
</geometry>
```

### Cylinder Geometry

```xml
<geometry>
  <cylinder radius="0.1" length="0.5"/>
</geometry>
```

### Sphere Geometry

```xml
<geometry>
  <sphere radius="0.1"/>
</geometry>
```

### Mesh Geometry

```xml
<geometry>
  <mesh filename="package://my_robot/meshes/link_mesh.stl" scale="1.0 1.0 1.0"/>
</geometry>
```

**Attributes:**
- `filename`: Path to mesh file (using ROS package syntax)
- `scale`: Optional scaling factor (default: 1.0 1.0 1.0)

## Joints

Joints connect links and define their relative motion.

### Basic Joint Structure

```xml
<joint name="joint_name" type="joint_type">
  <origin xyz="0.0 0.0 0.0" rpy="0.0 0.0 0.0"/>
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <axis xyz="1 0 0"/>
  <limit lower="-3.14" upper="3.14" effort="100.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
  <safety_controller k_position="100.0" k_velocity="10.0" soft_lower_limit="-3.0" soft_upper_limit="3.0"/>
</joint>
```

### Joint Types

1. **revolute**: Rotational joint with limits
2. **continuous**: Rotational joint without limits
3. **prismatic**: Linear sliding joint with limits
4. **fixed**: No movement (rigid connection)
5. **floating**: 6 DOF (rarely used)
6. **planar**: Movement in a plane (rarely used)

### Joint Elements

- `<origin>`: Pose of joint frame relative to parent link
- `<parent>`: Name of parent link
- `<child>`: Name of child link
- `<axis>`: Joint axis in joint frame (normalized)
- `<limit>`: Joint limits (for revolute/prismatic)
- `<dynamics>`: Joint dynamics (damping, friction)
- `<safety_controller>`: Safety limits

### Joint Limits

For revolute and prismatic joints:

```xml
<limit lower="-1.57" upper="1.57" effort="100.0" velocity="1.0"/>
```

**Attributes:**
- `lower`: Lower joint limit (radians for revolute, meters for prismatic)
- `upper`: Upper joint limit
- `effort`: Maximum joint effort (N for prismatic, N-m for revolute)
- `velocity`: Maximum joint velocity

### Joint Dynamics

```xml
<dynamics damping="0.1" friction="0.0"/>
```

**Attributes:**
- `damping`: Damping coefficient
- `friction`: Static friction coefficient

## Transmissions

Transmissions define how joints connect to actuators:

```xml
<transmission name="transmission_name">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="actuator_name">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Hardware Interfaces

Common hardware interfaces:
- `hardware_interface/EffortJointInterface`
- `hardware_interface/VelocityJointInterface`
- `hardware_interface/PositionJointInterface`

## Gazebo Extensions

For simulation in Gazebo, you can add Gazebo-specific elements:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Red</material>
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <self_collide>false</self_collide>
  <gravity>true</gravity>
  <max_contacts>10</max_contacts>
</gazebo>
```

### Sensor Definitions

```xml
<gazebo reference="sensor_link">
  <sensor name="camera_sensor" type="camera">
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov>
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
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Complete Humanoid Robot Example

Here's a complete example of a simplified humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white">
        <color rgba="0.9 0.9 0.9 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <origin xyz="0 0 0.9" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="head"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left arm base -->
  <link name="left_shoulder">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Left shoulder joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <origin xyz="0.15 0.1 0.7" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="left_shoulder"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Right arm (similar to left) -->
  <link name="right_shoulder">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>

    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Right shoulder joint -->
  <joint name="right_shoulder_joint" type="revolute">
    <origin xyz="0.15 -0.1 0.7" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="right_shoulder"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="50.0" velocity="2.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <!-- Left leg -->
  <link name="left_hip">
    <inertial>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Left hip joint -->
  <joint name="left_hip_joint" type="revolute">
    <origin xyz="-0.1 0.05 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="left_hip"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100.0" velocity="1.0"/>
    <dynamics damping="0.2" friction="0.1"/>
  </joint>

  <!-- Right leg -->
  <link name="right_hip">
    <inertial>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>

    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0.2 0.8 0.2 1.0"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.08" length="0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Right hip joint -->
  <joint name="right_hip_joint" type="revolute">
    <origin xyz="-0.1 -0.05 0" rpy="0 0 0"/>
    <parent link="base_link"/>
    <child link="right_hip"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="100.0" velocity="1.0"/>
    <dynamics damping="0.2" friction="0.1"/>
  </joint>

  <!-- Transmissions for ROS 2 Control -->
  <transmission name="left_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_shoulder_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_shoulder_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_shoulder_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_shoulder_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/simple_humanoid</robotNamespace>
    </plugin>
  </gazebo>

</robot>
```

## Xacro Integration

Xacro (XML Macros) allows for parameterization and reusability:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="1.0" />

  <!-- Macro for creating limbs -->
  <xacro:macro name="limb" params="name parent xyz rpy axis">
    <link name="${name}_link">
      <inertial>
        <origin xyz="0 0 -0.25" rpy="0 0 0"/>
        <mass value="3.0"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
      </inertial>

      <visual>
        <origin xyz="0 0 -0.25" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.08" length="0.5"/>
        </geometry>
      </visual>

      <collision>
        <origin xyz="0 0 -0.25" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="0.08" length="0.5"/>
        </geometry>
      </collision>
    </link>

    <joint name="${name}_joint" type="revolute">
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <axis xyz="${axis}"/>
      <limit lower="-1.57" upper="1.57" effort="100.0" velocity="1.0"/>
      <dynamics damping="0.2" friction="0.1"/>
    </joint>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_width} ${base_height}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
  </link>

  <!-- Use macro to create limbs -->
  <xacro:limb name="left_leg" parent="base_link" xyz="-0.1 0.05 0" rpy="0 0 0" axis="1 0 0"/>
  <xacro:limb name="right_leg" parent="base_link" xyz="-0.1 -0.05 0" rpy="0 0 0" axis="1 0 0"/>

</robot>
```

## Validation and Testing

### URDF Validation

```bash
# Validate URDF syntax
check_urdf /path/to/robot.urdf

# Visualize URDF structure
urdf_to_graphiz /path/to/robot.urdf

# Check for collisions
ros2 run rviz2 rviz2
# Load robot_description and visualize
```

### Common Validation Issues

1. **Missing parent/child links**: Ensure all joint references exist
2. **Non-unique names**: All links and joints must have unique names
3. **Invalid geometry**: Check all geometry parameters are positive
4. **Inertial issues**: Ensure inertia matrix is positive definite

## Best Practices for Humanoid Robots

### 1. Proper Inertial Properties
- Calculate realistic inertial properties for each link
- Use CAD software to compute accurate inertias
- Ensure center of mass is properly positioned

### 2. Joint Limit Considerations
- Set appropriate joint limits based on mechanical constraints
- Include safety margins in joint limits
- Consider human-like joint ranges for humanoid robots

### 3. Collision Avoidance
- Design collision geometries that are conservative enough to prevent collisions
- Use multiple collision elements if needed
- Test collision detection in simulation

### 4. Visualization vs Collision
- Use detailed meshes for visualization
- Use simplified geometries for collision detection
- Keep collision geometries convex when possible

### 5. Frame Conventions
- Follow ROS coordinate frame conventions (X forward, Y left, Z up)
- Use consistent naming conventions
- Define appropriate tool frames for end effectors

## Troubleshooting

### Common Issues and Solutions

1. **"Joint references unknown link"**: Verify all referenced links exist
2. **"Inertia matrix not positive definite"**: Check inertia values and signs
3. **"Zero mass link"**: Ensure all links have mass > 0
4. **"Invalid geometry"**: Check geometry parameters are positive
5. **"Floating point errors"**: Use appropriate precision for coordinates

### Debugging URDF

```bash
# Check URDF for errors
check_urdf robot.urdf

# Visualize the robot tree
urdf_to_graphiz robot.urdf
evince robot.pdf  # Open the generated PDF

# Load in RViz
ros2 run rviz2 rviz2
# Add RobotModel display and set robot_description
```

This comprehensive specification provides all the details needed to create proper URDF files for humanoid robots. Use this as a reference when designing your robot models throughout the Physical AI & Humanoid Robotics Book.