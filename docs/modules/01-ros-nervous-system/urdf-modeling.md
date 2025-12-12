---
sidebar_position: 4
---

# URDF: Defining Robot Components

## Understanding URDF

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and other components like sensors and actuators.

URDF is crucial for:
- Robot simulation in Gazebo and other simulators
- Robot visualization in RViz
- Kinematic analysis and motion planning
- Collision detection and avoidance

## URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links definition -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints definition -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Links

Links represent rigid parts of the robot. Each link can have multiple properties:

### Visual Properties
Defines how the link appears in visualization tools:

```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Choose one geometry type -->
      <box size="1 1 1"/>
      <!-- OR -->
      <cylinder radius="0.1" length="0.2"/>
      <!-- OR -->
      <sphere radius="0.1"/>
      <!-- OR -->
      <mesh filename="package://my_robot/meshes/link.stl"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
</link>
```

### Collision Properties
Defines the collision geometry for physics simulation:

```xml
<link name="link_name">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties
Defines the physical properties for dynamics simulation:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

## Joints

Joints connect links and define how they can move relative to each other. Common joint types include:

### Fixed Joint
No movement between links:

```xml
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>
```

### Revolute Joint
Rotation around a single axis with limits:

```xml
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
</joint>
```

### Continuous Joint
Unlimited rotation around a single axis:

```xml
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>
```

### Prismatic Joint
Linear motion along a single axis:

```xml
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0.0" upper="0.5" effort="10.0" velocity="1.0"/>
</joint>
```

## Complete Humanoid Robot Example

Here's a simplified example of a humanoid robot torso:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.6" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin_color">
        <color rgba="0.9 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
  </joint>

  <!-- Left upper arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left shoulder joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

## Xacro: URDF with Macros

Xacro is a macro language that extends URDF, allowing for more complex and reusable robot descriptions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Define properties -->
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Define a macro for wheels -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.1 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.1 0" rpy="0 0 0"/>

</robot>
```

## Best Practices for URDF Modeling

### Structure and Organization
- Use a clear hierarchy with the base link as the root
- Name links and joints descriptively
- Group related components logically

### Visual and Collision Separation
- Use simple shapes for collision to improve performance
- Use detailed meshes for visual representation
- Consider using different resolutions for different purposes

### Inertial Properties
- Calculate realistic inertial properties for accurate simulation
- Use CAD tools to calculate inertial properties when possible
- Verify that the center of mass is correctly positioned

### Joint Limits and Safety
- Always set appropriate joint limits to prevent damage
- Consider the physical constraints of real hardware
- Use safety margins in joint limits

## Summary

URDF is a powerful tool for describing robot models in ROS. Understanding how to properly define links, joints, and their properties is essential for robot simulation, visualization, and control. The combination of visual, collision, and inertial properties allows for comprehensive robot modeling that works across different simulation and visualization environments.

In the next section, we'll explore hands-on exercises to apply these concepts practically.