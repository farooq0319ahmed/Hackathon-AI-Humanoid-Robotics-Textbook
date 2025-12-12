# Data Model: Physical AI & Humanoid Robotics Book

**Date**: 2025-12-12
**Feature**: 001-humanoid-robotics-book
**Status**: Complete

## Key Entities

### Learning Module
- **Description**: Educational content unit focused on a specific aspect of humanoid robotics
- **Attributes**:
  - module_id: string (unique identifier, e.g., "01-ros-nervous-system")
  - title: string (e.g., "The Robotic Nervous System")
  - objectives: array of strings (learning objectives)
  - content: string (main content in Docusaurus-compatible Markdown)
  - exercises: array of Exercise objects
  - examples: array of Example objects
  - summary: string (key takeaways)
  - prerequisites: array of strings (required knowledge)
  - duration: integer (estimated completion time in minutes)

### Exercise
- **Description**: Practical implementation activity that allows students to apply theoretical concepts
- **Attributes**:
  - exercise_id: string (unique within module)
  - title: string (e.g., "Create Your First ROS 2 Node")
  - description: string (what the student will accomplish)
  - steps: array of strings (step-by-step instructions)
  - requirements: array of strings (hardware/software needed)
  - expected_outcome: string (what success looks like)
  - difficulty: enum ("beginner", "intermediate", "advanced")

### Example
- **Description**: Code or configuration example within a module
- **Attributes**:
  - example_id: string (unique within module)
  - title: string (e.g., "Publisher Node Example")
  - code: string (actual code content)
  - language: string (e.g., "python", "bash", "xml")
  - explanation: string (what the code does)
  - use_case: string (when to use this example)

### Simulation Environment
- **Description**: Virtual space where robot behaviors can be tested safely
- **Attributes**:
  - env_id: string (unique identifier)
  - name: string (e.g., "Gazebo Indoor Navigation")
  - platform: enum ("gazebo", "unity", "isaac-sim")
  - description: string (what the environment simulates)
  - robot_model: string (URDF/SDF model to use)
  - sensors: array of strings (e.g., "lidar", "camera", "imu")
  - physics_config: object (gravity, collision parameters)
  - scenario: string (specific test scenario)

### Robot Model
- **Description**: Digital representation of a humanoid robot with defined links, joints, sensors, and actuators
- **Attributes**:
  - model_id: string (unique identifier)
  - name: string (e.g., "Simple Humanoid")
  - format: enum ("urdf", "sdf", "usd")
  - links: array of Link objects
  - joints: array of Joint objects
  - sensors: array of Sensor objects
  - actuators: array of Actuator objects
  - description: string (what this robot model represents)

### Link
- **Description**: Rigid body component of a robot model
- **Attributes**:
  - link_id: string (unique within robot model)
  - name: string (e.g., "torso", "left_arm")
  - visual: object (visual representation)
  - collision: object (collision properties)
  - inertial: object (mass, center of mass)

### Joint
- **Description**: Connection between two links that allows motion
- **Attributes**:
  - joint_id: string (unique within robot model)
  - name: string (e.g., "hip_joint", "shoulder_joint")
  - type: enum ("revolute", "prismatic", "fixed", etc.)
  - parent_link: string (name of parent link)
  - child_link: string (name of child link)
  - limits: object (min/max values for joint motion)

### Sensor
- **Description**: Component that provides sensory information about the environment
- **Attributes**:
  - sensor_id: string (unique within robot model)
  - name: string (e.g., "front_camera", "lidar_3d")
  - type: enum ("camera", "lidar", "imu", "force_torque")
  - parent_link: string (which link the sensor is attached to)
  - parameters: object (sensor-specific configuration)

### Actuator
- **Description**: Component that provides motion control for joints
- **Attributes**:
  - actuator_id: string (unique within robot model)
  - name: string (e.g., "hip_actuator", "gripper_actuator")
  - joint: string (which joint this actuator controls)
  - type: enum ("position", "velocity", "effort")
  - parameters: object (actuator-specific configuration)

### AI Pipeline
- **Description**: Sequence of processing steps for perception, decision-making, and action execution
- **Attributes**:
  - pipeline_id: string (unique identifier)
  - name: string (e.g., "VSLAM Pipeline", "Navigation Pipeline")
  - type: enum ("vslam", "navigation", "manipulation", "reinforcement_learning")
  - input_topics: array of strings (ROS 2 topics the pipeline subscribes to)
  - output_topics: array of strings (ROS 2 topics the pipeline publishes to)
  - parameters: object (algorithm-specific configuration)
  - description: string (what this pipeline accomplishes)

### Capstone Project
- **Description**: Comprehensive end-to-end project integrating all learned concepts
- **Attributes**:
  - project_id: string (unique identifier)
  - title: string (e.g., "Autonomous Humanoid Task Execution")
  - description: string (overall project goal)
  - requirements: array of strings (what's needed to complete the project)
  - phases: array of Phase objects
  - success_criteria: array of strings (how to know the project is complete)
  - estimated_duration: integer (in hours)

### Phase
- **Description**: Sequential step within a capstone project
- **Attributes**:
  - phase_id: string (unique within project)
  - title: string (e.g., "Voice Command Processing")
  - description: string (what happens in this phase)
  - dependencies: array of strings (what must be completed first)
  - deliverables: array of strings (what is produced)
  - validation_steps: array of strings (how to verify completion)