---
sidebar_position: 3
---

# Nav2 Path Planning: Navigation for Humanoid Robots

## Overview

Navigation is the process of planning and executing paths for robots to move from their current location to a desired destination while avoiding obstacles. The Navigation 2 (Nav2) stack is the standard navigation system for ROS 2, providing a complete framework for mobile robot navigation that includes global and local path planning, obstacle avoidance, and recovery behaviors.

For humanoid robots, navigation poses unique challenges compared to wheeled robots, including bipedal locomotion patterns, different kinematics, and specific gait requirements. This section explores how to configure and use Nav2 specifically for humanoid robots.

## Core Navigation Concepts

### Navigation Stack Architecture

The Nav2 stack consists of several key components working together:

```
Global Planner → Controller → Local Planner → Robot
     ↑              ↑           ↑
  Global Map   Trajectory    Local Map
                 Control
```

1. **Global Planner**: Creates a path from start to goal based on the global map
2. **Controller**: Generates low-level velocity commands to follow the path
3. **Local Planner**: Creates short-term trajectories while avoiding obstacles
4. **World Model**: Maintains information about obstacles and free space
5. **Recovery Behaviors**: Handles navigation failures and obstacles

### Navigation States

The navigation system transitions through various states:

```
[UNKNOWN] → [IDLE] → [NAVIGATING] → [CONTROLLING] → [RECOVERING] → [FAILED]
                                    ↓
                               [SUCCEEDED]
```

## Nav2 Configuration for Humanoid Robots

### Basic Navigation Configuration

For humanoid robots, the navigation configuration needs to account for bipedal locomotion characteristics:

```yaml
# nav2_params_humanoid.yaml
bt_navigator:
  ros__parameters:
    use_sim_time: false
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    default_server_timeout: 20
    # Recovery nodes
    navigators: ["spin", "backup", "drive_on_heading", "assisted_teleop"]
    navigate_through_poses:
      plugin: "nav2_bt_navigator/NavigateThroughPoses"
    navigate_to_pose:
      plugin: "nav2_bt_navigator/NavigateToPose"

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: false

controller_server:
  ros__parameters:
    use_sim_time: false
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.15
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      primary_controller: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      # Humanoid-specific parameters
      simulate_ahead_time: 1.0
      max_allowed_time_to_collision_up_to_carrot: 1.0
      carrot_planner_max_angular_vel: 0.5
      lookahead_time: 1.5
      rotate_to_heading_angular_vel: 0.3
      min_angle_to_turn: 0.2

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: false
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Larger for humanoid robots
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: false
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: false
      robot_radius: 0.3  # Larger for humanoid robots
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific parameters
      visualize_potential: false

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "drive_on_heading", "assisted_teleop"]
    recovery_plugin_types: ["nav2_recoveries/Spin", "nav2_recoveries/Backup", "nav2_recoveries/DriveOnHeading", "nav2_recoveries/AssistedTeleop"]
    spin:
      spin_dist: 1.57
      time_allowance: 10.0
    backup:
      backup_dist: -0.15
      backup_speed: 0.025
    drive_on_heading:
      drive_on_heading_angle_tolerance: 0.785
      drive_on_heading_forward_sampling_distance: 0.5
      drive_on_heading_lateral_samples: 5
      drive_on_heading_angular_samples: 10
    assisted_teleop:
      rotation_speed_limit: 0.4
      translation_speed_limit: 0.2

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1000
```

### Humanoid-Specific Parameters

For humanoid robots, special parameters account for bipedal movement patterns:

```yaml
# Humanoid-specific controller parameters
controller_server:
  ros__parameters:
    # Bipedal locomotion parameters
    step_frequency: 2.0  # Hz, typical for human-like walking
    step_length: 0.3     # meters per step
    step_height: 0.05    # clearance for foot lift
    foot_separation: 0.2 # distance between feet during walking

    # Balance constraints
    com_height: 0.8      # center of mass height
    zmp_tolerance: 0.05  # zero moment point deviation tolerance

    # Gait planning
    gait_pattern: "walk" # Options: walk, trot, pace, bound
    cadence: 2.0         # steps per second
    stride_length: 0.3   # distance per step

    # Stability margins
    support_polygon_margin: 0.1
    max_step_deviation: 0.1
```

## Path Planning Algorithms

### Global Path Planners

Nav2 supports multiple global planners optimized for different scenarios:

#### Navfn Planner
```cpp
// Navfn is the default global planner in Nav2
// Good for general navigation with Dijkstra's algorithm
// Configured in the YAML file above
```

#### CARMA Planner
For more complex humanoid navigation:

```yaml
# CARMA (Collision Avoidance and Reactive Motion for Autonomous Systems) Planner
planner_server:
  ros__parameters:
    planner_plugins: ["CARMAPlanner"]
    CARMAPlanner:
      plugin: "nav2_carma_planner/CARMANavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific CARMA parameters
      max_curve_radius: 0.5
      min_curve_radius: 0.2
      curve_adaptation_factor: 0.8
```

### Local Path Planners

For humanoid robots, local planners need to account for complex dynamics:

#### Regulated Pure Pursuit Controller

```yaml
# Configured in controller_server section above
# Optimized for non-holonomic robots but can be adapted for bipeds
controller_server:
  ros__parameters:
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.2      # Conservative speed for bipedal robots
      lookahead_dist: 0.6
      lookahead_time: 1.0
      transform_tolerance: 0.1
      use_velocity_scaled_lookahead_dist: true
      min_approach_linear_velocity: 0.05
      arrive_linear_velocity: 0.0
      proportional_gain: 2.0
      speed_scaling_instead_of_clipping: true
      use_interpolation: true
      # Humanoid-specific adjustments
      regulate_frequency: 20.0
      tracking_threshold: 0.1
```

## Behavior Trees for Navigation

Nav2 uses behavior trees to define navigation workflows. For humanoid robots, the behavior tree can be customized to include robot-specific actions:

```xml
<!-- navigate_w_humanoid_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <PipelineSequence name="NavigateWithReplanning">
            <RateController hz="1.0">
                <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
            </RateController>
            <RecoveryNode number_of_retries="6" name="NavigateRecovery">
                <PipelineSequence name="ClearingActions">
                    <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
                    <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
                </PipelineSequence>
                <PipelineSequence name="MoveActions">
                    <Spin spin_dist="1.57"/>
                    <Backup backup_dist="0.15" backup_speed="0.025"/>
                    <DriveOnHeading drive_on_heading_angle_tolerance="0.785" drive_on_heading_forward_sampling_distance="0.5" drive_on_heading_lateral_samples="5" drive_on_heading_angular_samples="10"/>
                </PipelineSequence>
            </RecoveryNode>
        </PipelineSequence>
        <ReactiveFallback name="MoveRobot">
            <GoalUpdated/>
            <PipelineSequence name="LocalPlanAndControl">
                <ControllerFollowPath path="{path}" controller_id="FollowPath" goal_checker_id="goal_checker"/>
            </PipelineSequence>
        </ReactiveFallback>
    </BehaviorTree>
</root>
```

## Humanoid-Specific Navigation Considerations

### Balance and Stability

Humanoid robots must maintain balance during navigation:

```cpp
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav2_core/controller.hpp"

class HumanoidController : public nav2_core::Controller
{
public:
    HumanoidController() : balance_threshold_(0.05), com_height_(0.8) {}

    void configure(
        const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
        std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
        const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros) override
    {
        controller_name_ = name;
        node_ = node;
        tf_ = tf;
        costmap_ros_ = costmap_ros;

        // Subscribe to IMU for balance monitoring
        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10,
            std::bind(&HumanoidController::imuCallback, this, std::placeholders::_1));
    }

    geometry_msgs::msg::Twist computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity,
        nav2_core::GoalChecker * goal_checker) override
    {
        geometry_msgs::msg::Twist cmd_vel;

        // Check balance before issuing commands
        if (!isBalanced()) {
            RCLCPP_WARN(node_->get_logger(), "Robot is unbalanced, stopping movement");
            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;
            return cmd_vel;
        }

        // Compute normal velocity commands
        cmd_vel = computeNormalCommands(pose, velocity);

        // Apply stability constraints
        cmd_vel = applyStabilityConstraints(cmd_vel);

        return cmd_vel;
    }

private:
    bool isBalanced()
    {
        // Check center of mass projection relative to support polygon
        double zmp_x = imu_data_.linear_acceleration.x * com_height_ / 9.81;
        double zmp_y = imu_data_.linear_acceleration.y * com_height_ / 9.81;

        // Check if ZMP is within support polygon
        return std::abs(zmp_x) < balance_threshold_ && std::abs(zmp_y) < balance_threshold_;
    }

    geometry_msgs::msg::Twist computeNormalCommands(
        const geometry_msgs::msg::PoseStamped & pose,
        const geometry_msgs::msg::Twist & velocity)
    {
        // Standard path following commands
        geometry_msgs::msg::Twist cmd_vel;
        // ... path following algorithm ...
        return cmd_vel;
    }

    geometry_msgs::msg::Twist applyStabilityConstraints(geometry_msgs::msg::Twist cmd_vel)
    {
        // Adjust velocities to maintain stability
        if (std::abs(cmd_vel.angular.z) > max_angular_velocity_) {
            cmd_vel.angular.z = std::copysign(max_angular_velocity_, cmd_vel.angular.z);
        }

        if (std::abs(cmd_vel.linear.x) > max_linear_velocity_) {
            cmd_vel.linear.x = std::copysign(max_linear_velocity_, cmd_vel.linear.x);
        }

        return cmd_vel;
    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        imu_data_ = *msg;
    }

    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    std::string controller_name_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    sensor_msgs::msg::Imu imu_data_;

    double balance_threshold_;
    double com_height_;
    double max_linear_velocity_{0.2};    // Conservative for bipedal
    double max_angular_velocity_{0.3};   // Conservative for balance
};
```

### Footstep Planning

Advanced humanoid navigation requires footstep planning:

```python
import numpy as np
from scipy.spatial.distance import euclidean

class FootstepPlanner:
    def __init__(self, robot_params):
        self.foot_spacing = robot_params['foot_spacing']  # Distance between feet
        self.step_length = robot_params['step_length']
        self.max_step_deviation = robot_params['max_step_deviation']

    def plan_footsteps(self, path, start_pose):
        """
        Plan safe footsteps along a path for a humanoid robot
        """
        footsteps = []

        # Current foot positions (left, right)
        current_left = self.offset_pose(start_pose, -self.foot_spacing/2, 0)
        current_right = self.offset_pose(start_pose, self.foot_spacing/2, 0)

        # Start with current stance
        footsteps.append(('left', current_left))
        footsteps.append(('right', current_right))

        # Plan footsteps along path
        for i in range(len(path)-1):
            segment_start = path[i]
            segment_end = path[i+1]

            # Determine which foot to move based on path direction
            next_left, next_right = self.calculate_next_step(
                current_left, current_right, segment_end)

            # Check if footstep is valid
            if self.is_valid_footstep(next_left) and \
               self.distance_to_path(next_left, path) < self.max_step_deviation:
                footsteps.append(('left', next_left))
                current_left = next_left
            elif self.is_valid_footstep(next_right) and \
                 self.distance_to_path(next_right, path) < self.max_step_deviation:
                footsteps.append(('right', next_right))
                current_right = next_right

        return footsteps

    def calculate_next_step(self, left_foot, right_foot, target_pose):
        """
        Calculate the next step position for each foot
        """
        # Calculate direction toward target
        dx = target_pose.x - (left_foot.x + right_foot.x) / 2
        dy = target_pose.y - (left_foot.y + right_foot.y) / 2

        # Determine which foot leads based on path direction
        angle_to_target = np.arctan2(dy, dx)

        # Calculate next positions
        next_left = self.advance_foot(left_foot, angle_to_target, self.step_length)
        next_right = self.advance_foot(right_foot, angle_to_target, self.step_length)

        return next_left, next_right

    def advance_foot(self, current_foot, direction, step_length):
        """
        Advance a foot in the specified direction
        """
        new_x = current_foot.x + step_length * np.cos(direction)
        new_y = current_foot.y + step_length * np.sin(direction)

        # Maintain foot spacing perpendicular to direction of travel
        perp_offset_x = -np.sin(direction) * self.foot_spacing / 2
        perp_offset_y = np.cos(direction) * self.foot_spacing / 2

        # Alternate foot placement for stability
        new_pose = type('Pose', (), {'x': new_x, 'y': new_y})()
        return new_pose

    def is_valid_footstep(self, pose):
        """
        Check if footstep is valid (not in obstacle, within bounds)
        """
        # Implementation would check costmap for obstacles
        # and ensure footstep is within navigation bounds
        return True

    def distance_to_path(self, pose, path):
        """
        Calculate minimum distance from foot position to path
        """
        min_dist = float('inf')
        for point in path:
            dist = euclidean([pose.x, pose.y], [point.x, point.y])
            min_dist = min(min_dist, dist)
        return min_dist

# Integration with Nav2
class HumanoidFootstepIntegration:
    def __init__(self):
        self.footstep_planner = FootstepPlanner({
            'foot_spacing': 0.2,
            'step_length': 0.3,
            'max_step_deviation': 0.1
        })

    def integrate_with_nav2(self, global_plan):
        """
        Integrate footstep planning with Nav2's global planner
        """
        # Convert Nav2 plan to detailed footstep plan
        footstep_sequence = self.footstep_planner.plan_footsteps(global_plan, start_pose=None)

        # Return both high-level path and detailed footsteps
        return {
            'nav2_path': global_plan,
            'footsteps': footstep_sequence,
            'gait_commands': self.convert_to_gait_commands(footstep_sequence)
        }

    def convert_to_gait_commands(self, footstep_sequence):
        """
        Convert footstep sequence to gait execution commands
        """
        gait_commands = []
        for foot, pose in footstep_sequence:
            command = {
                'type': 'move_foot',
                'foot': foot,
                'target_pose': pose,
                'timing': self.calculate_timing(pose)  # Ensure stable transfer
            }
            gait_commands.append(command)

        return gait_commands

    def calculate_timing(self, target_pose):
        """
        Calculate appropriate timing for foot movement to maintain balance
        """
        # Timing based on balance and stability constraints
        # Would include control theory calculations for bipedal gait
        return 0.5  # Half second per step (adjustable)
```

## Integration with Isaac and ROS 2

### Isaac ROS Navigation Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32

class IsaacNavIntegration(Node):
    def __init__(self):
        super().__init__('isaac_nav_integration')

        # Navigation system initialization
        self.nav_client = self.create_client(NavigateToPose, '/navigate_to_pose')

        # Sensor subscriptions for navigation
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Humanoid-specific publishers
        self.gait_cmd_pub = self.create_publisher(Twist, '/gait_commands', 10)
        self.balance_pub = self.create_publisher(Float32, '/balance_score', 10)

        # Footstep planning component
        self.footstep_planner = FootstepPlanner({
            'foot_spacing': 0.2,
            'step_length': 0.3,
            'max_step_deviation': 0.1
        })

        # Navigation timer
        self.nav_timer = self.create_timer(0.1, self.navigation_control_loop)

        self.current_pose = None
        self.current_velocity = None
        self.balance_score = 0.0

    def scan_callback(self, msg):
        """Process LiDAR data for navigation"""
        # Update local costmap with scan data
        self.update_local_costmap(msg)

    def imu_callback(self, msg):
        """Process IMU data for balance monitoring"""
        # Calculate balance score from IMU data
        self.balance_score = self.calculate_balance_score(msg)
        self.balance_pub.publish(Float32(data=self.balance_score))

    def odom_callback(self, msg):
        """Process odometry data for navigation"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def navigation_control_loop(self):
        """Main navigation control loop"""
        if self.current_pose is not None and self.is_safe_to_navigate():
            # Check if robot is balanced
            if self.balance_score > 0.7:  # Acceptable balance threshold
                # Execute navigation command
                cmd = self.compute_navigation_command()
                self.gait_cmd_pub.publish(cmd)
            else:
                # Stop navigation until balance is restored
                self.stop_navigation()
        else:
            self.stop_navigation()

    def compute_navigation_command(self):
        """Compute navigation command considering humanoid constraints"""
        # This would integrate with Nav2 for path planning
        # and generate gait-appropriate commands
        twist_cmd = Twist()

        # Get next waypoint from Nav2
        next_waypoint = self.get_next_waypoint()

        # Calculate appropriate velocity based on terrain and balance
        if next_waypoint:
            twist_cmd = self.calculate_gait_command(next_waypoint)

        return twist_cmd

    def calculate_gait_command(self, waypoint):
        """Calculate gait-appropriate movement command"""
        twist = Twist()

        # Calculate direction and distance to waypoint
        dx = waypoint.position.x - self.current_pose.position.x
        dy = waypoint.position.y - self.current_pose.position.y
        distance = (dx*dx + dy*dy)**0.5

        # Humanoid-appropriate speeds
        if distance > 0.1:  # Threshold for movement
            # Calculate linear velocity based on path and safety
            twist.linear.x = min(0.2, distance * 0.5)  # Conservative max speed

            # Calculate angular velocity for direction
            target_angle = np.arctan2(dy, dx)
            current_yaw = self.quaternion_to_yaw(self.current_pose.orientation)
            angle_diff = target_angle - current_yaw

            # Normalize angle difference
            while angle_diff > np.pi:
                angle_diff -= 2*np.pi
            while angle_diff < -np.pi:
                angle_diff += 2*np.pi

            twist.angular.z = np.clip(angle_diff * 0.5, -0.3, 0.3)

        return twist

    def is_safe_to_navigate(self):
        """Check if navigation is safe considering humanoid constraints"""
        # Check for sufficient balance
        if self.balance_score < 0.5:
            return False

        # Check for safe obstacles distances
        if hasattr(self, 'local_map') and not self.is_path_clear():
            return False

        # Check for other safety conditions
        return True

    def stop_navigation(self):
        """Safely stop navigation for balance"""
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = 0.0
        self.gait_cmd_pub.publish(stop_cmd)

def main(args=None):
    rclpy.init(args=args)

    nav_node = IsaacNavIntegration()

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Navigation Features

### Dynamic Obstacle Avoidance

For humanoid robots operating in populated environments:

```yaml
# Dynamic obstacle parameters
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["voxel_layer", "dynamic_layer", "inflation_layer"]
      dynamic_layer:
        plugin: "nav2_dynamic_obstacle_layer::DynamicObstacleLayer"
        enabled: true
        observation_sources: people_tracker
        people_tracker:
          topic: /people_tracked
          max_obstacle_height: 1.8
          clearing: false
          marking: true
          data_type: "TrackedPeople"
          obstacle_max_range: 5.0
          obstacle_min_range: 0.5
          # Prediction parameters for moving people
          prediction_horizon: 2.0  # seconds
          velocity_uncertainty: 0.5
```

### Multi-robot Navigation

For humanoid robots navigating in teams:

```cpp
// Multi-robot collision avoidance
#include "nav2_core/costmap_model.hpp"
#include "geometry_msgs/msg/pose_array.hpp"

class MultiRobotNav2: public nav2_core::Controller
{
public:
    MultiRobotNav2() {
        // Subscribe to other robots' positions
        other_robot_sub_ = node_->create_subscription<geometry_msgs::msg::PoseArray>(
            "/other_robots/poses", 10,
            std::bind(&MultiRobotNav2::otherRobotsCallback, this, std::placeholders::_1));
    }

private:
    void otherRobotsCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        other_robot_poses_ = *msg;
        updateMultiRobotCostmap();
    }

    void updateMultiRobotCostmap()
    {
        // Add other robots as temporary obstacles in local costmap
        for (const auto& pose : other_robot_poses_.poses) {
            addTemporaryObstacle(pose, robot_radius_);
        }
    }

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr other_robot_sub_;
    geometry_msgs::msg::PoseArray other_robot_poses_;
    double robot_radius_{0.3}; // Humanoid robot radius
};
```

## Best Practices

### Performance Optimization

1. **Map Resolution**: Balance resolution with computational cost
2. **Update Frequency**: Optimize costmap and planning frequencies for real-time operation
3. **Recovery Behaviors**: Configure appropriate recovery strategies
4. **Safety Margins**: Set conservative safety margins for humanoid stability

### Tuning Guidelines

1. **Start Conservative**: Begin with slow speeds and large safety margins
2. **Gradual Adjustment**: Incrementally improve performance parameters
3. **Testing**: Test in simulation before real-world deployment
4. **Monitoring**: Continuously monitor navigation performance metrics

## Troubleshooting

### Common Navigation Issues

**Issue**: Robot oscillates when approaching goal
- Solution: Adjust goal tolerance and controller parameters

**Issue**: Robot gets stuck in local minima
- Solution: Improve global planner or add random walk recovery

**Issue**: Navigation is too slow
- Solution: Increase velocities within safety limits, optimize algorithms

**Issue**: Robot falls over during navigation
- Solution: Reduce speed, increase support margins, improve balance control

## Summary

Navigation for humanoid robots requires special considerations due to their complex dynamics, balance requirements, and bipedal locomotion patterns. The Nav2 framework provides a robust foundation that can be customized with humanoid-specific controllers, footstep planning, and balance monitoring. By properly configuring the navigation stack and implementing appropriate safety measures, humanoid robots can navigate complex environments safely and efficiently.