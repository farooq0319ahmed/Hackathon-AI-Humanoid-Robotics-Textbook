---
sidebar_position: 5
---

# Hands-On Exercises: AI-Based Robot Perception and Navigation

## Exercise 1: Visual SLAM Pipeline Implementation

### Objective
Implement a complete Visual SLAM pipeline using NVIDIA Isaac tools and validate its performance in mapping and localization.

### Prerequisites
- NVIDIA GPU with CUDA support
- Isaac Sim installed
- ROS 2 Humble Hawksbill
- OpenCV and ORB-SLAM2 dependencies

### Exercise Steps

1. **Setup Isaac Sim Environment**
   ```bash
   # Launch Isaac Sim with a sample environment
   cd /opt/isaac_sim
   python3 -m omni.isaac.sim.python_app --config=standalone --enable-gui

   # Create a new stage with a simple room environment
   # Add landmarks and objects for feature tracking
   ```

2. **Create SLAM Node**
   ```python
   # Create a ROS 2 package for SLAM
   mkdir -p ~/isaac_ws/src/slam_package
   cd ~/isaac_ws/src/slam_package
   ros2 pkg create --build-type ament_python slam_ros

   # Create the SLAM node implementation
   # This should include:
   # - Camera image subscription
   # - Feature extraction and tracking
   # - Pose estimation
   # - Map building and publishing
   ```

3. **Implement ORB-SLAM2 Integration**
   ```cpp
   // In slam_ros/src/orb_slam_node.cpp
   #include <ORB_SLAM2.h>
   #include "rclcpp/rclcpp.hpp"
   #include "sensor_msgs/msg/image.hpp"
   #include "cv_bridge/cv_bridge.h"

   class OrbSlamNode : public rclcpp::Node
   {
   public:
       OrbSlamNode() : Node("orb_slam_node")
       {
           // Initialize ORB-SLAM2 system
           slam_system_ = std::make_unique<ORB_SLAM2::System>(
               "path/to/vocabulary/orbvoc.bin",
               "path/to/settings/monocular.yaml",
               ORB_SLAM2::System::MONOCULAR,
               true
           );

           // Create camera subscription
           camera_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
               "/camera/rgb/image_raw",
               10,
               std::bind(&OrbSlamNode::imageCallback, this, std::placeholders::_1)
           );

           // Create pose publisher
           pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
               "/slam/pose", 10
           );
       }

   private:
       void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
       {
           cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");

           // Process image through SLAM system
           cv::Mat pose = slam_system_->TrackMonocular(
               cv_ptr->image,
               msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9
           );

           if (!pose.empty()) {
               // Publish estimated pose
               publishPose(pose, msg->header);
           }
       }

       void publishPose(const cv::Mat& pose, const std_msgs::msg::Header& header)
       {
           geometry_msgs::msg::PoseStamped pose_msg;
           pose_msg.header = header;

           // Convert SLAM pose to ROS pose
           pose_msg.pose.position.x = pose.at<float>(0, 3);
           pose_msg.pose.position.y = pose.at<float>(1, 3);
           pose_msg.pose.position.z = pose.at<float>(2, 3);

           // Convert rotation matrix to quaternion
           cv::Mat R = pose.rowRange(0, 3).colRange(0, 3);
           tf2::Matrix3x3 tf3d(
               tf2::Vector3(R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2)),
               tf2::Vector3(R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2)),
               tf2::Vector3(R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2))
           );
           tf3d.getRotation(pose_msg.pose.orientation);

           pose_pub_->publish(pose_msg);
       }

       std::unique_ptr<ORB_SLAM2::System> slam_system_;
       rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub_;
       rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
   };
   ```

4. **Create Launch File**
   ```xml
   <!-- slam_package/launch/slam_pipeline.launch.xml -->
   <launch>
       <!-- Start the SLAM node -->
       <node pkg="slam_package" exec="orb_slam_node" name="orb_slam_node">
           <param name="vocabulary_path" value="path/to/vocabulary/orbvoc.bin"/>
           <param name="settings_path" value="path/to/settings/monocular.yaml"/>
       </node>

       <!-- Start RViz for visualization -->
       <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(find-pkg-share slam_package)/rviz/slam.rviz"/>
   </launch>
   ```

5. **Validation and Testing**
   - Navigate the robot through the environment
   - Monitor the map building process in RViz
   - Check for loop closure and drift correction
   - Measure localization accuracy against ground truth

### Expected Outcomes
- SLAM system successfully builds a map of the environment
- Robot pose is accurately estimated and published
- Loop closure is detected and drift is corrected
- Map quality metrics meet the specified requirements

## Exercise 2: Nav2 Path Planning for Humanoids

### Objective
Configure and test Nav2 specifically for humanoid robot navigation, including footstep planning and balance considerations.

### Prerequisites
- ROS 2 Humble with Nav2 installed
- Humanoid robot model in Gazebo
- Navigation2 packages

### Exercise Steps

1. **Create Humanoid Navigation Configuration**
   ```yaml
   # config/humanoid_nav2_config.yaml
   bt_navigator:
     ros__parameters:
       use_sim_time: True
       global_frame: map
       robot_base_frame: base_link
       odom_topic: /odom
       bt_loop_duration: 10
       default_server_timeout: 20
       enable_groot_monitoring: True
       groot_zmq_publisher_port: 1666
       groot_zmq_server_port: 1667
       default_nav_through_poses_bt_xml: "nav2_bt_xml_v0.14/navigate_w_replanning_and_recovery.xml"
       default_nav_to_pose_bt_xml: "nav2_bt_xml_v0.14/navigate_w_replanning_and_recovery.xml"
       plugin_lib_names:
         - nav2_compute_path_to_pose_action_bt_node
         - nav2_compute_path_through_poses_action_bt_node
         - nav2_smooth_path_action_bt_node
         - nav2_follow_path_action_bt_node
         - nav2_spin_action_bt_node
         - nav2_wait_action_bt_node
         - nav2_assisted_teleop_action_bt_node
         - nav2_back_up_action_bt_node
         - nav2_drive_on_heading_bt_node
         - nav2_clear_costmap_service_bt_node
         - nav2_is_stuck_condition_bt_node
         - nav2_goal_reached_condition_bt_node
         - nav2_goal_updated_condition_bt_node
         - nav2_globally_updated_goal_condition_bt_node
         - nav2_is_path_valid_condition_bt_node
         - nav2_initial_pose_received_condition_bt_node
         - nav2_reinitialize_global_localization_service_bt_node
         - nav2_rate_controller_bt_node
         - nav2_distance_controller_bt_node
         - nav2_speed_controller_bt_node
         - nav2_truncate_path_action_bt_node
         - nav2_truncate_path_local_action_bt_node
         - nav2_goal_updater_node_bt_node
         - nav2_recovery_node_bt_node
         - nav2_pipeline_sequence_bt_node
         - nav2_round_robin_node_bt_node
         - nav2_transform_available_condition_bt_node
         - nav2_time_expired_condition_bt_node
         - nav2_path_expiring_timer_condition
         - nav2_distance_traveled_condition_bt_node
         - nav2_single_trigger_bt_node
         - nav2_is_battery_low_condition_bt_node
         - nav2_navigate_through_poses_action_bt_node
         - nav2_navigate_to_pose_action_bt_node
         - nav2_remove_passed_goals_action_bt_node
         - nav2_planner_selector_bt_node
         - nav2_controller_selector_bt_node
         - nav2_goal_checker_selector_bt_node
         - nav2_controller_cancel_bt_node
         - nav2_path_longer_on_approach_bt_node
         - nav2_wait_cancel_bt_node
         - nav2_spin_cancel_bt_node
         - nav2_back_up_cancel_bt_node
         - nav2_assisted_teleop_cancel_bt_node
         - nav2_drive_on_heading_cancel_bt_node

   controller_server:
     ros__parameters:
       use_sim_time: True
       controller_frequency: 20.0
       min_x_velocity_threshold: 0.001
       min_y_velocity_threshold: 0.5
       min_theta_velocity_threshold: 0.001
       failure_tolerance: 0.3
       progress_checker_plugin: "progress_checker"
       goal_checker_plugins: ["general_goal_checker"]
       controller_plugins: ["FollowPath"]

       # Humanoid-specific controller
       FollowPath:
         plugin: "nav2_mppi_controller::MPPIController"
         time_steps: 50
         model_dt: 0.05
         batch_size: 2000
         vx_std: 0.2
         vy_std: 0.1
         wz_std: 0.3
         vx_max: 0.8
         vx_min: -0.3
         vy_max: 0.3
         vy_min: -0.3
         wz_max: 1.0
         wz_min: -1.0
         xy_goal_tolerance: 0.25
         yaw_goal_tolerance: 0.25
         state_reset_tol: 0.5
         forward_penalty: 1.0
         angular_velocity_penalty: 0.4
         goal_angle_penalty: 2.0
         xy_window: 0.0
         xy_window_size: 0.0
         trajectory_visualization: true

   local_costmap:
     local_costmap:
       ros__parameters:
         update_frequency: 5.0
         publish_frequency: 2.0
         global_frame: odom
         robot_base_frame: base_link
         use_sim_time: True
         rolling_window: true
         width: 6
         height: 6
         resolution: 0.05
         robot_radius: 0.3  # Humanoid-specific radius
         plugins: ["voxel_layer", "inflation_layer"]
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
         voxel_layer:
           plugin: "nav2_costmap_2d::VoxelLayer"
           enabled: True
           publish_voxel_map: True
           origin_z: 0.0
           z_resolution: 0.05
           z_voxels: 16
           max_obstacle_height: 2.0
         static_layer:
           plugin: "nav2_costmap_2d::StaticLayer"
           map_subscribe_transient_local: True
         always_send_full_costmap: True

   global_costmap:
     global_costmap:
       ros__parameters:
         update_frequency: 1.0
         publish_frequency: 1.0
         global_frame: map
         robot_base_frame: base_link
         use_sim_time: True
         robot_radius: 0.3  # Humanoid-specific radius
         resolution: 0.05
         track_unknown_space: true
         plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
         obstacle_layer:
           plugin: "nav2_costmap_2d::ObstacleLayer"
           enabled: True
           observation_sources: scan
           scan:
             topic: /scan
             max_obstacle_height: 2.0
             clearing: True
             marking: True
             data_type: "LaserScan"
         static_layer:
           plugin: "nav2_costmap_2d::StaticLayer"
           map_subscribe_transient_local: True
         inflation_layer:
           plugin: "nav2_costmap_2d::InflationLayer"
           cost_scaling_factor: 3.0
           inflation_radius: 0.55
   ```

2. **Create Footstep Planner Node**
   ```python
   # Create a ROS 2 node for footstep planning
   # footstep_planner.py
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped
   from nav_msgs.msg import Path
   import numpy as np

   class FootstepPlannerNode(Node):
       def __init__(self):
           super().__init__('footstep_planner_node')

           # Subscribe to navigation path
           self.path_sub = self.create_subscription(
               Path, '/plan', self.path_callback, 10
           )

           # Publish footstep plan
           self.footstep_pub = self.create_publisher(
               Path, '/footstep_plan', 10
           )

           # Humanoid parameters
           self.step_width = 0.2  # Distance between feet
           self.step_length = 0.3  # Step length
           self.step_height = 0.1  # Step height for obstacles

       def path_callback(self, path_msg):
           """Generate footstep plan from navigation path"""
           footstep_path = Path()
           footstep_path.header = path_msg.header

           # Generate footsteps along the path
           footsteps = self.generate_footsteps(path_msg.poses)

           for footstep in footsteps:
               pose_stamped = PoseStamped()
               pose_stamped.header = path_msg.header
               pose_stamped.pose = footstep
               footstep_path.poses.append(pose_stamped)

           self.footstep_pub.publish(footstep_path)

       def generate_footsteps(self, poses):
           """Generate footstep plan from path poses"""
           footsteps = []

           # Alternate between left and right foot
           left_foot = True

           for i, pose_stamped in enumerate(poses):
               # Calculate foot position based on step pattern
               foot_pose = self.calculate_foot_position(
                   pose_stamped.pose, left_foot
               )
               footsteps.append(foot_pose)

               # Alternate feet
               left_foot = not left_foot

           return footsteps

       def calculate_foot_position(self, base_pose, is_left_foot):
           """Calculate foot position from base pose"""
           foot_pose = base_pose

           if is_left_foot:
               # Offset to left foot position
               foot_pose.position.y += self.step_width / 2
           else:
               # Offset to right foot position
               foot_pose.position.y -= self.step_width / 2

           return foot_pose
   ```

3. **Launch Navigation System**
   ```bash
   # Create launch file for complete navigation
   # launch/humanoid_navigation.launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os

   def generate_launch_description():
       config_dir = os.path.join(
           get_package_share_directory('slam_package'),
           'config'
       )

       return LaunchDescription([
           # Launch Nav2 stack
           Node(
               package='nav2_controller',
               executable='controller_server',
               parameters=[os.path.join(config_dir, 'humanoid_nav2_config.yaml')],
               output='screen'
           ),

           Node(
               package='nav2_planner',
               executable='planner_server',
               parameters=[os.path.join(config_dir, 'humanoid_nav2_config.yaml')],
               output='screen'
           ),

           Node(
               package='nav2_recoveries',
               executable='recoveries_server',
               parameters=[os.path.join(config_dir, 'humanoid_nav2_config.yaml')],
               output='screen'
           ),

           Node(
               package='nav2_bt_navigator',
               executable='bt_navigator',
               parameters=[os.path.join(config_dir, 'humanoid_nav2_config.yaml')],
               output='screen'
           ),

           Node(
               package='nav2_lifecycle_manager',
               executable='lifecycle_manager',
               name='lifecycle_manager_navigation',
               output='screen',
               parameters=[{'use_sim_time': True},
                          {'autostart': True},
                          {'node_names': ['controller_server',
                                        'planner_server',
                                        'recoveries_server',
                                        'bt_navigator']}]
           ),

           # Launch footstep planner
           Node(
               package='slam_package',
               executable='footstep_planner',
               name='footstep_planner',
               parameters=[{'step_width': 0.2, 'step_length': 0.3}],
               output='screen'
           )
       ])
   ```

4. **Test Navigation**
   ```bash
   # Launch the complete navigation system
   ros2 launch slam_package humanoid_navigation.launch.py

   # Send navigation goal
   ros2 run nav2_msgs goal_pose [x] [y] [theta]

   # Monitor footstep planning in RViz
   ```

### Expected Outcomes
- Navigation system plans paths considering humanoid dimensions
- Footstep planner generates appropriate stepping patterns
- Robot successfully navigates to goal while maintaining balance
- Recovery behaviors work correctly for humanoid-specific failures

## Exercise 3: Reinforcement Learning for Humanoid Locomotion

### Objective
Train a reinforcement learning policy for humanoid walking using NVIDIA Isaac Gym and deploy it on a simulated humanoid.

### Prerequisites
- NVIDIA Isaac Gym installed
- PyTorch and reinforcement learning libraries
- CUDA-enabled GPU

### Exercise Steps

1. **Create Isaac Gym Environment**
   ```python
   # Create humanoid environment for RL training
   # isaac_gym_env.py
   import isaacgym
   from isaacgym import gymapi, gymtorch
   import torch
   import numpy as np

   class HumanoidLocomotionEnv:
       def __init__(self, cfg):
           # Initialize Isaac Gym
           self.gym = gymapi.acquire_gym()

           # Create simulation
           self.sim = self.gym.create_sim(
               device_id=0,
               graphics_device_id=0,
               physics_engine=isaacgym.gymapi.SIM_PHYSX,
               sim_params=cfg.sim_params
           )

           # Create ground plane
           plane_params = gymapi.PlaneParams()
           plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
           self.gym.add_ground(self.sim, plane_params)

           # Load humanoid asset
           asset_options = gymapi.AssetOptions()
           asset_options.fix_base_link = False
           asset_options.disable_gravity = False
           asset_options.thickness = 0.01

           self.humanoid_asset = self.gym.load_asset(
               self.sim,
               cfg.asset_root,
               cfg.humanoid_asset_file,
               asset_options
           )

           # Create environments
           self.envs = []
           self.setup_environments(cfg)

           # Initialize tensors
           self.setup_tensors(cfg)

       def setup_environments(self, cfg):
           """Create multiple parallel environments"""
           spacing = cfg.env_spacing
           env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
           env_upper = gymapi.Vec3(spacing, spacing, spacing)

           for i in range(cfg.num_envs):
               env = self.gym.create_env(
                   self.sim, env_lower, env_upper, 1
               )

               # Add humanoid to environment
               humanoid_pose = gymapi.Transform()
               humanoid_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
               humanoid_handle = self.gym.create_actor(
                   env, self.humanoid_asset, humanoid_pose, "humanoid", i, 0
               )

               # Set DOF properties
               dof_props = self.gym.get_actor_dof_properties(env, humanoid_handle)
               dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
               dof_props["stiffness"] = cfg.joint_stiffness
               dof_props["damping"] = cfg.joint_damping
               self.gym.set_actor_dof_properties(env, humanoid_handle, dof_props)

               self.envs.append(env)

       def reset(self):
           """Reset all environments and return initial observations"""
           # Reset robot states to initial positions
           # Return observation tensors
           pass

       def step(self, actions):
           """Execute actions and return observations, rewards, dones"""
           # Apply actions to robots
           # Compute rewards based on locomotion success
           # Return (observations, rewards, dones, info)
           pass
   ```

2. **Implement PPO Algorithm**
   ```python
   # Create PPO implementation for humanoid control
   # ppo_agent.py
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.distributions import Normal

   class HumanoidPPOAgent:
       def __init__(self, state_dim, action_dim, cfg):
           self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

           self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
           self.critic = CriticNetwork(state_dim).to(self.device)

           self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
           self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

           self.clip_epsilon = cfg.clip_epsilon
           self.entropy_coef = cfg.entropy_coef

       def select_action(self, state):
           """Select action using current policy"""
           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

           with torch.no_grad():
               mu, std = self.actor(state_tensor)
               dist = Normal(mu, std)
               action = dist.sample()
               log_prob = dist.log_prob(action).sum(dim=-1)

           return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

       def update(self, states, actions, old_log_probs, returns, advantages):
           """Update policy using PPO objective"""
           states = torch.FloatTensor(states).to(self.device)
           actions = torch.FloatTensor(actions).to(self.device)
           old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
           returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
           advantages = torch.FloatTensor(advantages).to(self.device)

           # Actor update
           mu, std = self.actor(states)
           dist = Normal(mu, std)
           new_log_probs = dist.log_prob(actions).sum(dim=-1)

           ratio = torch.exp(new_log_probs - old_log_probs)
           surr1 = ratio * advantages
           surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
           actor_loss = -torch.min(surr1, surr2).mean()

           # Add entropy regularization
           entropy = dist.entropy().sum(dim=-1).mean()
           actor_loss -= self.entropy_coef * entropy

           self.optimizer_actor.zero_grad()
           actor_loss.backward()
           self.optimizer_actor.step()

           # Critic update
           values = self.critic(states)
           critic_loss = (returns - values).pow(2).mean()

           self.optimizer_critic.zero_grad()
           critic_loss.backward()
           self.optimizer_critic.step()

   class ActorNetwork(nn.Module):
       def __init__(self, state_dim, action_dim):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(state_dim, 512),
               nn.ReLU(),
               nn.Linear(512, 512),
               nn.ReLU(),
               nn.Linear(512, 256),
               nn.ReLU()
           )
           self.mu_head = nn.Linear(256, action_dim)
           self.std_head = nn.Linear(256, action_dim)

       def forward(self, state):
           features = self.network(state)
           mu = torch.tanh(self.mu_head(features))
           std = torch.ones_like(mu) * 0.5
           return mu, std

   class CriticNetwork(nn.Module):
       def __init__(self, state_dim):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(state_dim, 512),
               nn.ReLU(),
               nn.Linear(512, 512),
               nn.ReLU(),
               nn.Linear(512, 256),
               nn.ReLU(),
               nn.Linear(256, 1)
           )

       def forward(self, state):
           return self.network(state)
   ```

3. **Train the Policy**
   ```python
   # Training script
   # train_humanoid.py
   import numpy as np
   from collections import deque
   import torch

   def train_humanoid_policy():
       # Initialize environment and agent
       env = HumanoidLocomotionEnv(cfg)
       agent = HumanoidPPOAgent(state_dim=cfg.state_dim, action_dim=cfg.action_dim, cfg=cfg)

       # Training loop
       for episode in range(cfg.max_episodes):
           states = []
           actions = []
           rewards = []
           log_probs = []

           state = env.reset()
           done = False

           while not done:
               action, log_prob = agent.select_action(state)

               next_state, reward, done, info = env.step(action)

               states.append(state)
               actions.append(action)
               rewards.append(reward)
               log_probs.append(log_prob)

               state = next_state

           # Compute returns and advantages
           returns = compute_returns(rewards, cfg.gamma)
           advantages = compute_advantages(rewards, values, cfg.gamma, cfg.lambda_)

           # Update agent
           agent.update(states, actions, log_probs, returns, advantages)

           # Log training progress
           if episode % cfg.log_interval == 0:
               avg_reward = np.mean(rewards)
               print(f"Episode {episode}, Average Reward: {avg_reward}")

   def compute_returns(rewards, gamma):
       """Compute discounted returns"""
       returns = []
       R = 0
       for r in reversed(rewards):
           R = r + gamma * R
           returns.insert(0, R)
       return returns

   def compute_advantages(rewards, values, gamma, lambda_):
       """Compute advantages using Generalized Advantage Estimation"""
       advantages = []
       gae = 0
       for i in reversed(range(len(rewards))):
           if i == len(rewards) - 1:
               next_value = 0
           else:
               next_value = values[i + 1]

           delta = rewards[i] + gamma * next_value - values[i]
           gae = delta + gamma * lambda_ * gae
           advantages.insert(0, gae)

       return advantages
   ```

4. **Deploy Trained Policy**
   ```python
   # Create ROS node for deploying trained policy
   # deployment_node.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import JointState, Imu
   from std_msgs.msg import Float32MultiArray
   import torch

   class HumanoidPolicyDeployment(Node):
       def __init__(self):
           super().__init__('humanoid_policy_deployment')

           # Load trained policy
           self.policy = torch.jit.load('trained_policy.pt')
           self.policy.eval()

           # Create subscribers for sensor data
           self.joint_state_sub = self.create_subscription(
               JointState, '/joint_states', self.joint_state_callback, 10
           )
           self.imu_sub = self.create_subscription(
               Imu, '/imu/data', self.imu_callback, 10
           )

           # Create publisher for joint commands
           self.joint_cmd_pub = self.create_publisher(
               Float32MultiArray, '/joint_commands', 10
           )

           # Timer for control loop
           self.control_timer = self.create_timer(0.02, self.control_loop)  # 50 Hz

           # State buffers
           self.joint_positions = np.zeros(28)
           self.joint_velocities = np.zeros(28)
           self.imu_data = np.zeros(6)

       def joint_state_callback(self, msg):
           """Update joint state buffer"""
           self.joint_positions = np.array(msg.position)
           self.joint_velocities = np.array(msg.velocity)

       def imu_callback(self, msg):
           """Update IMU data buffer"""
           self.imu_data[:4] = [msg.orientation.x, msg.orientation.y,
                               msg.orientation.z, msg.orientation.w]
           self.imu_data[4:] = [msg.angular_velocity.x, msg.angular_velocity.y,
                               msg.angular_velocity.z]

       def get_observation(self):
           """Construct observation from sensor data"""
           obs = np.concatenate([
               self.joint_positions,
               self.joint_velocities,
               self.imu_data
           ])
           return obs

       def control_loop(self):
           """Main control loop"""
           obs = self.get_observation()
           obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

           with torch.no_grad():
               action = self.policy(obs_tensor).squeeze(0).numpy()

           # Publish action as joint commands
           cmd_msg = Float32MultiArray()
           cmd_msg.data = action.tolist()
           self.joint_cmd_pub.publish(cmd_msg)
   ```

### Expected Outcomes
- RL policy successfully learns humanoid walking behavior
- Policy achieves stable locomotion in simulation
- Trained policy can be deployed on the simulated robot
- Robot demonstrates learned walking behavior in various terrains

## Exercise 4: Perception Pipeline Integration

### Objective
Integrate multiple perception modules into a cohesive pipeline that processes sensor data and provides actionable information for robot decision-making.

### Prerequisites
- ROS 2 Humble with perception packages
- Camera and LiDAR sensors in simulation
- OpenCV and PCL libraries

### Exercise Steps

1. **Create Perception Node**
   ```python
   # perception_pipeline.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan, PointCloud2, Imu
   from geometry_msgs.msg import PoseStamped, Point
   from std_msgs.msg import String
   import cv2
   from cv_bridge import CvBridge
   import numpy as np
   import sensor_msgs.point_cloud2 as pc2
   from sklearn.cluster import DBSCAN

   class PerceptionPipeline(Node):
       def __init__(self):
           super().__init__('perception_pipeline')

           self.cv_bridge = CvBridge()

           # Create subscribers
           self.image_sub = self.create_subscription(
               Image, '/camera/rgb/image_raw', self.image_callback, 10
           )
           self.lidar_sub = self.create_subscription(
               LaserScan, '/scan', self.lidar_callback, 10
           )
           self.imu_sub = self.create_subscription(
               Imu, '/imu/data', self.imu_callback, 10
           )

           # Create publishers
           self.object_pub = self.create_publisher(
               String, '/detected_objects', 10
           )
           self.obstacle_pub = self.create_publisher(
               Point, '/nearest_obstacle', 10
           )
           self.perception_status_pub = self.create_publisher(
               String, '/perception_status', 10
           )

           # Perception components
           self.object_detector = ObjectDetector()
           self.obstacle_detector = ObstacleDetector()
           self.tracker = ObjectTracker()

       def image_callback(self, msg):
           """Process camera image for object detection"""
           cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

           # Detect objects in image
           detections = self.object_detector.detect(cv_image)

           # Update tracker with new detections
           tracked_objects = self.tracker.update(detections)

           # Publish detected objects
           self.publish_objects(tracked_objects)

       def lidar_callback(self, msg):
           """Process LiDAR data for obstacle detection"""
           # Convert scan to point cloud
           ranges = np.array(msg.ranges)
           angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

           # Filter valid measurements
           valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
           valid_ranges = ranges[valid_mask]
           valid_angles = angles[valid_mask]

           # Convert to Cartesian coordinates
           x_coords = valid_ranges * np.cos(valid_angles)
           y_coords = valid_ranges * np.sin(valid_angles)

           point_cloud_2d = np.column_stack((x_coords, y_coords))

           # Detect obstacles
           obstacles = self.obstacle_detector.detect(point_cloud_2d)

           # Publish nearest obstacle
           if obstacles:
               nearest_obstacle = min(obstacles, key=lambda o: np.linalg.norm(o))
               obstacle_msg = Point()
               obstacle_msg.x = nearest_obstacle[0]
               obstacle_msg.y = nearest_obstacle[1]
               obstacle_msg.z = 0.0
               self.obstacle_pub.publish(obstacle_msg)

       def imu_callback(self, msg):
           """Process IMU data for orientation"""
           # Use IMU data for sensor fusion
           pass

       def publish_objects(self, objects):
           """Publish detected objects"""
           objects_str = ", ".join([obj.class_name for obj in objects])
           status_msg = String()
           status_msg.data = f"Detected: {objects_str}"
           self.perception_status_pub.publish(status_msg)
   ```

2. **Implement Object Detection**
   ```python
   # object_detector.py
   import cv2
   import numpy as np
   from dataclasses import dataclass

   @dataclass
   class Detection:
       class_name: str
       confidence: float
       bbox: tuple  # (x, y, w, h)
       center: tuple  # (x, y)

   class ObjectDetector:
       def __init__(self):
           # Initialize detection model (could be YOLO, SSD, etc.)
           # For this exercise, we'll use simple color-based detection
           pass

       def detect(self, image):
           """Detect objects in image"""
           detections = []

           # Convert BGR to HSV for color detection
           hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

           # Define color ranges for different objects
           color_ranges = {
               'red': ([0, 50, 50], [10, 255, 255]),
               'blue': ([100, 50, 50], [130, 255, 255]),
               'green': ([40, 50, 50], [80, 255, 255])
           }

           for class_name, (lower, upper) in color_ranges.items():
               lower = np.array(lower, dtype="uint8")
               upper = np.array(upper, dtype="uint8")

               # Create mask for color range
               mask = cv2.inRange(hsv, lower, upper)

               # Find contours
               contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

               for contour in contours:
                   if cv2.contourArea(contour) > 500:  # Filter small contours
                       x, y, w, h = cv2.boundingRect(contour)
                       center_x = x + w // 2
                       center_y = y + h // 2

                       detection = Detection(
                           class_name=class_name,
                           confidence=0.8,  # Fixed confidence for this example
                           bbox=(x, y, w, h),
                           center=(center_x, center_y)
                       )
                       detections.append(detection)

           return detections
   ```

3. **Create Object Tracker**
   ```python
   # object_tracker.py
   import numpy as np
   from scipy.optimize import linear_sum_assignment
   from collections import defaultdict

   class TrackedObject:
       def __init__(self, detection, track_id):
           self.id = track_id
           self.class_name = detection.class_name
           self.bbox = detection.bbox
           self.center = detection.center
           self.confidence = detection.confidence
           self.age = 0
           self.hits = 1
           self.hit_streak = 1

   class ObjectTracker:
       def __init__(self):
           self.tracks = []
           self.next_id = 0

       def update(self, detections):
           """Update object tracks with new detections"""
           if not self.tracks:
               # Create initial tracks
               for det in detections:
                   track = TrackedObject(det, self.next_id)
                   self.tracks.append(track)
                   self.next_id += 1
               return self.tracks

           # Calculate distance matrix between tracks and detections
           dist_matrix = self.calculate_distance_matrix(self.tracks, detections)

           # Associate tracks with detections
           track_indices, detection_indices = linear_sum_assignment(dist_matrix)

           # Update matched tracks
           matched_tracks = set()
           for track_idx, det_idx in zip(track_indices, detection_indices):
               if dist_matrix[track_idx, det_idx] < 50:  # Distance threshold
                   track = self.tracks[track_idx]
                   det = detections[det_idx]
                   track.bbox = det.bbox
                   track.center = det.center
                   track.confidence = det.confidence
                   track.hits += 1
                   track.hit_streak += 1
                   matched_tracks.add(track_idx)

           # Add new tracks for unmatched detections
           for i, det in enumerate(detections):
               if i not in detection_indices:
                   track = TrackedObject(det, self.next_id)
                   self.tracks.append(track)
                   self.next_id += 1

           # Remove old tracks that haven't been seen recently
           self.tracks = [t for i, t in enumerate(self.tracks)
                         if i in matched_tracks or t.hit_streak > 0]

           return self.tracks

       def calculate_distance_matrix(self, tracks, detections):
           """Calculate distance matrix for track-detection association"""
           n_tracks = len(tracks)
           n_detections = len(detections)

           dist_matrix = np.zeros((n_tracks, n_detections))

           for i, track in enumerate(tracks):
               for j, det in enumerate(detections):
                   # Calculate distance between track center and detection center
                   dist = np.linalg.norm(
                       np.array(track.center) - np.array(det.center)
                   )
                   dist_matrix[i, j] = dist

           return dist_matrix
   ```

4. **Launch and Test Pipeline**
   ```bash
   # Create launch file for perception pipeline
   # launch/perception_pipeline.launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node

   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='slam_package',
               executable='perception_pipeline',
               name='perception_pipeline',
               parameters=[{
                   'camera_topic': '/camera/rgb/image_raw',
                   'lidar_topic': '/scan',
                   'detection_threshold': 0.5
               }],
               output='screen'
           )
       ])
   ```

### Expected Outcomes
- Perception pipeline successfully processes multiple sensor streams
- Objects are detected and tracked over time
- Obstacle information is published for navigation
- Perception system integrates multiple modalities effectively

## Exercise 5: Complete AI Integration

### Objective
Integrate all AI components (VSLAM, Nav2, RL, Perception) into a complete autonomous humanoid system.

### Exercise Steps

1. **Create System Integration Node**
   ```python
   # system_integration.py
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped, Twist
   from sensor_msgs.msg import Image, Imu, LaserScan
   from std_msgs.msg import String, Bool
   import numpy as np

   class HumanoidAISystem(Node):
       def __init__(self):
           super().__init__('humanoid_ai_system')

           # System state
           self.current_task = 'idle'
           self.navigation_goal = None
           self.perception_data = {}

           # Create action clients for different systems
           self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

           # Create subscribers for all sensor data
           self.create_subscription(Image, '/camera/rgb/image_raw',
                                  self.image_callback, 10)
           self.create_subscription(Imu, '/imu/data',
                                  self.imu_callback, 10)
           self.create_subscription(LaserScan, '/scan',
                                  self.lidar_callback, 10)

           # Create publishers
           self.task_status_pub = self.create_publisher(String, '/task_status', 10)
           self.system_status_pub = self.create_publisher(Bool, '/system_active', 10)

           # Timer for main control loop
           self.control_timer = self.create_timer(0.1, self.main_control_loop)

       def image_callback(self, msg):
           """Process camera data through perception pipeline"""
           # Forward to perception system
           pass

       def imu_callback(self, msg):
           """Process IMU data for balance and orientation"""
           # Forward to SLAM and balance systems
           pass

       def lidar_callback(self, msg):
           """Process LiDAR data for navigation"""
           # Forward to navigation system
           pass

       def main_control_loop(self):
           """Main control loop for AI system"""
           # Check system status
           if not self.system_active:
               return

           # Update perception
           self.update_perception()

           # Make high-level decisions
           self.make_decision()

           # Execute tasks
           self.execute_task()

       def update_perception(self):
           """Update perception system with latest sensor data"""
           # This would integrate with the perception pipeline
           pass

       def make_decision(self):
           """Make high-level decisions based on perception"""
           # Check for objects of interest
           if self.object_detected('person'):
               self.set_task('approach_person')
           elif self.object_detected('goal'):
               self.set_task('navigate_to_goal')
           else:
               self.set_task('explore_environment')

       def execute_task(self):
           """Execute current task"""
           if self.current_task == 'navigate_to_goal':
               self.execute_navigation()
           elif self.current_task == 'approach_person':
               self.execute_approach()
           elif self.current_task == 'explore_environment':
               self.execute_exploration()

       def execute_navigation(self):
           """Execute navigation task using Nav2"""
           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.pose = self.navigation_goal

           self.nav_client.send_goal_async(goal_msg)

       def object_detected(self, class_name):
           """Check if specific object class is detected"""
           # This would check perception results
           return False

       def set_task(self, task):
           """Set current system task"""
           if self.current_task != task:
               self.current_task = task
               status_msg = String()
               status_msg.data = f"Task changed to: {task}"
               self.task_status_pub.publish(status_msg)
   ```

2. **Create Complete Launch File**
   ```xml
   <!-- launch/complete_ai_system.launch.xml -->
   <launch>
       <!-- Start SLAM system -->
       <include file="$(find-pkg-share slam_package)/launch/slam_pipeline.launch.xml"/>

       <!-- Start Navigation system -->
       <include file="$(find-pkg-share slam_package)/launch/humanoid_navigation.launch.py"/>

       <!-- Start Perception pipeline -->
       <include file="$(find-pkg-share slam_package)/launch/perception_pipeline.launch.py"/>

       <!-- Start RL policy (if deployed) -->
       <node pkg="slam_package" exec="humanoid_policy_deployment"
             name="rl_controller" output="screen"/>

       <!-- Start system integration -->
       <node pkg="slam_package" exec="humanoid_ai_system"
             name="ai_system" output="screen"/>
   </launch>
   ```

3. **Test Complete System**
   ```bash
   # Launch the complete AI system
   ros2 launch slam_package complete_ai_system.launch.xml

   # Monitor system behavior
   ros2 topic echo /task_status
   ros2 topic echo /perception_status
   ros2 run rviz2 rviz2
   ```

### Expected Outcomes
- All AI components work together in a unified system
- Robot demonstrates autonomous behavior based on perception
- Navigation, perception, and control systems integrate seamlessly
- System can handle complex tasks requiring multiple AI components

## Evaluation Criteria

### Performance Metrics
- **VSLAM**: Map accuracy, localization precision, processing time
- **Navigation**: Path efficiency, obstacle avoidance, success rate
- **RL**: Locomotion stability, energy efficiency, adaptability
- **Perception**: Detection accuracy, tracking precision, processing speed

### Validation Tests
1. **Mapping Accuracy Test**: Compare generated map with ground truth
2. **Navigation Success Rate**: Percentage of successful goal reaches
3. **Locomotion Stability**: Walking success rate over time
4. **Perception Reliability**: Detection and tracking accuracy metrics

### Troubleshooting Guide
- Check sensor calibration and synchronization
- Verify ROS topic connections
- Monitor system resource usage
- Validate parameter configurations
- Test individual components before integration