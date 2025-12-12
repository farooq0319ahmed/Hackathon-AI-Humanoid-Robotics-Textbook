---
sidebar_position: 5
---

# Capstone Project: End-to-End Autonomous Humanoid Implementation

## Overview

The capstone project brings together all the concepts learned throughout the course to create a comprehensive, end-to-end autonomous humanoid robot system. This project demonstrates the integration of ROS 2 communication, simulation environments, AI perception and navigation, and voice-language-action capabilities into a single, functioning autonomous system.

The capstone project involves implementing a humanoid robot that can:
- Understand and respond to natural voice commands
- Navigate complex environments autonomously
- Manipulate objects in its surroundings
- Interact naturally with humans
- Adapt to changing environmental conditions

This project serves as the culmination of the entire course, showcasing the student's ability to integrate multiple complex systems into a cohesive, functional robot.

## Project Architecture

### System Integration Overview

The capstone project integrates all previous modules into a unified architecture:

```python
# Example: Capstone project system architecture
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, Imu, LaserScan
import asyncio
import threading
from typing import Dict, Any, Optional

class CapstoneSystem(Node):
    def __init__(self):
        super().__init__('capstone_system')

        # Initialize all subsystems
        self.ros2_manager = ROS2Manager(self)
        self.simulation_interface = SimulationInterface(self)
        self.ai_brain = AIBrain(self)
        self.vla_system = VLASystem(self)
        self.humanoid_controller = HumanoidController(self)

        # System state management
        self.system_state = {
            'initialized': False,
            'active_task': None,
            'safety_status': 'SAFE',
            'battery_level': 100.0,
            'operational_mode': 'IDLE'
        }

        # Create main control timer
        self.control_timer = self.create_timer(0.1, self.main_control_loop)

        # Initialize system
        self.initialize_system()

    def initialize_system(self):
        """Initialize all subsystems"""
        self.get_logger().info("Initializing capstone system...")

        # Initialize ROS 2 communication layer
        self.ros2_manager.initialize()

        # Initialize simulation interface
        self.simulation_interface.initialize()

        # Initialize AI brain (perception, navigation, planning)
        self.ai_brain.initialize()

        # Initialize VLA (Voice-Language-Action) system
        self.vla_system.initialize()

        # Initialize humanoid controller
        self.humanoid_controller.initialize()

        # Set system state
        self.system_state['initialized'] = True
        self.system_state['operational_mode'] = 'IDLE'

        self.get_logger().info("Capstone system initialized successfully")

    def main_control_loop(self):
        """Main system control loop"""
        if not self.system_state['initialized']:
            return

        # Update system status
        self.update_system_status()

        # Process incoming commands
        self.process_commands()

        # Execute active task
        self.execute_active_task()

        # Monitor safety conditions
        self.monitor_safety()

        # Update system state
        self.publish_system_status()

    def update_system_status(self):
        """Update system status from all subsystems"""
        # Get status from each subsystem
        ros_status = self.ros2_manager.get_status()
        sim_status = self.simulation_interface.get_status()
        ai_status = self.ai_brain.get_status()
        vla_status = self.vla_system.get_status()
        controller_status = self.humanoid_controller.get_status()

        # Update overall system status
        self.system_state.update({
            'ros_status': ros_status,
            'simulation_status': sim_status,
            'ai_status': ai_status,
            'vla_status': vla_status,
            'controller_status': controller_status
        })

    def process_commands(self):
        """Process incoming commands from various sources"""
        # Check for voice commands
        voice_command = self.vla_system.get_latest_command()
        if voice_command:
            self.handle_voice_command(voice_command)

        # Check for remote commands
        remote_command = self.get_remote_command()
        if remote_command:
            self.handle_remote_command(remote_command)

    def handle_voice_command(self, command: Dict[str, Any]):
        """Handle voice command through VLA system"""
        self.get_logger().info(f"Processing voice command: {command}")

        # Parse and validate command
        parsed_command = self.vla_system.parse_command(command)

        if parsed_command:
            # Plan and execute command
            plan = self.ai_brain.plan_command(parsed_command)
            if plan:
                self.execute_plan(plan)

    def handle_remote_command(self, command: Dict[str, Any]):
        """Handle remote command"""
        self.get_logger().info(f"Processing remote command: {command}")
        # Implementation depends on command type

    def execute_active_task(self):
        """Execute the currently active task"""
        if self.system_state['active_task']:
            task = self.system_state['active_task']
            result = self.execute_plan(task['plan'])

            if result['completed']:
                self.system_state['active_task'] = None
                self.get_logger().info(f"Task completed: {task['description']}")

    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cognitive plan"""
        execution_result = {
            'completed': True,
            'success': True,
            'actions_executed': [],
            'actions_failed': []
        }

        for action in plan.get('actions', []):
            action_result = self.execute_action(action)
            if action_result['success']:
                execution_result['actions_executed'].append(action)
            else:
                execution_result['actions_failed'].append(action)
                execution_result['success'] = False
                break  # Stop on first failure for this example

        execution_result['completed'] = len(execution_result['actions_failed']) == 0
        return execution_result

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get('name', '').upper()
        parameters = action.get('parameters', {})

        self.get_logger().info(f"Executing action: {action_type}")

        try:
            if 'NAVIGATE' in action_type:
                return self.humanoid_controller.navigate_to_location(parameters)
            elif 'PICK' in action_type or 'GRASP' in action_type:
                return self.humanoid_controller.manipulate_object(parameters)
            elif 'SPEAK' in action_type:
                return self.vla_system.speak(parameters.get('text', ''))
            elif 'DETECT' in action_type:
                return self.ai_brain.detect_object(parameters)
            else:
                return {'success': False, 'error': f'Unknown action: {action_type}'}

        except Exception as e:
            self.get_logger().error(f"Action execution error: {e}")
            return {'success': False, 'error': str(e)}

    def monitor_safety(self):
        """Monitor system safety conditions"""
        # Check battery level
        if self.system_state['battery_level'] < 10.0:
            self.system_state['safety_status'] = 'BATTERY_LOW'
            self.return_to_base()

        # Check for safety violations
        if self.humanoid_controller.is_unsafe_condition():
            self.system_state['safety_status'] = 'UNSAFE'
            self.emergency_stop()

    def return_to_base(self):
        """Return to charging/base station"""
        self.get_logger().info("Returning to base due to low battery")
        # Implementation to navigate to base station

    def emergency_stop(self):
        """Emergency stop all robot operations"""
        self.get_logger().info("Emergency stop activated")
        self.humanoid_controller.emergency_stop()
        self.system_state['operational_mode'] = 'EMERGENCY_STOP'

    def publish_system_status(self):
        """Publish system status"""
        status_msg = String()
        status_msg.data = str({
            'state': self.system_state,
            'timestamp': self.get_clock().now().to_msg()
        })
        # Publish to status topic

    def get_remote_command(self) -> Optional[Dict[str, Any]]:
        """Get remote command (placeholder)"""
        # Implementation would interface with remote command interface
        return None

class ROS2Manager:
    def __init__(self, node):
        self.node = node
        self.subscribers = []
        self.publishers = []
        self.services = []

    def initialize(self):
        """Initialize ROS 2 communication layer"""
        # Create publishers and subscribers for all required topics
        self.setup_communication_channels()

    def setup_communication_channels(self):
        """Setup all necessary ROS 2 communication channels"""
        # Navigation topics
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)

        # Sensor topics
        self.image_sub = self.node.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10
        )
        self.imu_sub = self.node.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.scan_sub = self.node.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # System status topics
        self.status_pub = self.node.create_publisher(String, '/system_status', 10)
        self.emergency_stop_pub = self.node.create_publisher(Bool, '/emergency_stop', 10)

    def image_callback(self, msg):
        """Handle image messages"""
        # Forward to perception system
        pass

    def imu_callback(self, msg):
        """Handle IMU messages"""
        # Forward to balance and orientation systems
        pass

    def scan_callback(self, msg):
        """Handle laser scan messages"""
        # Forward to navigation and obstacle detection
        pass

    def get_status(self) -> Dict[str, Any]:
        """Get ROS 2 system status"""
        return {'status': 'OK', 'topics_active': len(self.subscribers)}
```

### Integration Layer

The integration layer connects all modules seamlessly:

```python
# Example: Integration layer connecting all modules
class IntegrationLayer:
    def __init__(self):
        self.ros_bridge = ROSBridge()
        self.simulation_sync = SimulationSynchronizer()
        self.ai_pipeline = AIPipeline()
        self.vla_bridge = VLABridge()

    def initialize_integration(self):
        """Initialize the integration between all modules"""
        # Initialize communication bridges
        self.ros_bridge.initialize()
        self.simulation_sync.initialize()
        self.ai_pipeline.initialize()
        self.vla_bridge.initialize()

        # Establish data flow between modules
        self.setup_data_flow()

    def setup_data_flow(self):
        """Setup data flow between all modules"""
        # ROS 2 -> Perception
        self.ros_bridge.subscribe_to_sensor_data(
            callback=self.ai_pipeline.process_sensor_data
        )

        # Perception -> Navigation
        self.ai_pipeline.subscribe_to_perception_updates(
            callback=self.ai_pipeline.update_navigation_map
        )

        # VLA -> Cognitive Planning
        self.vla_bridge.subscribe_to_commands(
            callback=self.ai_pipeline.plan_from_command
        )

        # Planning -> Control
        self.ai_pipeline.subscribe_to_plans(
            callback=self.ros_bridge.execute_plan
        )

        # Simulation -> All modules
        self.simulation_sync.subscribe_to_sim_state(
            callback=self.update_all_modules
        )

    def update_all_modules(self, sim_state: Dict[str, Any]):
        """Update all modules with simulation state"""
        # Update perception with ground truth
        self.ai_pipeline.update_ground_truth(sim_state)

        # Update navigation with environment changes
        self.ai_pipeline.update_environment(sim_state.get('environment', {}))

        # Update VLA with scene context
        self.vla_bridge.update_context(sim_state.get('scene_context', {}))

class ROSBridge:
    def __init__(self):
        self.node = None
        self.data_handlers = {}

    def initialize(self):
        """Initialize ROS bridge"""
        rclpy.init()
        self.node = rclpy.create_node('capstone_integration_bridge')

        # Setup all necessary publishers and subscribers
        self.setup_ros_interfaces()

    def setup_ros_interfaces(self):
        """Setup ROS interfaces for all modules"""
        # Sensor data publishers (from simulation to perception)
        self.image_pub = self.node.create_publisher(Image, '/sim/camera/rgb/image_raw', 10)
        self.imu_pub = self.node.create_publisher(Imu, '/sim/imu/data', 10)
        self.scan_pub = self.node.create_publisher(LaserScan, '/sim/scan', 10)

        # Command subscribers (from AI to control)
        self.cmd_vel_sub = self.node.create_subscription(
            Twist, '/cmd_vel', self.handle_cmd_vel, 10
        )
        self.joint_cmd_sub = self.node.create_subscription(
            String, '/joint_commands', self.handle_joint_commands, 10
        )

    def handle_cmd_vel(self, msg):
        """Handle velocity commands"""
        if 'cmd_vel' in self.data_handlers:
            self.data_handlers['cmd_vel'](msg)

    def handle_joint_commands(self, msg):
        """Handle joint commands"""
        if 'joint_cmd' in self.data_handlers:
            self.data_handlers['joint_cmd'](msg)

    def subscribe_to_sensor_data(self, callback):
        """Subscribe to sensor data with callback"""
        self.data_handlers['sensor'] = callback

    def execute_plan(self, plan):
        """Execute a plan through ROS"""
        for action in plan.get('actions', []):
            self.execute_action_ros(action)

    def execute_action_ros(self, action: Dict[str, Any]):
        """Execute action through ROS interface"""
        action_type = action.get('name', '').upper()

        if 'NAVIGATE' in action_type:
            self.publish_navigation_command(action)
        elif 'PICK' in action_type:
            self.publish_manipulation_command(action)
        elif 'SPEAK' in action_type:
            self.publish_speech_command(action)

    def publish_navigation_command(self, action):
        """Publish navigation command"""
        # Implementation to send navigation commands
        pass

    def publish_manipulation_command(self, action):
        """Publish manipulation command"""
        # Implementation to send manipulation commands
        pass

    def publish_speech_command(self, action):
        """Publish speech command"""
        # Implementation to send speech commands
        pass
```

## Voice Command Integration

### VLA System Implementation

The VLA system processes voice commands and integrates them with the robot's cognitive capabilities:

```python
# Example: VLA system for the capstone project
import openai
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class VLASystem:
    def __init__(self, node):
        self.node = node
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialize OpenAI client
        self.openai_client = None

        # Command processing queue
        self.command_queue = asyncio.Queue()
        self.processing_thread = threading.Thread(target=self.process_commands_loop)
        self.processing_thread.start()

        # Current command state
        self.current_command = None
        self.command_history = []

    def initialize(self):
        """Initialize VLA system"""
        # Set up OpenAI API key
        api_key = self.node.declare_parameter('openai_api_key', '').value
        if api_key:
            openai.api_key = api_key
            self.openai_client = openai

        self.node.get_logger().info("VLA system initialized")

    def process_commands_loop(self):
        """Process commands in separate thread"""
        asyncio.run(self.async_command_processor())

    async def async_command_processor(self):
        """Asynchronous command processing loop"""
        while rclpy.ok():
            try:
                command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                await self.process_command_async(command)
            except asyncio.TimeoutError:
                continue

    async def process_command_async(self, command: Dict[str, Any]):
        """Process command asynchronously"""
        try:
            # Parse and understand command
            parsed_command = await self.parse_command_async(command)

            if parsed_command:
                # Add to history
                self.command_history.append(parsed_command)
                if len(self.command_history) > 50:  # Limit history size
                    self.command_history.pop(0)

                # Set as current command
                self.current_command = parsed_command

                self.node.get_logger().info(f"Processed command: {parsed_command}")

        except Exception as e:
            self.node.get_logger().error(f"Command processing error: {e}")

    async def parse_command_async(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse command using LLM asynchronously"""
        if not self.openai_client:
            # Fallback to simple parsing
            return self.simple_parse_command(command)

        # Use LLM for sophisticated parsing
        command_text = command.get('text', '')
        context = command.get('context', {})

        system_prompt = """
        You are a command parser for a humanoid robot. Parse the given command
        and convert it into a structured format that can be used for cognitive planning.

        Available command types:
        - NAVIGATION: Move to location
        - MANIPULATION: Pick up, place, or manipulate objects
        - INTERACTION: Greet, follow, or interact with humans
        - INFORMATION: Answer questions or provide information
        - UTILITY: Charging, maintenance, or system commands

        Response format: JSON with 'type', 'target', 'parameters', and 'context' fields.
        """

        try:
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Command: {command_text}\nContext: {context}"}
                ],
                temperature=0.1,
                max_tokens=300
            )

            parsed_json = response.choices[0].message.content
            parsed_command = eval(parsed_json)  # In practice, use json.loads safely
            return parsed_command

        except Exception as e:
            self.node.get_logger().error(f"LLM parsing error: {e}")
            # Fallback to simple parsing
            return self.simple_parse_command(command)

    def simple_parse_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Simple command parsing as fallback"""
        command_text = command.get('text', '').lower()

        if any(word in command_text for word in ['go to', 'navigate', 'move to', 'walk to']):
            return {
                'type': 'NAVIGATION',
                'target': self.extract_location(command_text),
                'parameters': {},
                'confidence': 0.7
            }
        elif any(word in command_text for word in ['pick up', 'grasp', 'get', 'take']):
            return {
                'type': 'MANIPULATION',
                'target': self.extract_object(command_text),
                'parameters': {},
                'confidence': 0.7
            }
        elif any(word in command_text for word in ['hello', 'hi', 'greet', 'hey']):
            return {
                'type': 'INTERACTION',
                'target': 'human',
                'parameters': {},
                'confidence': 0.8
            }
        else:
            return {
                'type': 'UNKNOWN',
                'target': command_text,
                'parameters': {},
                'confidence': 0.3
            }

    def extract_location(self, command: str) -> str:
        """Extract location from command"""
        # Simple location extraction
        locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom']
        for loc in locations:
            if loc in command:
                return loc
        return 'unknown'

    def extract_object(self, command: str) -> str:
        """Extract object from command"""
        # Simple object extraction
        objects = ['cup', 'book', 'ball', 'box', 'bottle', 'phone']
        for obj in objects:
            if obj in command:
                return obj
        return 'unknown'

    def get_latest_command(self) -> Optional[Dict[str, Any]]:
        """Get the latest processed command"""
        return self.current_command

    def speak(self, text: str) -> Dict[str, Any]:
        """Make the robot speak"""
        try:
            # Publish speech command
            speech_msg = String()
            speech_msg.data = text
            # Would publish to speech synthesis topic

            return {'success': True, 'text': text}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def queue_command(self, command: Dict[str, Any]):
        """Queue a command for processing"""
        try:
            asyncio.run_coroutine_threadsafe(
                self.command_queue.put(command),
                asyncio.get_event_loop()
            )
        except:
            # Queue is full, drop the command
            pass
```

## Navigation and Path Planning

### Integrated Navigation System

The navigation system integrates with perception and planning for autonomous movement:

```python
# Example: Integrated navigation system for capstone
import numpy as np
from typing import List, Tuple, Dict, Any
import heapq

class IntegratedNavigation:
    def __init__(self, node):
        self.node = node
        self.map_resolution = 0.05  # 5cm per cell
        self.map_width = 200  # 10m x 10m map
        self.map_height = 200
        self.occupancy_grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        # Robot parameters
        self.robot_radius = 0.3  # 30cm radius
        self.path = []
        self.current_goal = None
        self.current_pose = None

    def update_occupancy_grid(self, laser_scan: LaserScan, robot_pose: PoseStamped):
        """Update occupancy grid with laser scan data"""
        # Convert laser scan to occupancy grid
        angles = np.linspace(
            laser_scan.angle_min,
            laser_scan.angle_max,
            len(laser_scan.ranges)
        )

        robot_x = int(robot_pose.pose.position.x / self.map_resolution + self.map_width // 2)
        robot_y = int(robot_pose.pose.position.y / self.map_resolution + self.map_height // 2)

        for i, (angle, range_val) in enumerate(zip(angles, laser_scan.ranges)):
            if range_val < laser_scan.range_min or range_val > laser_scan.range_max:
                continue

            # Calculate endpoint of laser beam
            end_x = robot_x + int((range_val * np.cos(angle + robot_pose.pose.orientation.z)) / self.map_resolution)
            end_y = robot_y + int((range_val * np.sin(angle + robot_pose.pose.orientation.z)) / self.map_resolution)

            # Mark endpoint as occupied if it's a valid obstacle
            if 0 <= end_x < self.map_width and 0 <= end_y < self.map_height:
                if range_val < 2.0:  # Consider obstacles within 2m
                    self.occupancy_grid[end_y, end_x] = 100  # Occupied

            # Mark path as free
            self.mark_free_path(robot_x, robot_y, end_x, end_y)

    def mark_free_path(self, x0: int, y0: int, x1: int, y1: int):
        """Mark the path between two points as free"""
        # Bresenham's line algorithm to mark free space
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < self.map_width and 0 <= y < self.map_height:
                # Only mark as free if not already marked as occupied
                if self.occupancy_grid[y, x] < 50:
                    self.occupancy_grid[y, x] = 0

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Plan path using A* algorithm"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if (0 <= nx < self.map_width and 0 <= ny < self.map_height and
                    self.occupancy_grid[ny, nx] < 50):  # Not occupied
                    # Check if robot can fit (simple circle collision)
                    if self.is_valid_position(nx, ny):
                        neighbors.append((nx, ny))
            return neighbors

        def is_valid_position(self, x: int, y: int) -> bool:
            """Check if position is valid for robot (considering size)"""
            robot_cells = self.get_robot_cells(x, y)
            for rx, ry in robot_cells:
                if (rx < 0 or rx >= self.map_width or ry < 0 or ry >= self.map_height or
                    self.occupancy_grid[ry, rx] >= 50):
                    return False
            return True

        def get_robot_cells(self, x: int, y: int) -> List[Tuple[int, int]]:
            """Get all grid cells occupied by robot at position"""
            cells = []
            robot_radius_cells = int(self.robot_radius / self.map_resolution)
            for dx in range(-robot_radius_cells, robot_radius_cells + 1):
                for dy in range(-robot_radius_cells, robot_radius_cells + 1):
                    if dx*dx + dy*dy <= robot_radius_cells*robot_radius_cells:
                        cells.append((x + dx, y + dy))
            return cells

        # A* algorithm
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next_pos in get_neighbors(current):
                new_cost = cost_so_far[current] + 1  # Simple uniform cost

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from.get(current)
            if current is None:
                return []  # No path found

        path.reverse()
        return path

    def execute_navigation(self, goal_pose: PoseStamped) -> Dict[str, Any]:
        """Execute navigation to goal pose"""
        try:
            # Convert goal to grid coordinates
            goal_x = int(goal_pose.pose.position.x / self.map_resolution + self.map_width // 2)
            goal_y = int(goal_pose.pose.position.y / self.map_resolution + self.map_height // 2)

            # Convert current pose to grid coordinates
            current_x = int(self.current_pose.pose.position.x / self.map_resolution + self.map_width // 2)
            current_y = int(self.current_pose.pose.position.y / self.map_resolution + self.map_height // 2)

            # Plan path
            path = self.plan_path((current_x, current_y), (goal_x, goal_y))

            if not path:
                return {'success': False, 'error': 'No path found'}

            # Convert path back to world coordinates and execute
            self.path = self.convert_path_to_world(path)
            execution_result = self.follow_path(self.path)

            return execution_result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def convert_path_to_world(self, grid_path: List[Tuple[int, int]]) -> List[PoseStamped]:
        """Convert grid path to world coordinates"""
        world_path = []
        for grid_x, grid_y in grid_path:
            pose = PoseStamped()
            pose.pose.position.x = (grid_x - self.map_width // 2) * self.map_resolution
            pose.pose.position.y = (grid_y - self.map_height // 2) * self.map_resolution
            world_path.append(pose)
        return world_path

    def follow_path(self, path: List[PoseStamped]) -> Dict[str, Any]:
        """Follow the planned path"""
        for i, pose in enumerate(path):
            # Publish velocity commands to follow path
            velocity_cmd = self.calculate_velocity_to_pose(pose)

            # Would publish to /cmd_vel topic
            # self.cmd_vel_publisher.publish(velocity_cmd)

            # Check for obstacles and replan if necessary
            if self.check_for_obstacles():
                return {'success': False, 'error': 'Obstacle encountered, replanning needed'}

        return {'success': True, 'completed': True}

    def calculate_velocity_to_pose(self, target_pose: PoseStamped) -> Twist:
        """Calculate velocity command to reach target pose"""
        cmd_vel = Twist()

        # Simple proportional controller
        dx = target_pose.pose.position.x - self.current_pose.pose.position.x
        dy = target_pose.pose.position.y - self.current_pose.pose.position.y

        # Calculate distance and angle to target
        distance = np.sqrt(dx*dx + dy*dy)
        target_angle = np.arctan2(dy, dx)
        current_angle = 2 * np.arctan2(
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w
        )

        # Proportional control
        linear_vel = min(0.5, distance * 0.5)  # Max 0.5 m/s
        angular_vel = (target_angle - current_angle) * 1.0

        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel

        return cmd_vel

    def check_for_obstacles(self) -> bool:
        """Check for obstacles in the path"""
        # This would check the latest laser scan data
        # For this example, return False (no obstacles)
        return False
```

## Manipulation System

### Object Manipulation Integration

The manipulation system handles object interaction and grasping:

```python
# Example: Object manipulation system for capstone
import numpy as np
from geometry_msgs.msg import Point, Pose
from sensor_msgs.msg import JointState

class ManipulationSystem:
    def __init__(self, node):
        self.node = node
        self.joint_names = [
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw',
            'left_elbow_pitch', 'left_forearm_yaw', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
            'right_elbow_pitch', 'right_forearm_yaw', 'right_wrist_pitch', 'right_wrist_yaw'
        ]
        self.current_joints = JointState()
        self.gripper_state = {'left': 0.0, 'right': 0.0}  # 0.0=open, 1.0=closed

    def initialize(self):
        """Initialize manipulation system"""
        # Setup joint state subscriber
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

    def joint_state_callback(self, msg):
        """Update current joint states"""
        self.current_joints = msg

    def detect_object(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Detect object in the environment"""
        # This would interface with perception system
        # For this example, return a mock detection
        return {
            'name': object_name,
            'position': {'x': 1.0, 'y': 0.5, 'z': 0.8},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0},
            'confidence': 0.9
        }

    def plan_grasp(self, object_pose: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan grasp trajectory for object"""
        object_pos = object_pose['position']

        # Simple grasp planning
        # Approach from above to avoid collisions
        approach_pos = {
            'x': object_pos['x'],
            'y': object_pos['y'],
            'z': object_pos['z'] + 0.2  # 20cm above object
        }

        grasp_pos = {
            'x': object_pos['x'],
            'y': object_pos['y'],
            'z': object_pos['z'] + 0.05  # 5cm above object
        }

        return {
            'approach_pose': approach_pos,
            'grasp_pose': grasp_pos,
            'gripper_command': 'close',
            'sequence': ['approach', 'descend', 'grasp', 'lift']
        }

    def execute_grasp(self, grasp_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute grasp plan"""
        try:
            for step in grasp_plan['sequence']:
                if step == 'approach':
                    success = self.move_to_approach_pose(grasp_plan['approach_pose'])
                elif step == 'descend':
                    success = self.descend_to_grasp_pose(grasp_plan['grasp_pose'])
                elif step == 'grasp':
                    success = self.close_gripper()
                elif step == 'lift':
                    success = self.lift_object()

                if not success:
                    return {'success': False, 'error': f'Failed at step: {step}'}

            return {'success': True, 'completed': True}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def move_to_approach_pose(self, pose: Dict[str, Any]) -> bool:
        """Move arm to approach pose"""
        # Calculate inverse kinematics for approach pose
        joint_angles = self.calculate_ik(pose, 'right_arm')

        if joint_angles:
            # Publish joint commands
            self.publish_joint_commands(joint_angles)
            # Wait for execution
            return self.wait_for_execution()
        return False

    def descend_to_grasp_pose(self, pose: Dict[str, Any]) -> bool:
        """Descend to grasp pose"""
        # Calculate inverse kinematics for grasp pose
        joint_angles = self.calculate_ik(pose, 'right_arm')

        if joint_angles:
            # Publish joint commands
            self.publish_joint_commands(joint_angles)
            # Wait for execution
            return self.wait_for_execution()
        return False

    def close_gripper(self) -> bool:
        """Close gripper to grasp object"""
        # Publish gripper close command
        gripper_cmd = {'right': 1.0}  # Close right gripper
        self.publish_gripper_command(gripper_cmd)
        return True

    def lift_object(self) -> bool:
        """Lift object after grasping"""
        # Move to safe lift pose
        lift_pose = {
            'x': self.current_joints.position[0],  # Use current x
            'y': self.current_joints.position[1],  # Use current y
            'z': self.current_joints.position[2] + 0.1  # Lift 10cm
        }

        joint_angles = self.calculate_ik(lift_pose, 'right_arm')
        if joint_angles:
            self.publish_joint_commands(joint_angles)
            return self.wait_for_execution()
        return False

    def calculate_ik(self, target_pose: Dict[str, Any], arm: str) -> Optional[List[float]]:
        """Calculate inverse kinematics for target pose"""
        # This would use a proper IK solver
        # For this example, return mock joint angles
        return [0.0] * 7  # Mock joint angles for 7-DOF arm

    def publish_joint_commands(self, joint_angles: List[float]):
        """Publish joint angle commands"""
        # Would publish to joint trajectory controller
        pass

    def publish_gripper_command(self, gripper_cmd: Dict[str, float]):
        """Publish gripper commands"""
        # Would publish to gripper controller
        pass

    def wait_for_execution(self) -> bool:
        """Wait for action execution to complete"""
        # This would monitor joint state feedback
        import time
        time.sleep(1)  # Mock wait
        return True
```

## Cognitive Planning Integration

### High-Level Task Planning

The cognitive planning system orchestrates complex tasks:

```python
# Example: Cognitive planning system for capstone
import asyncio
import json
from typing import Dict, Any, List

class CognitivePlanner:
    def __init__(self, node):
        self.node = node
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.task_history = []

    def initialize(self):
        """Initialize cognitive planning system"""
        # Start task processing loop
        self.processing_task = asyncio.create_task(self.process_tasks_loop())

    async def process_tasks_loop(self):
        """Process tasks in the queue"""
        while rclpy.ok():
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self.execute_task(task)
            except asyncio.TimeoutError:
                continue

    async def plan_task(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Plan a task from command"""
        command_type = command.get('type', 'UNKNOWN')

        if command_type == 'NAVIGATION':
            return await self.plan_navigation_task(command)
        elif command_type == 'MANIPULATION':
            return await self.plan_manipulation_task(command)
        elif command_type == 'INTERACTION':
            return await self.plan_interaction_task(command)
        else:
            return await self.plan_generic_task(command)

    async def plan_navigation_task(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Plan navigation task"""
        location = command.get('target', 'unknown')

        # Define navigation subtasks
        subtasks = [
            {
                'name': 'FIND_LOCATION',
                'parameters': {'location': location},
                'dependencies': []
            },
            {
                'name': 'NAVIGATE_TO_LOCATION',
                'parameters': {'target': location},
                'dependencies': ['FIND_LOCATION']
            },
            {
                'name': 'ARRIVE_AT_LOCATION',
                'parameters': {'location': location},
                'dependencies': ['NAVIGATE_TO_LOCATION']
            }
        ]

        return {
            'task_id': self.generate_task_id(),
            'type': 'NAVIGATION',
            'description': f'Navigate to {location}',
            'subtasks': subtasks,
            'status': 'PLANNED',
            'created_at': self.node.get_clock().now().nanoseconds / 1e9
        }

    async def plan_manipulation_task(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Plan manipulation task"""
        object_name = command.get('target', 'unknown')

        # Define manipulation subtasks
        subtasks = [
            {
                'name': 'DETECT_OBJECT',
                'parameters': {'object_name': object_name},
                'dependencies': []
            },
            {
                'name': 'NAVIGATE_TO_OBJECT',
                'parameters': {'object_name': object_name},
                'dependencies': ['DETECT_OBJECT']
            },
            {
                'name': 'GRASP_OBJECT',
                'parameters': {'object_name': object_name},
                'dependencies': ['NAVIGATE_TO_OBJECT']
            },
            {
                'name': 'TRANSPORT_OBJECT',
                'parameters': {'object_name': object_name, 'destination': 'default'},
                'dependencies': ['GRASP_OBJECT']
            }
        ]

        return {
            'task_id': self.generate_task_id(),
            'type': 'MANIPULATION',
            'description': f'Pick up {object_name}',
            'subtasks': subtasks,
            'status': 'PLANNED',
            'created_at': self.node.get_clock().now().nanoseconds / 1e9
        }

    async def plan_interaction_task(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Plan interaction task"""
        target = command.get('target', 'human')

        # Define interaction subtasks
        subtasks = [
            {
                'name': 'DETECT_HUMAN',
                'parameters': {'target': target},
                'dependencies': []
            },
            {
                'name': 'NAVIGATE_TO_HUMAN',
                'parameters': {'target': target},
                'dependencies': ['DETECT_HUMAN']
            },
            {
                'name': 'GREET_HUMAN',
                'parameters': {'target': target},
                'dependencies': ['NAVIGATE_TO_HUMAN']
            },
            {
                'name': 'MAINTAIN_INTERACTION',
                'parameters': {'target': target},
                'dependencies': ['GREET_HUMAN']
            }
        ]

        return {
            'task_id': self.generate_task_id(),
            'type': 'INTERACTION',
            'description': f'Interact with {target}',
            'subtasks': subtasks,
            'status': 'PLANNED',
            'created_at': self.node.get_clock().now().nanoseconds / 1e9
        }

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planned task"""
        self.node.get_logger().info(f"Executing task: {task['description']}")

        task['status'] = 'EXECUTING'
        self.active_tasks[task['task_id']] = task

        try:
            # Execute subtasks in dependency order
            execution_results = []
            for subtask in task['subtasks']:
                result = await self.execute_subtask(subtask)
                execution_results.append(result)

                if not result['success']:
                    task['status'] = 'FAILED'
                    break

            if all(r['success'] for r in execution_results):
                task['status'] = 'COMPLETED'
            else:
                task['status'] = 'PARTIAL'

        except Exception as e:
            task['status'] = 'ERROR'
            self.node.get_logger().error(f"Task execution error: {e}")

        # Add to history
        self.task_history.append(task)
        if len(self.task_history) > 100:
            self.task_history.pop(0)

        self.active_tasks.pop(task['task_id'], None)
        return task

    async def execute_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask"""
        subtask_name = subtask['name']
        parameters = subtask['parameters']

        self.node.get_logger().info(f"Executing subtask: {subtask_name}")

        try:
            if subtask_name == 'NAVIGATE_TO_LOCATION':
                result = self.execute_navigation_subtask(parameters)
            elif subtask_name == 'GRASP_OBJECT':
                result = self.execute_manipulation_subtask(parameters)
            elif subtask_name == 'GREET_HUMAN':
                result = self.execute_interaction_subtask(parameters)
            elif subtask_name == 'DETECT_OBJECT':
                result = self.execute_perception_subtask(parameters)
            else:
                result = {'success': False, 'error': f'Unknown subtask: {subtask_name}'}

            return result

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_navigation_subtask(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation subtask"""
        # This would interface with navigation system
        return {'success': True, 'completed': True}

    def execute_manipulation_subtask(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manipulation subtask"""
        # This would interface with manipulation system
        return {'success': True, 'completed': True}

    def execute_interaction_subtask(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute interaction subtask"""
        # This would interface with VLA system
        return {'success': True, 'completed': True}

    def execute_perception_subtask(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute perception subtask"""
        # This would interface with perception system
        return {'success': True, 'completed': True}

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())[:8]

    def queue_task(self, task: Dict[str, Any]):
        """Queue a task for execution"""
        asyncio.run_coroutine_threadsafe(
            self.task_queue.put(task),
            asyncio.get_event_loop()
        )
```

## Safety and Monitoring

### Safety System Implementation

Implementing comprehensive safety monitoring:

```python
# Example: Safety and monitoring system for capstone
from enum import Enum
import threading
import time

class SafetyLevel(Enum):
    SAFE = 1
    WARNING = 2
    DANGER = 3
    EMERGENCY = 4

class SafetyMonitor:
    def __init__(self, node):
        self.node = node
        self.safety_level = SafetyLevel.SAFE
        self.safety_conditions = {
            'collision_risk': False,
            'balance_risk': False,
            'power_low': False,
            'communication_lost': False,
            'hardware_error': False
        }
        self.emergency_stop_active = False

        # Monitoring intervals
        self.monitoring_interval = 0.1  # 10Hz monitoring
        self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
        self.running = True

    def start_monitoring(self):
        """Start safety monitoring"""
        self.monitoring_thread.start()

    def monitoring_loop(self):
        """Continuous safety monitoring loop"""
        while self.running:
            self.check_safety_conditions()
            self.update_safety_level()

            if self.safety_level == SafetyLevel.EMERGENCY:
                self.activate_emergency_stop()
            elif self.emergency_stop_active and self.safety_level == SafetyLevel.SAFE:
                self.deactivate_emergency_stop()

            time.sleep(self.monitoring_interval)

    def check_safety_conditions(self):
        """Check all safety conditions"""
        # Check collision risk from navigation system
        self.safety_conditions['collision_risk'] = self.check_collision_risk()

        # Check balance from IMU data
        self.safety_conditions['balance_risk'] = self.check_balance_risk()

        # Check power level
        self.safety_conditions['power_low'] = self.check_power_level()

        # Check communication status
        self.safety_conditions['communication_lost'] = self.check_communication_status()

        # Check hardware status
        self.safety_conditions['hardware_error'] = self.check_hardware_status()

    def check_collision_risk(self) -> bool:
        """Check for collision risk"""
        # This would interface with navigation/obstacle detection
        # For example, check if distance to nearest obstacle is below threshold
        return False  # Placeholder

    def check_balance_risk(self) -> bool:
        """Check for balance risk using IMU data"""
        # Check if robot is tilting beyond safe angles
        # This would use actual IMU data
        return False  # Placeholder

    def check_power_level(self) -> bool:
        """Check if power is critically low"""
        # Check battery level
        # This would interface with power management system
        return False  # Placeholder

    def check_communication_status(self) -> bool:
        """Check if critical communications are lost"""
        # Check if important ROS topics are still active
        return False  # Placeholder

    def check_hardware_status(self) -> bool:
        """Check for hardware errors"""
        # Check joint states, sensor status, etc.
        return False  # Placeholder

    def update_safety_level(self):
        """Update safety level based on conditions"""
        if (self.safety_conditions['collision_risk'] or
            self.safety_conditions['balance_risk'] or
            self.safety_conditions['hardware_error']):
            self.safety_level = SafetyLevel.DANGER
        elif (self.safety_conditions['power_low'] or
              self.safety_conditions['communication_lost']):
            self.safety_level = SafetyLevel.WARNING
        else:
            self.safety_level = SafetyLevel.SAFE

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.node.get_logger().error("EMERGENCY STOP ACTIVATED")

            # Send emergency stop command to all systems
            self.send_emergency_stop_command()

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        if self.emergency_stop_active:
            self.emergency_stop_active = False
            self.node.get_logger().info("Emergency stop deactivated")

    def send_emergency_stop_command(self):
        """Send emergency stop command to all systems"""
        # Publish to emergency stop topic
        emergency_msg = Bool()
        emergency_msg.data = True
        # self.emergency_stop_publisher.publish(emergency_msg)

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status"""
        return {
            'level': self.safety_level.name,
            'conditions': self.safety_conditions.copy(),
            'emergency_stop': self.emergency_stop_active,
            'timestamp': time.time()
        }

    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.running = False
        self.monitoring_thread.join()
```

## Performance Evaluation

### System Testing and Validation

Testing the complete integrated system:

```python
# Example: System testing and validation framework
import unittest
import time
from typing import Dict, Any

class CapstoneSystemTester:
    def __init__(self, system):
        self.system = system
        self.test_results = []

    def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        self.test_basic_functionality()
        self.test_integration_scenarios()
        self.test_safety_systems()
        self.test_performance_metrics()

        return self.generate_test_report()

    def test_basic_functionality(self):
        """Test basic functionality of each subsystem"""
        print("Testing basic functionality...")

        # Test ROS communication
        ros_test = self.test_ros_communication()
        self.test_results.append(('ROS Communication', ros_test))

        # Test perception system
        perception_test = self.test_perception_system()
        self.test_results.append(('Perception System', perception_test))

        # Test navigation system
        navigation_test = self.test_navigation_system()
        self.test_results.append(('Navigation System', navigation_test))

        # Test manipulation system
        manipulation_test = self.test_manipulation_system()
        self.test_results.append(('Manipulation System', manipulation_test))

        # Test VLA system
        vla_test = self.test_vla_system()
        self.test_results.append(('VLA System', vla_test))

    def test_integration_scenarios(self):
        """Test integration scenarios"""
        print("Testing integration scenarios...")

        # Test voice command to navigation
        voice_nav_test = self.test_voice_navigation_integration()
        self.test_results.append(('Voice Navigation Integration', voice_nav_test))

        # Test perception to manipulation
        perception_manip_test = self.test_perception_manipulation_integration()
        self.test_results.append(('Perception Manipulation Integration', perception_manip_test))

        # Test full task execution
        full_task_test = self.test_full_task_execution()
        self.test_results.append(('Full Task Execution', full_task_test))

    def test_safety_systems(self):
        """Test safety systems"""
        print("Testing safety systems...")

        # Test emergency stop
        emergency_stop_test = self.test_emergency_stop()
        self.test_results.append(('Emergency Stop', emergency_stop_test))

        # Test collision avoidance
        collision_avoidance_test = self.test_collision_avoidance()
        self.test_results.append(('Collision Avoidance', collision_avoidance_test))

        # Test balance maintenance
        balance_test = self.test_balance_maintenance()
        self.test_results.append(('Balance Maintenance', balance_test))

    def test_performance_metrics(self):
        """Test performance metrics"""
        print("Testing performance metrics...")

        # Measure response time
        response_time = self.measure_response_time()
        self.test_results.append(('Response Time', response_time))

        # Measure accuracy
        accuracy = self.measure_accuracy()
        self.test_results.append(('Accuracy', accuracy))

        # Measure throughput
        throughput = self.measure_throughput()
        self.test_results.append(('Throughput', throughput))

    def test_ros_communication(self) -> Dict[str, Any]:
        """Test ROS communication functionality"""
        try:
            # Check if critical topics are available
            topics = self.system.node.get_topic_names_and_types()
            critical_topics = ['/cmd_vel', '/joint_states', '/scan', '/camera/rgb/image_raw']

            available = [t for t in critical_topics if any(t in topic[0] for topic in topics)]

            success = len(available) == len(critical_topics)
            return {
                'success': success,
                'details': f'Available topics: {len(available)}/{len(critical_topics)}',
                'metrics': {'topic_availability': len(available) / len(critical_topics)}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_perception_system(self) -> Dict[str, Any]:
        """Test perception system functionality"""
        try:
            # Simulate perception input and check output
            # This would involve processing simulated sensor data
            perception_result = self.system.ai_brain.process_sensor_data({
                'image': 'simulated_image_data',
                'laser_scan': 'simulated_scan_data',
                'imu': 'simulated_imu_data'
            })

            success = perception_result is not None
            return {
                'success': success,
                'details': f'Perception result: {bool(perception_result)}',
                'metrics': {'processing_time': 0.1}  # Simulated
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_navigation_system(self) -> Dict[str, Any]:
        """Test navigation system functionality"""
        try:
            # Test path planning and execution
            start_pose = {'x': 0.0, 'y': 0.0}
            goal_pose = {'x': 1.0, 'y': 1.0}

            path = self.system.ai_brain.navigation.plan_path(start_pose, goal_pose)
            success = len(path) > 0

            return {
                'success': success,
                'details': f'Path length: {len(path) if path else 0}',
                'metrics': {'path_length': len(path) if path else 0}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_manipulation_system(self) -> Dict[str, Any]:
        """Test manipulation system functionality"""
        try:
            # Test object detection and grasping
            object_info = self.system.ai_brain.manipulation.detect_object('cup')
            if object_info:
                grasp_plan = self.system.ai_brain.manipulation.plan_grasp(object_info)
                success = grasp_plan is not None
            else:
                success = False

            return {
                'success': success,
                'details': f'Grasp plan: {bool(object_info)}',
                'metrics': {'detection_success': bool(object_info)}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_vla_system(self) -> Dict[str, Any]:
        """Test VLA system functionality"""
        try:
            # Test voice command processing
            command = {'text': 'go to kitchen', 'context': {}}
            parsed = self.system.vla_system.parse_command(command)
            success = parsed is not None

            return {
                'success': success,
                'details': f'Parsed command: {bool(parsed)}',
                'metrics': {'parsing_success': bool(parsed)}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_voice_navigation_integration(self) -> Dict[str, Any]:
        """Test voice command to navigation integration"""
        try:
            # Test complete pipeline: voice -> VLA -> cognitive planning -> navigation
            command = {'text': 'navigate to the kitchen', 'context': {}}

            # Process through VLA
            parsed_command = self.system.vla_system.parse_command(command)

            # Plan task
            task_plan = self.system.ai_brain.cognitive_planner.plan_task(parsed_command)

            # Execute task
            execution_result = self.system.ai_brain.cognitive_planner.execute_task(task_plan)

            success = execution_result.get('status') == 'COMPLETED'

            return {
                'success': success,
                'details': f'Task status: {execution_result.get("status", "unknown")}',
                'metrics': {'integration_success': success}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_perception_manipulation_integration(self) -> Dict[str, Any]:
        """Test perception to manipulation integration"""
        try:
            # Test object detection -> manipulation planning -> execution
            object_info = self.system.ai_brain.perception.detect_object('cup')

            if object_info:
                grasp_plan = self.system.ai_brain.manipulation.plan_grasp(object_info)
                execution_result = self.system.ai_brain.manipulation.execute_grasp(grasp_plan)
                success = execution_result.get('success', False)
            else:
                success = False

            return {
                'success': success,
                'details': f'Integration success: {success}',
                'metrics': {'end_to_end_success': success}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_full_task_execution(self) -> Dict[str, Any]:
        """Test full task execution from voice command to completion"""
        try:
            # Execute a complete task: "go to kitchen and pick up the cup"
            command = {'text': 'go to kitchen and pick up the cup', 'context': {}}

            # Process through entire pipeline
            result = self.system.process_complete_task(command)
            success = result.get('completed', False)

            return {
                'success': success,
                'details': f'Task completed: {success}',
                'metrics': {'task_completion_rate': 1.0 if success else 0.0}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def test_emergency_stop(self) -> Dict[str, Any]:
        """Test emergency stop functionality"""
        try:
            # Simulate emergency condition and check response
            self.system.safety_monitor.safety_conditions['collision_risk'] = True
            time.sleep(0.2)  # Allow monitoring to detect condition

            safety_status = self.system.safety_monitor.get_safety_status()
            success = safety_status['emergency_stop']

            # Reset condition
            self.system.safety_monitor.safety_conditions['collision_risk'] = False

            return {
                'success': success,
                'details': f'Emergency stop activated: {success}',
                'metrics': {'response_time': 0.2}
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, result in self.test_results if result.get('success', False))

        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0
            },
            'detailed_results': self.test_results,
            'timestamp': time.time()
        }

        return report
```

## Deployment and Optimization

### System Optimization

Optimizing the complete system for deployment:

```python
# Example: System optimization and deployment
import psutil
import time
from typing import Dict, Any

class SystemOptimizer:
    def __init__(self, system):
        self.system = system
        self.optimization_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'throughput': []
        }

    def optimize_for_deployment(self):
        """Optimize system for deployment"""
        self.optimize_resource_usage()
        self.optimize_real_time_performance()
        self.optimize_power_consumption()
        self.optimize_reliability()

    def optimize_resource_usage(self):
        """Optimize CPU and memory usage"""
        # Monitor and optimize resource usage
        def resource_monitor():
            while True:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                self.optimization_metrics['cpu_usage'].append(cpu_percent)
                self.optimization_metrics['memory_usage'].append(memory_percent)

                # If resources are high, consider reducing processing rate
                if cpu_percent > 80:
                    self.reduce_processing_load()

                time.sleep(1)

    def optimize_real_time_performance(self):
        """Optimize for real-time performance"""
        # Implement real-time scheduling
        # Prioritize critical tasks
        # Optimize algorithms for speed
        pass

    def optimize_power_consumption(self):
        """Optimize for power efficiency"""
        # Implement power management
        # Optimize sensor usage
        # Use sleep modes when appropriate
        pass

    def optimize_reliability(self):
        """Optimize for system reliability"""
        # Implement error handling
        # Add redundancy where critical
        # Implement health monitoring
        pass

    def reduce_processing_load(self):
        """Reduce processing load when resources are constrained"""
        # Lower update rates for non-critical systems
        # Reduce sensor processing frequency
        # Simplify algorithms temporarily
        pass

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization metrics and recommendations"""
        if not self.optimization_metrics['cpu_usage']:
            return {'status': 'No metrics collected yet'}

        avg_cpu = sum(self.optimization_metrics['cpu_usage']) / len(self.optimization_metrics['cpu_usage'])
        avg_memory = sum(self.optimization_metrics['memory_usage']) / len(self.optimization_metrics['memory_usage'])

        return {
            'average_cpu_usage': avg_cpu,
            'average_memory_usage': avg_memory,
            'recommendations': self.generate_recommendations(avg_cpu, avg_memory)
        }

    def generate_recommendations(self, avg_cpu: float, avg_memory: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        if avg_cpu > 70:
            recommendations.append("Consider reducing processing frequency or optimizing algorithms")
        if avg_memory > 80:
            recommendations.append("Consider reducing memory footprint or optimizing data structures")

        return recommendations
```

## Best Practices

### Development Guidelines

Key practices for successful capstone implementation:

1. **Modular Design**: Keep subsystems independent with well-defined interfaces
2. **Error Handling**: Implement comprehensive error handling and recovery
3. **Testing**: Develop comprehensive test suites for each component
4. **Documentation**: Maintain clear documentation for all interfaces
5. **Performance**: Monitor and optimize resource usage
6. **Safety**: Implement multiple layers of safety checks
7. **Scalability**: Design for future enhancements and modifications

### Troubleshooting

Common issues and solutions:

- **Integration Problems**: Use clear interfaces and thorough testing
- **Performance Issues**: Monitor resource usage and optimize bottlenecks
- **Safety Violations**: Implement multiple safety layers and validation
- **Communication Failures**: Use robust communication protocols
- **Sensor Noise**: Implement filtering and validation

## Summary

The capstone project demonstrates the integration of all course concepts into a comprehensive autonomous humanoid robot system. Success requires careful attention to system architecture, integration challenges, safety considerations, and performance optimization. The project showcases the student's ability to combine ROS 2 fundamentals, simulation environments, AI perception and navigation, and voice-language-action capabilities into a unified, functional robot system. Through rigorous testing and optimization, the system achieves the goal of natural human-robot interaction in complex environments.