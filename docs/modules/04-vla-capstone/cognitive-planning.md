---
sidebar_position: 3
---

# Cognitive Planning: Translating Commands into Actions

## Overview

Cognitive planning is the intelligence layer that transforms high-level human commands into executable sequences of robot actions. This system acts as the bridge between natural language understanding and robotic execution, decomposing complex tasks into manageable subtasks that consider environmental constraints, robot capabilities, and safety requirements. In humanoid robotics, cognitive planning must account for the complexity of bipedal locomotion, manipulation capabilities, and multi-modal perception systems.

The cognitive planning system in a Vision-Language-Action (VLA) framework must handle uncertainty, adapt to changing environments, and maintain safety while executing complex tasks. This involves integrating multiple AI systems including language models, perception pipelines, navigation systems, and control algorithms into a coherent planning framework.

## Cognitive Architecture

### Hierarchical Planning Structure

Cognitive planning in humanoid robots follows a hierarchical structure that decomposes high-level goals into executable actions:

```python
# Example: Hierarchical cognitive planning architecture
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
import openai

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class CognitiveTask:
    """Data structure for cognitive planning tasks"""
    id: str
    description: str
    priority: TaskPriority
    dependencies: List[str]
    action_sequence: List[Dict[str, Any]]
    context: Dict[str, Any]
    status: str = "PENDING"
    created_at: float = 0.0
    completed_at: Optional[float] = None

class HierarchicalPlanner:
    def __init__(self, config):
        self.config = config
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.task_history = []
        self.context_manager = ContextManager()

    async def plan_high_level_task(self, command: str, context: Dict[str, Any]) -> CognitiveTask:
        """Plan a high-level task from natural language command"""
        # Use LLM to decompose command into subtasks
        subtasks = await self.decompose_command(command, context)

        # Create cognitive task with action sequence
        task = CognitiveTask(
            id=self.generate_task_id(),
            description=command,
            priority=self.assess_priority(command),
            dependencies=[],
            action_sequence=subtasks,
            context=context
        )

        return task

    async def decompose_command(self, command: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose high-level command into executable actions"""
        system_prompt = f"""
        You are a cognitive planning system for a humanoid robot. Decompose the following command
        into a sequence of executable actions considering the robot's capabilities and environment.

        Robot capabilities:
        - Navigation: Move to locations, avoid obstacles
        - Manipulation: Pick up objects, place objects, open doors
        - Perception: Detect objects, recognize humans, map environment
        - Communication: Speak, gesture, display information

        Context: {json.dumps(context, indent=2)}

        Available actions:
        - NAVIGATE_TO_LOCATION: Move to a specific location
        - DETECT_OBJECT: Find a specific object in the environment
        - PICK_UP_OBJECT: Grasp an object
        - PLACE_OBJECT: Release an object at a location
        - FOLLOW_HUMAN: Follow a person
        - SPEAK: Verbal communication
        - GREET_HUMAN: Acknowledge and greet a person
        - WAIT: Pause execution for a duration
        - CHECK_CONDITION: Verify a condition is met

        Response format: JSON array of action objects with 'name' and 'parameters' fields.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Command: {command}"}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            actions_json = response.choices[0].message.content
            actions = json.loads(actions_json)
            return actions

        except Exception as e:
            print(f"Planning error: {e}")
            return self.fallback_decomposition(command)

    def fallback_decomposition(self, command: str) -> List[Dict[str, Any]]:
        """Fallback decomposition for error cases"""
        # Simple rule-based decomposition
        if "bring" in command.lower() or "get" in command.lower():
            return [
                {"name": "DETECT_OBJECT", "parameters": {"object_name": "unknown"}},
                {"name": "NAVIGATE_TO_LOCATION", "parameters": {"target": "object_location"}},
                {"name": "PICK_UP_OBJECT", "parameters": {"object_name": "unknown"}},
                {"name": "NAVIGATE_TO_LOCATION", "parameters": {"target": "delivery_location"}},
                {"name": "PLACE_OBJECT", "parameters": {"location": "delivery_location"}}
            ]
        elif "go to" in command.lower() or "move to" in command.lower():
            return [
                {"name": "NAVIGATE_TO_LOCATION", "parameters": {"target": "unknown_location"}}
            ]
        else:
            return [{"name": "SPEAK", "parameters": {"text": "Command not understood"}}]

    def assess_priority(self, command: str) -> TaskPriority:
        """Assess priority of command based on keywords"""
        high_priority_keywords = ["stop", "emergency", "danger", "help"]
        medium_priority_keywords = ["bring", "get", "follow", "greet"]

        command_lower = command.lower()

        for keyword in high_priority_keywords:
            if keyword in command_lower:
                return TaskPriority.HIGH

        for keyword in medium_priority_keywords:
            if keyword in command_lower:
                return TaskPriority.MEDIUM

        return TaskPriority.LOW

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())[:8]
```

### Context Management

Effective cognitive planning requires maintaining and updating context information:

```python
# Example: Context management for cognitive planning
import time
from typing import Any, Dict, List, Optional

class ContextManager:
    def __init__(self):
        self.context = {
            'robot_state': {},
            'environment': {},
            'task_history': [],
            'human_interactions': [],
            'object_locations': {},
            'navigation_map': {},
            'time_info': {}
        }
        self.context_history = []
        self.max_history = 50

    def update_context(self, updates: Dict[str, Any]):
        """Update context with new information"""
        for key, value in updates.items():
            if key in self.context:
                if isinstance(self.context[key], dict) and isinstance(value, dict):
                    self.context[key].update(value)
                else:
                    self.context[key] = value
            else:
                self.context[key] = value

        # Add to history for temporal context
        self.context_history.append({
            'timestamp': time.time(),
            'updates': updates.copy()
        })

        # Limit history size
        if len(self.context_history) > self.max_history:
            self.context_history.pop(0)

    def get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return self.context.copy()

    def get_relevant_context(self, task_description: str) -> Dict[str, Any]:
        """Get context relevant to specific task"""
        relevant_context = {}

        # Include robot state for all tasks
        relevant_context['robot_state'] = self.context['robot_state']

        # Include environment if navigation-related
        if any(keyword in task_description.lower() for keyword in ['go', 'move', 'navigate', 'location']):
            relevant_context['environment'] = self.context['environment']
            relevant_context['navigation_map'] = self.context['navigation_map']

        # Include object info if manipulation-related
        if any(keyword in task_description.lower() for keyword in ['pick', 'grasp', 'place', 'object']):
            relevant_context['object_locations'] = self.context['object_locations']

        # Include recent interactions for social tasks
        if any(keyword in task_description.lower() for keyword in ['greet', 'follow', 'human']):
            relevant_context['human_interactions'] = self.context['human_interactions'][-5:]  # Last 5 interactions

        return relevant_context

    def update_robot_state(self, state: Dict[str, Any]):
        """Update robot state in context"""
        self.context['robot_state'].update(state)

    def update_environment(self, env_data: Dict[str, Any]):
        """Update environment information"""
        self.context['environment'].update(env_data)

    def update_object_location(self, object_name: str, location: Dict[str, Any]):
        """Update object location in context"""
        self.context['object_locations'][object_name] = {
            'location': location,
            'timestamp': time.time(),
            'confidence': 0.9
        }

    def update_navigation_map(self, map_data: Dict[str, Any]):
        """Update navigation map in context"""
        self.context['navigation_map'].update(map_data)

    def add_task_to_history(self, task: Dict[str, Any]):
        """Add completed task to history"""
        self.context['task_history'].append({
            'task': task,
            'timestamp': time.time(),
            'success': True
        })

        # Limit task history
        if len(self.context['task_history']) > 100:
            self.context['task_history'] = self.context['task_history'][-50:]

    def get_temporal_context(self, seconds_back: int = 30) -> List[Dict[str, Any]]:
        """Get context changes within specified time window"""
        current_time = time.time()
        temporal_context = []

        for entry in reversed(self.context_history):
            if current_time - entry['timestamp'] <= seconds_back:
                temporal_context.append(entry)
            else:
                break

        return temporal_context
```

## LLM Integration for Cognitive Planning

### OpenAI Integration

Integrating large language models for sophisticated planning:

```python
# Example: LLM-based cognitive planning with OpenAI
import openai
import json
import asyncio
from typing import Dict, List, Any

class LLMCognitivePlanner:
    def __init__(self, config):
        openai.api_key = config.openai_api_key
        self.model = config.llm_model
        self.max_retries = 3
        self.context_window = config.context_window

    async def generate_plan(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cognitive plan using LLM"""
        system_prompt = self.create_system_prompt()

        user_message = f"""
        Command: {command}

        Context:
        {json.dumps(context, indent=2)}

        Generate a detailed cognitive plan that includes:
        1. Task decomposition into executable actions
        2. Required preconditions for each action
        3. Expected outcomes and success criteria
        4. Potential failure modes and recovery strategies
        5. Safety considerations and constraints
        6. Resource requirements and dependencies

        Response format: JSON object with 'plan', 'actions', 'constraints', and 'safety_checks' fields.
        """

        for attempt in range(self.max_retries):
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.3,
                    max_tokens=1500,
                    timeout=30
                )

                plan_json = response.choices[0].message.content
                plan = json.loads(plan_json)
                return plan

            except Exception as e:
                print(f"LLM planning attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return self.create_fallback_plan(command, context)
                await asyncio.sleep(1)  # Wait before retry

    def create_system_prompt(self) -> str:
        """Create system prompt for cognitive planning"""
        return """
        You are an advanced cognitive planning system for a humanoid robot. Your role is to:

        1. Analyze natural language commands and decompose them into executable action sequences
        2. Consider environmental constraints, robot capabilities, and safety requirements
        3. Generate detailed plans with preconditions, success criteria, and recovery strategies
        4. Account for uncertainty and provide robust planning with fallback options
        5. Ensure all plans follow safety protocols and operational constraints

        Robot capabilities include:
        - Navigation: Path planning, obstacle avoidance, human-aware navigation
        - Manipulation: Object grasping, placement, tool use, door opening
        - Perception: Object detection, human recognition, environment mapping
        - Communication: Speech synthesis, gesture, display interfaces
        - Locomotion: Bipedal walking, stair navigation, balance maintenance

        Safety constraints:
        - Avoid collisions with humans and obstacles
        - Maintain balance during all operations
        - Respect personal space and privacy
        - Stop immediately if unsafe conditions detected

        Response format: JSON with these fields:
        - 'plan': High-level plan description
        - 'actions': Array of action objects with name, parameters, preconditions
        - 'constraints': Environmental and operational constraints
        - 'safety_checks': Safety verification steps
        - 'recovery_strategies': Fallback plans for common failures
        """

    def create_fallback_plan(self, command: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback plan when LLM fails"""
        return {
            'plan': f"Basic execution of: {command}",
            'actions': self.basic_action_sequence(command),
            'constraints': {'max_execution_time': 300, 'safety_enabled': True},
            'safety_checks': ['obstacle_avoidance', 'balance_maintenance'],
            'recovery_strategies': ['stop_and_report', 'return_to_home']
        }

    def basic_action_sequence(self, command: str) -> List[Dict[str, Any]]:
        """Create basic action sequence for fallback"""
        command_lower = command.lower()

        if "navigate" in command_lower or "go to" in command_lower:
            return [
                {
                    'name': 'NAVIGATE_TO_LOCATION',
                    'parameters': {'target': 'unknown'},
                    'preconditions': ['navigation_system_ready', 'map_available']
                }
            ]
        elif "pick up" in command_lower or "grasp" in command_lower:
            return [
                {
                    'name': 'DETECT_OBJECT',
                    'parameters': {'object_name': 'unknown'},
                    'preconditions': ['perception_system_ready']
                },
                {
                    'name': 'NAVIGATE_TO_OBJECT',
                    'parameters': {'object_name': 'unknown'},
                    'preconditions': ['object_detected']
                },
                {
                    'name': 'PICK_UP_OBJECT',
                    'parameters': {'object_name': 'unknown'},
                    'preconditions': ['robot_at_object', 'gripper_ready']
                }
            ]
        else:
            return [
                {
                    'name': 'SPEAK',
                    'parameters': {'text': f"Processing command: {command}"},
                    'preconditions': ['speech_system_ready']
                }
            ]
```

### Plan Validation and Safety Checks

Implementing validation and safety for cognitive plans:

```python
# Example: Plan validation and safety checking
class PlanValidator:
    def __init__(self):
        self.safety_constraints = {
            'collision_avoidance': True,
            'balance_maintenance': True,
            'human_safety': True,
            'operational_limits': True
        }

    def validate_plan(self, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cognitive plan for safety and feasibility"""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'modified_plan': plan.copy()
        }

        # Check action feasibility
        for i, action in enumerate(plan.get('actions', [])):
            action_valid, issues = self.validate_action(action, robot_capabilities)
            if not action_valid:
                validation_result['is_valid'] = False
                validation_result['issues'].extend(issues)

        # Check safety constraints
        safety_valid, safety_issues = self.check_safety_constraints(plan)
        if not safety_valid:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(safety_issues)

        # Check resource availability
        resource_valid, resource_issues = self.check_resource_availability(plan, robot_capabilities)
        if not resource_valid:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(resource_issues)

        return validation_result

    def validate_action(self, action: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> tuple:
        """Validate individual action against robot capabilities"""
        issues = []

        action_name = action.get('name')
        if not action_name:
            issues.append("Action missing name")
            return False, issues

        # Check if action is supported by robot
        if action_name not in robot_capabilities.get('supported_actions', []):
            issues.append(f"Action '{action_name}' not supported by robot")
            return False, issues

        # Check action parameters
        required_params = robot_capabilities.get('action_requirements', {}).get(action_name, [])
        action_params = action.get('parameters', {})

        for param in required_params:
            if param not in action_params:
                issues.append(f"Missing required parameter '{param}' for action '{action_name}'")

        # Check preconditions
        preconditions = action.get('preconditions', [])
        for precondition in preconditions:
            if not self.check_precondition(precondition, robot_capabilities):
                issues.append(f"Unmet precondition: {precondition}")

        return len(issues) == 0, issues

    def check_precondition(self, precondition: str, robot_capabilities: Dict[str, Any]) -> bool:
        """Check if precondition is met"""
        # This would check actual robot state
        # For example: "navigation_system_ready", "gripper_empty", etc.
        return True  # Simplified for example

    def check_safety_constraints(self, plan: Dict[str, Any]) -> tuple:
        """Check safety constraints for the plan"""
        issues = []

        # Check for navigation safety
        nav_actions = [a for a in plan.get('actions', []) if a.get('name') == 'NAVIGATE_TO_LOCATION']
        for action in nav_actions:
            if not self.is_navigation_safe(action):
                issues.append("Unsafe navigation action detected")

        # Check for manipulation safety
        manipulation_actions = [a for a in plan.get('actions', [])
                              if a.get('name') in ['PICK_UP_OBJECT', 'PLACE_OBJECT']]
        for action in manipulation_actions:
            if not self.is_manipulation_safe(action):
                issues.append("Unsafe manipulation action detected")

        return len(issues) == 0, issues

    def is_navigation_safe(self, action: Dict[str, Any]) -> bool:
        """Check if navigation action is safe"""
        # Check if target location is in safe area
        # Check if path is clear of obstacles
        # Check if navigation is allowed in current environment
        return True  # Simplified for example

    def is_manipulation_safe(self, action: Dict[str, Any]) -> bool:
        """Check if manipulation action is safe"""
        # Check if object is safe to manipulate
        # Check if location is safe for manipulation
        # Check if human is in safe distance
        return True  # Simplified for example

    def check_resource_availability(self, plan: Dict[str, Any], robot_capabilities: Dict[str, Any]) -> tuple:
        """Check if required resources are available"""
        issues = []

        # Check battery level for plan execution
        estimated_battery_usage = self.estimate_battery_usage(plan)
        current_battery = robot_capabilities.get('battery_level', 100)

        if estimated_battery_usage > current_battery * 0.8:  # Leave 20% margin
            issues.append(f"Insufficient battery for plan execution (estimated: {estimated_battery_usage}%, current: {current_battery}%)")

        return len(issues) == 0, issues

    def estimate_battery_usage(self, plan: Dict[str, Any]) -> float:
        """Estimate battery usage for plan execution"""
        # Simplified estimation based on action types
        usage = 0.0
        for action in plan.get('actions', []):
            action_name = action.get('name', '')
            if 'NAVIGATE' in action_name:
                usage += 5.0  # 5% per navigation action
            elif 'PICK' in action_name or 'PLACE' in action_name:
                usage += 2.0  # 2% per manipulation action
            elif 'SPEAK' in action_name:
                usage += 0.1  # 0.1% per speech action

        return min(usage, 100.0)  # Cap at 100%
```

## ROS 2 Integration

### Cognitive Planning Node

Implementing the cognitive planning system as a ROS 2 node:

```python
# Example: Cognitive planning ROS 2 node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Initialize planning components
        self.planner = HierarchicalPlanner(self.get_planner_config())
        self.validator = PlanValidator()
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Create subscribers
        self.command_sub = self.create_subscription(
            String, '/high_level_command', self.command_callback, 10
        )
        self.context_sub = self.create_subscription(
            String, '/environment_context', self.context_callback, 10
        )

        # Create publishers
        self.action_pub = self.create_publisher(
            String, '/low_level_action', 10
        )
        self.plan_status_pub = self.create_publisher(
            String, '/plan_status', 10
        )
        self.feedback_pub = self.create_publisher(
            String, '/planning_feedback', 10
        )

        # Planning state
        self.current_plan = None
        self.plan_active = False

        self.get_logger().info("Cognitive planning node initialized")

    def get_planner_config(self):
        """Get planner configuration from parameters"""
        config = type('Config', (), {})()
        config.openai_api_key = self.declare_parameter('openai_api_key', '').value
        config.llm_model = self.declare_parameter('llm_model', 'gpt-4').value
        config.context_window = self.declare_parameter('context_window', 4096).value
        return config

    def command_callback(self, msg):
        """Process high-level command"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        # Get current context
        context = self.planner.context_manager.get_context()

        # Plan asynchronously
        future = asyncio.run_coroutine_threadsafe(
            self.process_command_async(command, context),
            asyncio.get_event_loop()
        )

    async def process_command_async(self, command: str, context: dict):
        """Process command asynchronously"""
        try:
            # Generate plan using LLM
            plan = await self.planner.generate_plan(command, context)

            # Validate plan
            validation_result = self.validator.validate_plan(plan, self.get_robot_capabilities())

            if validation_result['is_valid']:
                # Execute plan
                await self.execute_plan(plan)
            else:
                # Handle invalid plan
                self.handle_invalid_plan(validation_result, command)

        except Exception as e:
            self.get_logger().error(f"Planning error: {e}")
            self.publish_feedback(f"Planning failed: {str(e)}")

    def context_callback(self, msg):
        """Update context from environment data"""
        try:
            context_data = json.loads(msg.data)
            self.planner.context_manager.update_context(context_data)
        except json.JSONDecodeError:
            self.get_logger().error("Invalid context JSON received")

    async def execute_plan(self, plan: Dict[str, Any]):
        """Execute the planned actions"""
        self.plan_active = True
        self.current_plan = plan

        self.get_logger().info(f"Executing plan with {len(plan.get('actions', []))} actions")

        for i, action in enumerate(plan.get('actions', [])):
            if not self.plan_active:
                break

            self.get_logger().info(f"Executing action {i+1}/{len(plan.get('actions', []))}: {action.get('name')}")

            # Publish action
            action_msg = String()
            action_msg.data = json.dumps(action)
            self.action_pub.publish(action_msg)

            # Wait for action completion
            if not await self.wait_for_action_completion(action):
                self.get_logger().error(f"Action failed: {action}")
                break

        self.plan_active = False
        self.publish_plan_status("COMPLETED")

    async def wait_for_action_completion(self, action: Dict[str, Any]) -> bool:
        """Wait for action completion with timeout"""
        timeout = action.get('timeout', 30.0)  # Default 30 seconds
        start_time = self.get_clock().now().nanoseconds / 1e9

        while (self.get_clock().now().nanoseconds / 1e9 - start_time) < timeout:
            # Check if action is complete
            # This would interface with action execution feedback
            await asyncio.sleep(0.1)

        return True  # Simplified for example

    def handle_invalid_plan(self, validation_result: Dict[str, Any], original_command: str):
        """Handle invalid plan with recovery"""
        issues = validation_result['issues']
        self.get_logger().error(f"Invalid plan generated for command '{original_command}': {issues}")

        # Try to modify plan based on suggestions
        if validation_result.get('suggestions'):
            modified_plan = self.modify_plan_with_suggestions(
                self.current_plan,
                validation_result['suggestions']
            )
            if modified_plan:
                asyncio.run_coroutine_threadsafe(
                    self.execute_plan(modified_plan),
                    asyncio.get_event_loop()
                )
                return

        # Fall back to simple response
        self.publish_feedback(f"Cannot execute command: {original_command}. Issues: {', '.join(issues)}")

    def modify_plan_with_suggestions(self, plan: Dict[str, Any], suggestions: List[str]) -> Optional[Dict[str, Any]]:
        """Modify plan based on validation suggestions"""
        # Implementation would modify the plan based on suggestions
        # This is a simplified placeholder
        return None

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get current robot capabilities"""
        return {
            'supported_actions': [
                'NAVIGATE_TO_LOCATION',
                'PICK_UP_OBJECT',
                'PLACE_OBJECT',
                'DETECT_OBJECT',
                'FOLLOW_HUMAN',
                'SPEAK',
                'GREET_HUMAN'
            ],
            'action_requirements': {
                'NAVIGATE_TO_LOCATION': ['target'],
                'PICK_UP_OBJECT': ['object_name'],
                'PLACE_OBJECT': ['location'],
                'DETECT_OBJECT': ['object_name']
            },
            'battery_level': 85.0,  # Example battery level
            'navigation_system_ready': True,
            'manipulation_system_ready': True
        }

    def publish_plan_status(self, status: str):
        """Publish plan execution status"""
        status_msg = String()
        status_msg.data = status
        self.plan_status_pub.publish(status_msg)

    def publish_feedback(self, feedback: str):
        """Publish planning feedback"""
        feedback_msg = String()
        feedback_msg.data = feedback
        self.feedback_pub.publish(feedback_msg)
```

## Advanced Planning Techniques

### Reactive Planning

Implementing reactive planning for dynamic environments:

```python
# Example: Reactive planning for dynamic environments
class ReactivePlanner:
    def __init__(self):
        self.base_plan = None
        self.current_plan_index = 0
        self.reactive_rules = self.define_reactive_rules()

    def define_reactive_rules(self) -> List[Dict[str, Any]]:
        """Define reactive rules for plan adaptation"""
        return [
            {
                'condition': 'obstacle_detected',
                'action': 'replan_path',
                'priority': 'high'
            },
            {
                'condition': 'object_moved',
                'action': 'update_object_location',
                'priority': 'medium'
            },
            {
                'condition': 'human_approaching',
                'action': 'yield_right_of_way',
                'priority': 'high'
            },
            {
                'condition': 'battery_low',
                'action': 'return_to_charging_station',
                'priority': 'critical'
            }
        ]

    def update_plan_for_environment(self, environment_changes: Dict[str, Any], current_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update plan based on environment changes"""
        updated_plan = current_plan.copy()

        for change_type, change_data in environment_changes.items():
            for rule in self.reactive_rules:
                if rule['condition'] == change_type:
                    if rule['priority'] == 'critical':
                        # Insert critical action at beginning
                        critical_action = self.create_critical_action(rule['action'], change_data)
                        updated_plan.insert(0, critical_action)
                    elif rule['priority'] == 'high':
                        # Insert high-priority action after current action
                        next_action = self.create_reaction_action(rule['action'], change_data)
                        updated_plan.insert(self.current_plan_index + 1, next_action)
                    else:
                        # For medium/low priority, modify existing plan
                        updated_plan = self.modify_plan_for_condition(updated_plan, rule, change_data)

        return updated_plan

    def create_critical_action(self, action_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create critical action that takes immediate precedence"""
        if action_type == 'return_to_charging_station':
            return {
                'name': 'NAVIGATE_TO_LOCATION',
                'parameters': {'target': 'charging_station'},
                'priority': 'critical',
                'interrupt_current': True
            }
        elif action_type == 'yield_right_of_way':
            return {
                'name': 'STOP_ROBOT',
                'parameters': {'duration': 5.0},
                'priority': 'critical',
                'interrupt_current': True
            }

    def create_reaction_action(self, action_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create reactive action to handle environmental change"""
        if action_type == 'replan_path':
            return {
                'name': 'NAVIGATE_TO_LOCATION',
                'parameters': {'target': data.get('original_target'), 'avoid_obstacles': True},
                'priority': 'high'
            }
        elif action_type == 'update_object_location':
            return {
                'name': 'DETECT_OBJECT',
                'parameters': {'object_name': data.get('object_name')},
                'priority': 'medium'
            }

    def modify_plan_for_condition(self, plan: List[Dict[str, Any]], rule: Dict[str, Any], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Modify existing plan to accommodate condition"""
        modified_plan = plan.copy()

        if rule['action'] == 'replan_path':
            # Find navigation actions and update them
            for i, action in enumerate(modified_plan):
                if action.get('name') == 'NAVIGATE_TO_LOCATION':
                    action['parameters']['avoid_obstacles'] = True
                    action['parameters']['use_dynamic_map'] = True

        return modified_plan
```

### Multi-Agent Coordination

For scenarios with multiple robots or human-robot interaction:

```python
# Example: Multi-agent coordination in cognitive planning
class MultiAgentCoordinator:
    def __init__(self):
        self.agents = {}
        self.resource_manager = ResourceManager()
        self.conflict_resolver = ConflictResolver()

    def coordinate_plan(self, individual_plans: List[Dict[str, Any]], agent_ids: List[str]) -> List[Dict[str, Any]]:
        """Coordinate multiple agent plans to avoid conflicts"""
        coordinated_plans = []

        for i, plan in enumerate(individual_plans):
            agent_id = agent_ids[i]
            coordinated_plan = self.resolve_conflicts(plan, agent_id, individual_plans, agent_ids)
            coordinated_plans.append(coordinated_plan)

        return coordinated_plans

    def resolve_conflicts(self, plan: Dict[str, Any], agent_id: str, all_plans: List[Dict[str, Any]], all_agents: List[str]) -> Dict[str, Any]:
        """Resolve conflicts between agent plans"""
        resolved_plan = plan.copy()

        # Check for resource conflicts
        resource_conflicts = self.resource_manager.check_resource_conflicts(plan, agent_id, all_plans, all_agents)
        if resource_conflicts:
            resolved_plan = self.conflict_resolver.resolve_resource_conflicts(
                resolved_plan, resource_conflicts, agent_id
            )

        # Check for spatial conflicts
        spatial_conflicts = self.check_spatial_conflicts(plan, agent_id, all_plans, all_agents)
        if spatial_conflicts:
            resolved_plan = self.conflict_resolver.resolve_spatial_conflicts(
                resolved_plan, spatial_conflicts, agent_id
            )

        # Check for temporal conflicts
        temporal_conflicts = self.check_temporal_conflicts(plan, agent_id, all_plans, all_agents)
        if temporal_conflicts:
            resolved_plan = self.conflict_resolver.resolve_temporal_conflicts(
                resolved_plan, temporal_conflicts, agent_id
            )

        return resolved_plan

    def check_spatial_conflicts(self, plan: Dict[str, Any], agent_id: str, all_plans: List[Dict[str, Any]], all_agents: List[str]) -> List[Dict[str, Any]]:
        """Check for spatial conflicts between agent plans"""
        conflicts = []

        # This would involve checking planned paths and locations
        # for potential collisions or interference
        return conflicts

    def check_temporal_conflicts(self, plan: Dict[str, Any], agent_id: str, all_plans: List[Dict[str, Any]], all_agents: List[str]) -> List[Dict[str, Any]]:
        """Check for temporal conflicts between agent plans"""
        conflicts = []

        # This would involve checking timing of actions
        # to avoid interference
        return conflicts

class ResourceManager:
    def __init__(self):
        self.resources = {
            'charging_stations': [],
            'manipulation_zones': [],
            'navigation_corridors': []
        }
        self.resource_assignments = {}

    def check_resource_conflicts(self, plan: Dict[str, Any], agent_id: str, all_plans: List[Dict[str, Any]], all_agents: List[str]) -> List[Dict[str, Any]]:
        """Check for resource conflicts in plans"""
        conflicts = []

        for action in plan.get('actions', []):
            resource_needed = self.get_resource_requirement(action)
            if resource_needed:
                for other_agent_id, other_plan in zip(all_agents, all_plans):
                    if other_agent_id != agent_id:
                        for other_action in other_plan.get('actions', []):
                            other_resource = self.get_resource_requirement(other_action)
                            if other_resource == resource_needed:
                                # Check for temporal overlap
                                if self.resources_conflict_in_time(action, other_action):
                                    conflicts.append({
                                        'resource': resource_needed,
                                        'agents': [agent_id, other_agent_id],
                                        'actions': [action, other_action]
                                    })

        return conflicts

    def get_resource_requirement(self, action: Dict[str, Any]) -> Optional[str]:
        """Get resource requirement for action"""
        action_name = action.get('name', '')
        if 'CHARGE' in action_name:
            return 'charging_station'
        elif 'MANIPULATE' in action_name or 'PICK' in action_name or 'PLACE' in action_name:
            return 'manipulation_zone'
        elif 'NAVIGATE' in action_name:
            return 'navigation_path'
        return None

    def resources_conflict_in_time(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if resource usage conflicts in time"""
        # Simplified check - in practice would involve detailed timing analysis
        return True
```

## Performance Optimization

### Plan Caching and Reuse

Optimizing cognitive planning through caching and pattern recognition:

```python
# Example: Plan caching and pattern recognition
import hashlib
from datetime import datetime, timedelta

class PlanCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size

    def get_cached_plan(self, command: str, context_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached plan if available"""
        cache_key = self.generate_cache_key(command, context_hash)

        if cache_key in self.cache:
            # Update access time
            self.access_times[cache_key] = datetime.now()
            return self.cache[cache_key]

        return None

    def cache_plan(self, command: str, context_hash: str, plan: Dict[str, Any]):
        """Cache a plan for future use"""
        cache_key = self.generate_cache_key(command, context_hash)

        # Clean up if cache is full
        if len(self.cache) >= self.max_size:
            self.cleanup_oldest()

        self.cache[cache_key] = plan
        self.access_times[cache_key] = datetime.now()

    def generate_cache_key(self, command: str, context_hash: str) -> str:
        """Generate cache key from command and context"""
        key_string = f"{command}:{context_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def cleanup_oldest(self):
        """Remove oldest entries from cache"""
        if not self.access_times:
            return

        # Find oldest entry
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        # Remove oldest entry
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

class PatternRecognizer:
    def __init__(self):
        self.patterns = {}
        self.command_history = []

    def recognize_pattern(self, command: str) -> Optional[str]:
        """Recognize command pattern for optimized planning"""
        # Simple pattern matching
        command_lower = command.lower()

        # Common command patterns
        patterns = [
            (r'go to (kitchen|living room|bedroom|office)', 'NAVIGATION_TO_ROOM'),
            (r'pick up (cup|book|ball|box)', 'PICKUP_OBJECT'),
            (r'bring me (water|coffee|book)', 'DELIVERY_TASK'),
            (r'follow (me|the person)', 'FOLLOW_TASK')
        ]

        for pattern, task_type in patterns:
            import re
            if re.match(pattern, command_lower):
                return task_type

        return None

    def learn_from_history(self, command: str, successful_plan: List[Dict[str, Any]]):
        """Learn from successful plan executions"""
        self.command_history.append({
            'command': command,
            'plan': successful_plan,
            'timestamp': datetime.now()
        })

        # Keep only recent history
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-500:]
```

## Error Handling and Recovery

### Robust Planning with Error Recovery

Implementing robust planning with error handling and recovery:

```python
# Example: Robust planning with error recovery
class RobustPlanner:
    def __init__(self):
        self.recovery_strategies = self.define_recovery_strategies()

    def define_recovery_strategies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Define recovery strategies for different failure types"""
        return {
            'NAVIGATION_FAILURE': [
                {'action': 'use_alternative_path', 'priority': 1},
                {'action': 'request_human_assistance', 'priority': 2},
                {'action': 'return_to_known_location', 'priority': 3}
            ],
            'MANIPULATION_FAILURE': [
                {'action': 'adjust_grasp_approach', 'priority': 1},
                {'action': 'request_object_repositioning', 'priority': 2},
                {'action': 'use_alternative_manipulation_method', 'priority': 3}
            ],
            'PERCEPTION_FAILURE': [
                {'action': 'change_sensor_config', 'priority': 1},
                {'action': 'move_to_better_viewpoint', 'priority': 2},
                {'action': 'request_human_verification', 'priority': 3}
            ],
            'COMMUNICATION_FAILURE': [
                {'action': 'use_alternative_communication_method', 'priority': 1},
                {'action': 'retry_communication', 'priority': 2},
                {'action': 'use_predefined_response', 'priority': 3}
            ]
        }

    async def execute_plan_with_recovery(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan with built-in recovery mechanisms"""
        execution_result = {
            'success': True,
            'completed_actions': [],
            'failed_actions': [],
            'recovery_attempts': 0
        }

        for i, action in enumerate(plan.get('actions', [])):
            try:
                action_result = await self.execute_action_with_monitoring(action)

                if action_result['success']:
                    execution_result['completed_actions'].append(action)
                else:
                    # Attempt recovery
                    recovery_result = await self.attempt_recovery(
                        action, action_result, plan
                    )

                    if recovery_result['success']:
                        execution_result['completed_actions'].append(action)
                        execution_result['recovery_attempts'] += 1
                    else:
                        execution_result['failed_actions'].append({
                            'action': action,
                            'error': recovery_result.get('error', 'Unknown error'),
                            'recovery_failed': True
                        })
                        execution_result['success'] = False
                        break

            except Exception as e:
                execution_result['failed_actions'].append({
                    'action': action,
                    'error': str(e),
                    'recovery_failed': False
                })
                execution_result['success'] = False
                break

        return execution_result

    async def execute_action_with_monitoring(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with monitoring and early failure detection"""
        result = {
            'success': False,
            'action': action,
            'execution_time': 0.0,
            'error': None
        }

        start_time = datetime.now()

        try:
            # Execute action (this would interface with actual robot systems)
            # For example, publish action to appropriate ROS topic
            action_msg = String()
            action_msg.data = json.dumps(action)

            # Wait for action completion with timeout
            timeout = action.get('timeout', 30.0)

            # In a real implementation, this would wait for action feedback
            # and monitor for failure conditions
            await asyncio.sleep(min(timeout, 1.0))  # Simulated execution

            result['success'] = True
            result['execution_time'] = (datetime.now() - start_time).total_seconds()

        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = (datetime.now() - start_time).total_seconds()

        return result

    async def attempt_recovery(self, failed_action: Dict[str, Any], failure_info: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery from action failure"""
        failure_type = self.classify_failure(failed_action, failure_info)
        recovery_strategies = self.recovery_strategies.get(failure_type, [])

        for strategy in sorted(recovery_strategies, key=lambda x: x['priority']):
            try:
                recovery_result = await self.execute_recovery_strategy(
                    strategy['action'], failed_action, failure_info, plan
                )

                if recovery_result['success']:
                    return recovery_result

            except Exception as e:
                continue  # Try next recovery strategy

        # All recovery strategies failed
        return {
            'success': False,
            'error': f"All recovery strategies failed for {failure_type}",
            'original_failure': failure_info
        }

    def classify_failure(self, action: Dict[str, Any], failure_info: Dict[str, Any]) -> str:
        """Classify type of failure for appropriate recovery"""
        action_name = action.get('name', '').upper()

        if 'NAVIGATE' in action_name:
            return 'NAVIGATION_FAILURE'
        elif 'PICK' in action_name or 'PLACE' in action_name or 'GRASP' in action_name:
            return 'MANIPULATION_FAILURE'
        elif 'DETECT' in action_name or 'RECOGNIZE' in action_name:
            return 'PERCEPTION_FAILURE'
        else:
            return 'GENERAL_FAILURE'

    async def execute_recovery_strategy(self, strategy: str, failed_action: Dict[str, Any], failure_info: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific recovery strategy"""
        if strategy == 'use_alternative_path':
            # Modify navigation action with alternative route
            if 'NAVIGATE' in failed_action.get('name', '').upper():
                alternative_action = failed_action.copy()
                alternative_action['parameters']['use_alternative_route'] = True
                return await self.execute_action_with_monitoring(alternative_action)

        elif strategy == 'adjust_grasp_approach':
            # Modify manipulation action with different approach
            if any(keyword in failed_action.get('name', '').upper() for keyword in ['PICK', 'GRASP', 'PLACE']):
                alternative_action = failed_action.copy()
                alternative_action['parameters']['approach_angle'] = self.calculate_alternative_approach(
                    alternative_action['parameters']
                )
                return await self.execute_action_with_monitoring(alternative_action)

        # Add more recovery strategies as needed
        return {'success': False, 'error': f"Recovery strategy '{strategy}' not implemented"}
```

## Best Practices

### Design Considerations

When implementing cognitive planning systems, consider:

1. **Safety First**: Always prioritize safety in plan generation and execution
2. **Context Awareness**: Use environmental and temporal context for better planning
3. **Robustness**: Handle failures gracefully with recovery mechanisms
4. **Efficiency**: Optimize for real-time performance while maintaining quality
5. **Adaptability**: Allow plans to adapt to changing conditions
6. **Explainability**: Provide clear reasoning for planning decisions

### Performance Optimization

- Cache frequently used plans to reduce computation time
- Use hierarchical planning to break down complex tasks
- Implement parallel execution where possible
- Monitor resource usage and plan accordingly
- Use pattern recognition to speed up common tasks

## Summary

Cognitive planning is the intelligent layer that transforms high-level commands into executable robot actions. By integrating LLMs, context management, and robust error handling, we create systems that can handle complex, real-world scenarios while maintaining safety and reliability. The key to effective cognitive planning lies in balancing sophistication with practical constraints, ensuring that plans are both intelligent and executable in real-world humanoid robot applications.