---
sidebar_position: 2
---

# Expected Outputs: ROS 2 Examples

This document provides the expected outputs for each ROS 2 code example in Module 1.

## Publisher Node Example

**File:** `publisher_member_function.py`

**Expected Output:**
```
[INFO] [1699999999.999999999] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1699999999.999999999] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [1699999999.999999999] [minimal_publisher]: Publishing: "Hello World: 2"
...
```

**Behavior:**
- Publishes a message every 0.5 seconds
- Message counter increments with each publication
- Node continues running until interrupted

## Subscriber Node Example

**File:** `subscriber_member_function.py`

**Expected Output (when publisher is running):**
```
[INFO] [1699999999.999999999] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1699999999.999999999] [minimal_subscriber]: I heard: "Hello World: 1"
[INFO] [1699999999.999999999] [minimal_subscriber]: I heard: "Hello World: 2"
...
```

**Behavior:**
- Receives and logs messages from the publisher
- Messages appear approximately every 0.5 seconds
- Node continues running until interrupted

## Service Server Example

**File:** `service_member_function.py`

**Expected Output (when client calls the service):**
```
[INFO] [1699999999.999999999] [minimal_service]: Incoming request
a: 1, b: 2
Sum: 3
```

**Behavior:**
- Logs incoming requests with parameters
- Calculates and returns the sum
- Continues running to handle multiple requests

## Service Client Example

**File:** `client_member_function.py`

**Expected Output:**
```
[INFO] [1699999999.999999999] [minimal_client]: Service not available, waiting again...
[INFO] [1699999999.999999999] [minimal_client]: Result of add_two_ints: 3
```

**Behavior:**
- Waits for service if not immediately available
- Displays the result of the service call
- Exits after receiving the result

## Action Server Example

**File:** `action_server_member_function.py`

**Expected Output (when client sends goal):**
```
[INFO] [1699999999.999999999] [minimal_action_server]: Received goal request
[INFO] [1699999999.999999999] [minimal_action_server]: Executing goal...
[INFO] [1699999999.999999999] [minimal_action_server]: Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

**Behavior:**
- Logs goal receipt and execution
- Provides feedback during execution
- Returns final result sequence

## Action Client Example

**File:** `action_client_member_function.py`

**Expected Output:**
```
[INFO] [1699999999.999999999] [minimal_action_client]: Waiting for action server...
[INFO] [1699999999.999999999] [minimal_action_client]: Sending goal request with order 10...
[INFO] [1699999999.999999999] [minimal_action_client]: Received feedback: [0, 1]
[INFO] [1699999999.999999999] [minimal_action_client]: Received feedback: [0, 1, 1]
[INFO] [1699999999.999999999] [minimal_action_client]: Received feedback: [0, 1, 1, 2]
...
[INFO] [1699999999.999999999] [minimal_action_client]: Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

**Behavior:**
- Waits for action server to become available
- Sends goal and receives feedback during execution
- Displays final result

## Launch File Execution

**Command:** `ros2 launch examples_package publisher_subscriber_launch.py`

**Expected Output:**
```
[INFO] [launch]: All processes have started correctly...
[publisher-1] [INFO] [1699999999.999999999] [talker]: Publishing: "Hello World: 0"
[subscriber-1] [INFO] [1699999999.999999999] [listener]: I heard: "Hello World: 0"
...
[INFO] [launch]: Process has finished cleanly
```

**Behavior:**
- Both nodes start simultaneously
- Communication occurs as expected
- Nodes terminate cleanly when launch is stopped

## Parameter Loading

**Command:** `ros2 run package_name node_name --ros-args --params-file config/robot_params.yaml`

**Expected Output:**
```
[INFO] [1699999999.999999999] [node_name]: Parameter 'robot_name' loaded: simple_humanoid
[INFO] [1699999999.999999999] [node_name]: Parameter 'max_linear_velocity' loaded: 1.0
...
```

**Behavior:**
- Parameters loaded from YAML file
- Values applied to node
- Verification messages logged

## Validation Script Output

**File:** `test_communication.py`

**Expected Output:**
```
==================================================
COMMUNICATION VALIDATION RESULTS
==================================================
Messages sent: 5
Messages received: 5
Communication successful: True
Expected messages: ['Test message 1', 'Test message 2', ...]
Received messages: ['Test message 1', 'Test message 2', ...]
==================================================
✓ Communication validation PASSED
```

**Behavior:**
- Sends test messages and validates receipt
- Reports success/failure status
- Provides detailed results