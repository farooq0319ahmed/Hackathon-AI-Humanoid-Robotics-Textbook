---
sidebar_position: 2
---

# ROS 2 Architecture

## Overview

ROS 2 (Robot Operating System 2) is not an operating system but rather a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms and environments.

The architecture of ROS 2 is designed around a distributed system where processes called "nodes" communicate with each other using a publish/subscribe messaging model, services, and actions.

## Core Concepts

### Nodes

A **node** is an executable that uses ROS 2 to communicate with other nodes. Nodes are the fundamental building blocks of a ROS 2 program. Each node is designed to perform a specific task and communicate with other nodes to achieve complex robot behaviors.

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        # Node initialization code here
```

**Key characteristics of nodes:**
- Each node runs in its own process
- Nodes can be written in different programming languages (C++, Python, etc.)
- Nodes communicate with each other through topics, services, or actions
- A single system can have many nodes running simultaneously

### Topics and Messages

**Topics** are named buses over which nodes exchange messages. The communication is based on a publish/subscribe model where publishers send messages to topics and subscribers receive messages from topics.

**Messages** are the data structures that are sent between nodes. They are defined using a special interface definition language (IDL) and are strongly typed.

```python
# Publisher example
publisher = self.create_publisher(String, 'topic_name', 10)
msg = String()
msg.data = 'Hello World'
publisher.publish(msg)
```

**Characteristics of topics:**
- Unidirectional communication (publisher → subscriber)
- Multiple publishers and subscribers can exist for a single topic
- Data is sent continuously while both nodes are active
- No guaranteed delivery of messages

### Services

**Services** provide a request/reply communication model. A service client sends a request to a service server, which processes the request and sends back a response.

```python
# Service server example
from example_interfaces.srv import AddTwoInts

def add_two_ints_callback(request, response):
    response.sum = request.a + request.b
    return response

service = self.create_service(AddTwoInts, 'add_two_ints', add_two_ints_callback)
```

**Characteristics of services:**
- Synchronous communication
- Request-response pattern
- Request is processed completely before response is sent
- Suitable for operations that have a clear beginning and end

### Actions

**Actions** are a more advanced communication pattern that combines features of topics and services. They are used for long-running tasks that may take a significant amount of time to complete.

```python
# Action server example
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)
```

**Characteristics of actions:**
- Asynchronous communication
- Support for feedback during execution
- Can be canceled during execution
- Suitable for long-running tasks with intermediate results

## Communication Patterns Comparison

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Publish/Subscribe | Continuous data streams | Unidirectional, no delivery guarantee |
| Services | Request/Response | Short operations | Synchronous, request-response |
| Actions | Goal/Result/Feedback | Long-running tasks | Asynchronous, cancellable, feedback |

## Quality of Service (QoS)

ROS 2 provides Quality of Service (QoS) settings that allow you to configure how messages are delivered between nodes. This is particularly important for real-time systems and safety-critical applications.

Key QoS settings include:
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep all messages vs. keep last N messages

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy

qos_profile = QoSProfile(depth=10)
qos_profile.reliability = ReliabilityPolicy.RELIABLE
```

## Lifecycle Nodes

ROS 2 introduces lifecycle nodes that provide a standardized way to manage the state of a node through well-defined transitions. This is useful for managing complex systems where components need to be initialized, activated, and deactivated in a coordinated manner.

Lifecycle states include:
- Unconfigured → Inactive → Active → Finalized

## Summary

The ROS 2 architecture provides a flexible and robust framework for robot software development. Understanding these core concepts is essential for building effective robot applications. The modular design allows for distributed processing while maintaining clear communication patterns between different components.

In the next section, we'll dive deeper into the specific patterns of communication: nodes, topics, services, and actions.