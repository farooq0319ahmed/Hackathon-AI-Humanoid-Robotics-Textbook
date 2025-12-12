---
sidebar_position: 3
---

# Nodes, Topics, and Services in Detail

## Deep Dive into Nodes

### Node Creation and Lifecycle

In ROS 2, nodes are the fundamental building blocks of any robot application. Each node represents a single process that performs a specific task and communicates with other nodes through ROS 2's communication infrastructure.

To create a node in Python, you inherit from the `Node` class and call the parent constructor with a unique node name:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('my_node_name')
        self.get_logger().info('MyNode has been started')
```

### Node Parameters

Nodes can accept parameters that can be configured at runtime, making them more flexible and reusable:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('frequency', 1.0)
        self.declare_parameter('robot_name', 'robot1')

        # Get parameter values
        self.frequency = self.get_parameter('frequency').value
        self.robot_name = self.get_parameter('robot_name').value
```

## Topics and Publishers

### Publisher Implementation

A publisher sends messages to a topic. Here's a complete example of creating and using a publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Implementation

A subscriber receives messages from a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) Settings

When creating publishers and subscribers, you can specify QoS settings to control how messages are delivered:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile with specific settings
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)

publisher = self.create_publisher(String, 'topic', qos_profile)
subscriber = self.create_subscription(String, 'topic', callback, qos_profile)
```

## Services

### Service Server Implementation

A service server processes requests and sends back responses:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Implementation

A service client sends requests to a service server:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info('Result of add_two_ints: %d' % response.sum)
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Communication Patterns

### Multiple Publishers and Subscribers

You can have multiple publishers on the same topic, and multiple subscribers listening to that topic:

```python
# Publisher 1
publisher1 = self.create_publisher(String, 'sensor_data', 10)

# Publisher 2 (different node)
publisher2 = self.create_publisher(String, 'sensor_data', 10)

# Multiple subscribers
subscriber1 = self.create_subscription(String, 'sensor_data', callback1, 10)
subscriber2 = self.create_subscription(String, 'sensor_data', callback2, 10)
```

### Message Filters

For more complex scenarios, you might need to synchronize messages from multiple topics:

```python
from rclpy.qos import QoSProfile
from message_filters import ApproximateTimeSynchronizer, Subscriber

# This is a conceptual example - actual implementation may vary
# Use approximate time synchronization for messages from different sensors
```

## Best Practices

### Node Design

- **Single Responsibility**: Each node should have a clear, single purpose
- **Error Handling**: Always include proper error handling and logging
- **Resource Management**: Clean up resources properly when the node shuts down
- **Parameter Validation**: Validate input parameters and provide meaningful error messages

### Topic Design

- **Naming Conventions**: Use clear, descriptive names (e.g., `/robot1/sensors/laser_scan`)
- **Message Types**: Use appropriate message types for your data
- **Frequency**: Consider the update frequency and its impact on system performance
- **Data Size**: Be mindful of message size to avoid network congestion

### Service Design

- **Statelessness**: Services should be stateless when possible
- **Timeout Handling**: Always implement proper timeout handling in clients
- **Error Reporting**: Return appropriate error codes in service responses
- **Synchronous Limitations**: Remember that services are synchronous and will block

## Summary

Understanding nodes, topics, and services is crucial for effective ROS 2 development. Nodes provide the structure for your robot application, topics enable continuous data flow between components, and services provide request-response communication for specific operations.

These patterns form the foundation for building more complex robot behaviors. In the next section, we'll explore URDF modeling for defining robot components.