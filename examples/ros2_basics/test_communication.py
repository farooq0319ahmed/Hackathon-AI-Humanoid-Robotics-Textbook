#!/usr/bin/env python3
"""
Validation script to test ROS 2 node communication examples.
This script validates that publisher and subscriber nodes can communicate properly.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from threading import Thread


class CommunicationValidator(Node):
    def __init__(self):
        super().__init__('communication_validator')

        # Create publisher to send test messages
        self.publisher = self.create_publisher(String, 'test_topic', 10)

        # Create subscriber to receive test messages
        self.subscription = self.create_subscription(
            String,
            'test_topic',
            self.message_callback,
            10
        )

        self.received_messages = []
        self.expected_messages = []
        self.test_results = {
            'messages_sent': 0,
            'messages_received': 0,
            'success': False
        }

    def message_callback(self, msg):
        """Callback function to handle received messages."""
        self.received_messages.append(msg.data)
        self.test_results['messages_received'] += 1
        self.get_logger().info(f'Received: {msg.data}')

    def send_test_messages(self, num_messages=5):
        """Send a series of test messages."""
        for i in range(num_messages):
            msg = String()
            message_content = f'Test message {i+1}'
            msg.data = message_content
            self.publisher.publish(msg)
            self.expected_messages.append(message_content)
            self.test_results['messages_sent'] += 1
            self.get_logger().info(f'Sent: {message_content}')
            time.sleep(0.5)  # Wait 0.5 seconds between messages

    def validate_communication(self):
        """Validate that messages were sent and received correctly."""
        # Wait a bit more to ensure all messages are received
        time.sleep(2.0)

        success = (
            self.test_results['messages_sent'] == self.test_results['messages_received'] and
            len(self.received_messages) > 0
        )

        self.test_results['success'] = success

        # Print results
        print("\n" + "="*50)
        print("COMMUNICATION VALIDATION RESULTS")
        print("="*50)
        print(f"Messages sent: {self.test_results['messages_sent']}")
        print(f"Messages received: {self.test_results['messages_received']}")
        print(f"Communication successful: {success}")
        print(f"Expected messages: {self.expected_messages}")
        print(f"Received messages: {self.received_messages}")
        print("="*50)

        return success


def main(args=None):
    rclpy.init(args=args)

    validator = CommunicationValidator()

    # Create a thread to run the ROS spinner
    spinner_thread = Thread(target=rclpy.spin, args=(validator,))
    spinner_thread.start()

    try:
        # Wait a moment for setup
        time.sleep(1.0)

        # Send test messages
        validator.send_test_messages(5)

        # Wait for all messages to be processed
        time.sleep(3.0)

        # Validate results
        success = validator.validate_communication()

        # Shutdown
        rclpy.shutdown()
        spinner_thread.join()

        # Exit with appropriate code
        if success:
            print("✓ Communication validation PASSED")
            return 0
        else:
            print("✗ Communication validation FAILED")
            return 1

    except KeyboardInterrupt:
        print("Validation interrupted by user")
        rclpy.shutdown()
        spinner_thread.join()
        return 1


if __name__ == '__main__':
    exit(main())