#!/usr/bin/env python3
"""
Action client example that calls the Fibonacci action.
This node calls the 'fibonacci' action and prints feedback and results.
"""

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class MinimalActionClient(Node):

    def __init__(self):
        super().__init__('minimal_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order=10):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info(f'Sending goal request with order {order}...')

        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    action_client = MinimalActionClient()

    # Spin in a separate thread
    from threading import Thread
    spin_thread = Thread(target=rclpy.spin, args=(action_client,))
    spin_thread.start()

    # Send the goal
    action_client.send_goal(10)

    # Wait for the thread to finish
    spin_thread.join()


if __name__ == '__main__':
    main()