#!/usr/bin/env python3
"""
Action server example that implements a Fibonacci sequence generator.
This node provides a Fibonacci action that generates a sequence of numbers.
"""

import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from example_interfaces.action import Fibonacci


class MinimalActionServer(Node):

    def __init__(self):
        super().__init__('minimal_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')

        # Send initial feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        goal_handle.publish_feedback(feedback_msg)

        # Generate the Fibonacci sequence
        sequence = [0, 1]
        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Goal was cancelled')
                goal_handle.canceled()
                result = Fibonacci.Result()
                result.sequence = sequence
                return result

            sequence.append(sequence[i] + sequence[i-1])
            feedback_msg.sequence = sequence
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.1)  # Simulate work

        # Check if the goal was cancelled
        if goal_handle.is_cancel_requested:
            self.get_logger().info('Goal was cancelled')
            goal_handle.canceled()
            result = Fibonacci.Result()
            result.sequence = sequence
            return result

        # Complete the goal
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence

        self.get_logger().info(f'Result: {sequence}')
        return result


def main(args=None):
    rclpy.init(args=args)

    minimal_action_server = MinimalActionServer()

    try:
        rclpy.spin(minimal_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_action_server.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()