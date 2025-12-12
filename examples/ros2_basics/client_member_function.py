#!/usr/bin/env python3
"""
Service client example that calls the add_two_ints service.
This node calls the 'add_two_ints' service with two integers and prints the result.
"""

import sys
import rclpy
from rclpy.node import Node

from example_interfaces.srv import AddTwoInts


class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        return self.future


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClient()

    # Use command line arguments if provided, otherwise use default values
    if len(sys.argv) == 3:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    else:
        a = 1
        b = 2

    future = minimal_client.send_request(a, b)

    try:
        rclpy.spin_until_future_complete(minimal_client, future)
        if future.result() is not None:
            response = future.result()
            minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
        else:
            minimal_client.get_logger().info('Service call failed')
    except KeyboardInterrupt:
        pass
    finally:
        minimal_client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()