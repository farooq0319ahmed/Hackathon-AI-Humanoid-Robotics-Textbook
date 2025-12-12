from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'rate',
            default_value='1.0',
            description='Publish rate in Hz'
        ),

        # Publisher node
        Node(
            package='hello_robot_pkg',
            executable='publisher',
            name='talker',
            parameters=[
                {'rate': LaunchConfiguration('rate')}
            ],
            output='screen'
        ),

        # Subscriber node
        Node(
            package='hello_robot_pkg',
            executable='subscriber',
            name='listener',
            output='screen'
        )
    ])