from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='demo_nodes_py',
            executable='talker',
            name='publisher',
            parameters=[
                {'use_sim_time': False}
            ],
            remappings=[
                ('chatter', 'my_chatter')
            ]
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='subscriber',
            remappings=[
                ('chatter', 'my_chatter')
            ]
        )
    ])