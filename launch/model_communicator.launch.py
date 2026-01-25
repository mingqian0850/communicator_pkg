import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('communicator_pkg')
    params_path = os.path.join(pkg_share, 'config', 'communicator_config.yaml')
    return LaunchDescription([
        Node(
            package='communicator_pkg',
            executable='model_communicator',
            name='communicator_node',
            output='screen',
            parameters=[params_path],
        ),
    ])
