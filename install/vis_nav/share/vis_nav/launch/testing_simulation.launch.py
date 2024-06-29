#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_file_name = 'world_umi_without_robot.world'
    test = get_package_share_directory('vis_nav')
    #/home/regmed/dregmed/dev_ws_vn/src/vis_nav/worlds/environnement.world
    print(f'testing environnemnt = {test}')
    world = os.path.join(get_package_share_directory('vis_nav'), 'world', world_file_name)
    launch_file_dir = '/home/regmed/dregmed/vis_to_nav/install/vis_nav/share/vis_nav/launch'
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')


    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

         IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
            ),
        ),

        Node(package='vis_nav',
             executable='testing.py',
             output='screen'
        ),
	
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

    ])



