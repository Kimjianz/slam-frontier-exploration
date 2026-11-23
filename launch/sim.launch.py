"""Use the installed TurtleBot3 model and the local team maze."""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    tb3 = Path(get_package_share_directory('turtlebot3_gazebo'))
    gz = Path(get_package_share_directory('ros_gz_sim'))
    root = Path(__file__).resolve().parents[1]
    model = root/'models/camera_burger/model.sdf'
    robot = root/'models/camera_burger/robot.urdf'
    if not model.is_file() or not robot.is_file():
        raise RuntimeError('먼저 maze_escape 폴더에서 bash setup.sh를 실행해줘.')
    def include(file, arguments):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(file)),
                                         launch_arguments=arguments.items())
    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',str(tb3/'models')),
        include(gz/'launch/gz_sim.launch.py',
                {'gz_args':'-r -s -v2 '+str(root/'worlds/escape_room.world'),
                 'on_exit_shutdown':'true'}),
        include(gz/'launch/gz_sim.launch.py',{'gz_args':'-g -v2','on_exit_shutdown':'true'}),
        Node(package='robot_state_publisher',executable='robot_state_publisher',
             name='robot_state_publisher',parameters=[{'use_sim_time':True,
                    'robot_description':robot.read_text()}],output='screen'),
        Node(package='ros_gz_sim',executable='create',
             arguments=['-name','burger','-file',str(model),'-x','-2','-y','-2.5','-z','.01'],output='screen'),
        Node(package='ros_gz_bridge',executable='parameter_bridge',name='ros_gz_bridge',
             parameters=[{'use_sim_time':True,'config_file':str(tb3/'params/turtlebot3_burger_bridge.yaml')}],
             output='screen'),
        Node(package='ros_gz_bridge',executable='parameter_bridge',name='camera_bridge',
             parameters=[{'use_sim_time':True,'config_file':str(root/'config/camera_bridge.yaml')}],output='screen'),
    ])
