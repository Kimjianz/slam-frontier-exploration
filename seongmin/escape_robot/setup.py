from setuptools import find_packages, setup

package_name = 'escape_robot'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seongmin',
    maintainer_email='seongmin@todo.todo',
    description='LEVEL 1 autonomous escape room mission (FSM) for TurtleBot3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'escape_fsm = escape_robot.escape_fsm:main',
        ],
    },
)
