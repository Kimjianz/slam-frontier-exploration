from setuptools import find_packages, setup

package_name = 'escape_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='woojin',
    maintainer_email='woojin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'detector_node = escape_robot.detector_node:main',
                'approach_node = escape_robot.approach_node:main',
                'fsm_node = escape_robot.fsm_node:main',
                'escape_node = escape_robot.escape_node:main'
        ],
    },
)
