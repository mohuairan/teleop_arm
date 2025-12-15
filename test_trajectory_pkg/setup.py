from setuptools import find_packages, setup

package_name = 'test_trajectory_pkg'

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
    maintainer='jodell',
    maintainer_email='jodell@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'receive_joint_angle = test_trajectory_pkg.receive_joint_angle:main',
            'transmit_joint_angle = test_trajectory_pkg.transmit_joint_angle:main',
            'receive_from_aieyes = test_trajectory_pkg.receive_from_aieyes:main'
        ],
    },
)
