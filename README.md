## Note: Python 2.x support has officially been dropped.

# Berkeley AUTOLAB's GQCNN Package
<p>
   <a href="https://travis-ci.org/BerkeleyAutomation/gqcnn/">
       <img alt="Build Status" src="https://travis-ci.org/BerkeleyAutomation/gqcnn.svg?branch=master">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/releases/latest">
       <img alt="Release" src="https://img.shields.io/github/release/BerkeleyAutomation/gqcnn.svg?style=flat">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/blob/master/LICENSE">
       <img alt="Software License" src="https://img.shields.io/badge/license-REGENTS-brightgreen.svg">
   </a>
   <a>
       <img alt="Python 3 Versions" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-yellow.svg">
   </a>
</p>

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs). It is part of the ongoing [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project created and maintained by the [AUTOLAB](https://autolab.berkeley.edu) at UC Berkeley.

## Installation and Usage
Please see the [docs](https://berkeleyautomation.github.io/gqcnn/) for installation and usage instructions.


## Installation and Usage with docker

1.clone docker to your <workspace_ws>.
```bash
git clone https://github.com/errrr0501/docker_20.04_CUDA12_tf1.15
 ```
    
2.build it and run it.


3.clone and build realsense_ros2_wrapper.
```bash
#open a new terminal
mkdir <your realsense workspace>
cd <your realsense workspace>
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-development
colcon build
```

4.make a workspace and clone gqcnn and autolab_core.
```bash
#open a new terminal
mkdir <your gqcnn workspace>
cd <your gqcnn workspace>
git clone https://github.com/errrr0501/ROS2_gqcnn.git
git clone https://github.com/errrr0501/ROS2_autolab_core.git
colcon build
```
    
5.use with your camera topic.
```bash
#open a new terminal
cd <your realsense workspace>
source install/setup.bash
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 align_depth.enable:=true
#open a new terminal
cd <your gqcnn workspace>
source install/setup.bash
ros2 launch gqcnn grasp_planning_service.launch.py
#open a new terminal
source install/setup.bash
python3 src/ROS2_gqcnn/gqcnn/examples/policy_camera_ros2.py
```
    
## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).

