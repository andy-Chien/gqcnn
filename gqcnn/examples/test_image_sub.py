#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Displays robust grasps planned using a GQ-CNN-based policy on a set of saved
RGB-D images. The default configuration is cfg/examples/policy.yaml.

Author
------
Jeff Mahler
"""
import argparse
import logging
import numpy as np
import os
# import rosgraph.roslogging as rl
# import rospy
import sys
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import cv2
from cv_bridge import CvBridge, CvBridgeError

from autolab_core import (Point, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage)
from visualization import Visualizer2D as vis

from gqcnn.grasping import Grasp2D, SuctionPoint2D, GraspAction
from gqcnn_interfaces.msg import GQCNNGrasp
from gqcnn_interfaces.srv import GQCNNGraspPlanner, GQCNNGraspPlannerSegmask
from sensor_msgs.msg import CameraInfo, Image

# Set up logger.
logger = Logger.get_logger("examples/policy_camera_ros2.py")
# color_camera_msg = CameraInfo()
# depth_camera_msg = CameraInfo()
# depth_im = None
# def color_camera_Info_callback(msg):
#     camera_msg = msg
#     return camera_msg


class depth_image_Subscriber(Node):

    def __init__(self):
        super().__init__("depth_image_Subscriber")
        depth_camera_info_topic_ = '/camera/depth/camera_info'
        depth_image_topic_ = '/camera/depth/image_rect_raw'
        self.bridge = CvBridge()
        # color_camera_info_subscriber_ = node.create_subscription(CameraInfo, color_camera_info_topic_, color_camera_Info_callback, 10)
        # self.depth_camera_info_subscriber_ = self.create_subscription(CameraInfo, depth_camera_info_topic_,
        #                                                           self.depth_camera_Info_callback, 10)
        self.depth_image_subscriber_ = self.create_subscription(Image, depth_image_topic_, 
                                                           self.depth_image_Info_callback, 10)

        # self.depth_camera_info_subscriber_  # prevent unused variable warning
        self.has_received = False

    def depth_camera_Info_callback(self, msg):
        self.depth_camera_msg = msg
        return self.depth_camera_msg

    def depth_image_Info_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.height)
        print("-------------")
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_im = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # self.depth_im = np.array(msg.data, dtype=np.uint8)
        print(self.depth_im)
        self.has_received = True
        # self.destroy_subscription(self.depth_image_subscriber_)
        # return

        # rclpy.shutdown()
        # rclpy.shutdown()
        # return self.depth_im

def main(args=None):
    # Parse args.

    # Initialize the ROS node.
    rclpy.init(args=None)
    # node = rclpy.create_node('grasp_planning_example')
    depth_image_Sub = depth_image_Subscriber()

    
    # rclpy.spin(depth_image_Sub)
    # print("-----------")
    # depth_image_Sub.destroy_node()
    # rclpy.shutdown()
    # logging.getLogger().addHandler(rl.RosStreamHandler())

    # Setup filenames.

    while rclpy.ok() and not depth_image_Sub.has_received:
        rclpy.spin_once(depth_image_Sub)
    print("-----------")
    rclpy.shutdown()
if __name__ == "__main__":
    main()