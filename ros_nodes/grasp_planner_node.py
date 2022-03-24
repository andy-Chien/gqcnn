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

ROS Server for planning GQ-CNN grasps.

Author
-----
Vishal Satish & Jeff Mahler
"""
import json
import math
import os
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import quaternion as qtn
import rospy
from autolab_core import (YamlConfig, CameraIntrinsics, ColorImage,
                          DepthImage, BinaryImage, RgbdImage)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (Grasp2D, SuctionPoint2D, RgbdImageState,
                            RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode, NoValidGraspsException

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, Image

from task_msgs.srv import (GQCNNGraspPlanner, GQCNNGraspPlannerImg, GQCNNGraspPlannerImgRequest, GQCNNGraspPlannerBoundingBox,
                       GQCNNGraspPlannerSegmask)
from task_msgs.msg import GQCNNGrasp, BoundingBox

DEPTH_IMG_TYPE = '32FC1'
COLOR_IMG_TYPE = 'bgr8'
MM_TO_M = True


class GraspPlanner(object):

    def __init__(self, cfg, cv_bridge, grasping_policy, grasp_pose_publisher):
        """
        Parameters
        ----------
        cfg : dict
            Dictionary of configuration parameters.
        cv_bridge: :obj:`CvBridge`
            ROS `CvBridge`.
        grasping_policy: :obj:`GraspingPolicy`
            Grasping policy to use.
        grasp_pose_publisher: :obj:`Publisher`
            ROS publisher to publish pose of planned grasp for visualization.
        """
        self.cfg = cfg
        self.cv_bridge = cv_bridge
        self.grasping_policy = grasping_policy
        self.grasp_pose_publisher = grasp_pose_publisher
        self._subscribe_img_topic = True
        self._bounding_box = None

        # Set minimum input dimensions.
        policy_type = "cem"
        if "type" in self.cfg["policy"]:
            policy_type = self.cfg["policy"]["type"]

        fully_conv_policy_types = {"fully_conv_suction", "fully_conv_pj"}
        if policy_type in fully_conv_policy_types:
            self.min_width = self.cfg["policy"]["gqcnn_recep_w"]
            self.min_height = self.cfg["policy"]["gqcnn_recep_h"]
        else:
            pad = max(
                math.ceil(
                    np.sqrt(2) *
                    (float(self.cfg["policy"]["metric"]["crop_width"]) / 2)),
                math.ceil(
                    np.sqrt(2) *
                    (float(self.cfg["policy"]["metric"]["crop_height"]) / 2)))
            self.min_width = 2 * pad + self.cfg["policy"]["metric"][
                "crop_width"]
            self.min_height = 2 * pad + self.cfg["policy"]["metric"][
                "crop_height"]
        if self.cfg["policy"]["metric"]['type'] == 'fcgqcnn':
            self.roi_height = self.cfg["policy"]["metric"]["fully_conv_gqcnn_config"]["im_height"]
            self.roi_width = self.cfg["policy"]["metric"]["fully_conv_gqcnn_config"]["im_width"]
        else:
            self.roi_height = 480
            self.roi_width = 640


    def read_images(self, req, bb_in=None):
        """Reads images from a ROS service request.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS ServiceRequest for grasp planner service.
        """
        create_seg = self._depth_bg is not None and bb_in is not None
        segmask_im = None
        # Get the raw depth and color images as ROS `Image` objects.
        raw_color = req.color_image
        raw_depth = req.depth_image

        # Get the raw camera info as ROS `CameraInfo`.
        raw_camera_info = req.camera_info

        color_im = self.cv_bridge.imgmsg_to_cv2(raw_color, COLOR_IMG_TYPE)
        depth_im = self.cv_bridge.imgmsg_to_cv2(raw_depth, DEPTH_IMG_TYPE)
        color_im, bb_out, raw_camera_info = self.crop_img(color_im, bb_in, raw_camera_info)
        depth_im, _, _ = self.crop_img(depth_im,  bb_in)
        if create_seg:
            segmask_im = self.create_segmask(depth_im, bb_out)
        if MM_TO_M:
            depth_im *= 0.001
        # Wrap the camera info in a BerkeleyAutomation/autolab_core
        # `CameraIntrinsics` object.
        camera_intr = CameraIntrinsics(
            raw_camera_info.header.frame_id, raw_camera_info.K[0],
            raw_camera_info.K[4], raw_camera_info.K[2], raw_camera_info.K[5],
            raw_camera_info.K[1], raw_camera_info.height,
            raw_camera_info.width)

        # Create wrapped BerkeleyAutomation/autolab_core RGB and depth images
        # by unpacking the ROS images using ROS `CvBridge`
        try:
            color_im = ColorImage(color_im, frame=camera_intr.frame)
            depth_im = DepthImage(depth_im, frame=camera_intr.frame)
            if create_seg:
                segmask_im = BinaryImage(segmask_im, frame=color_im.frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)

        # Check image sizes.
        if color_im.height != depth_im.height or \
           color_im.width != depth_im.width:
            msg = ("Color image and depth image must be the same shape! Color"
                   " is %d x %d but depth is %d x %d") % (
                       color_im.height, color_im.width, depth_im.height,
                       depth_im.width)
            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        if (color_im.height < self.min_height
                or color_im.width < self.min_width):
            msg = ("Color image is too small! Must be at least %d x %d"
                   " resolution but the requested image is only %d x %d") % (
                       self.min_height, self.min_width, color_im.height,
                       color_im.width)
            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        return color_im, depth_im, camera_intr, bb_out, segmask_im

    def plan_grasp(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        self._subscribe_img_topic = False
        img_req = GQCNNGraspPlannerImgRequest()
        img_req.camera_info = self._camera_info
        img_req.color_image = self._color_img
        img_req.depth_image = self._depth_img

        color_im, depth_im, camera_intr, bb, segmask_im = self.read_images(img_req, self._bounding_box)
        print('len(segmask_im.nonzero_pixels()) = {}'.format(len(segmask_im.nonzero_pixels())))
        if len(segmask_im.nonzero_pixels()) < 100:
            self._subscribe_img_topic = True
            return None
        result = self._plan_grasp(color_im, depth_im, camera_intr,segmask=segmask_im)

        o = result.pose.orientation
        q = np.quaternion(o.w, o.x, o.y, o.z)
        rot = qtn.as_rotation_matrix(q)
        euler = qtn.as_euler_angles(q)
        print('[Grasp planner]: rot = {}'.format(rot))
        print('[Grasp planner]: euler = {}'.format(euler))
        self._subscribe_img_topic = True
        return result

    def plan_grasp_img(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        color_im, depth_im, camera_intr, _, _ = self.read_images(req)
        return self._plan_grasp(color_im, depth_im, camera_intr)

    def plan_grasp_bb(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            `ROS ServiceRequest` for grasp planner service.
        """
        color_im, depth_im, camera_intr, bb, _ = self.read_images(req, req.bounding_box)
        return self._plan_grasp(color_im,
                                depth_im,
                                camera_intr,
                                bounding_box=bb)

    def plan_grasp_segmask(self, req):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        color_im, depth_im, camera_intr, _, _ = self.read_images(req)
        raw_segmask = req.segmask
        try:
            segmask = BinaryImage(self.cv_bridge.imgmsg_to_cv2(
                raw_segmask, desired_encoding="passthrough"),
                                  frame=camera_intr.frame)
        except CvBridgeError as cv_bridge_exception:
            rospy.logerr(cv_bridge_exception)
        if color_im.height != segmask.height or \
           color_im.width != segmask.width:
            msg = ("Images and segmask must be the same shape! Color image is"
                   " %d x %d but segmask is %d x %d") % (
                       color_im.height, color_im.width, segmask.height,
                       segmask.width)
            rospy.logerr(msg)
            raise rospy.ServiceException(msg)

        return self._plan_grasp(color_im,
                                depth_im,
                                camera_intr,
                                segmask=segmask)

    def _plan_grasp(self,
                    color_im,
                    depth_im,
                    camera_intr,
                    bounding_box=None,
                    segmask=None):
        """Grasp planner request handler.

        Parameters
        ---------
        req: :obj:`ROS ServiceRequest`
            ROS `ServiceRequest` for grasp planner service.
        """
        rospy.loginfo("Planning Grasp")

        # Inpaint images.
        color_im = color_im.inpaint(
            rescale_factor=self.cfg["inpaint_rescale_factor"])
        depth_im = depth_im.inpaint(
            rescale_factor=self.cfg["inpaint_rescale_factor"])

        # Init segmask.
        if segmask is None:
            segmask = BinaryImage(255 *
                                  np.ones(depth_im.shape).astype(np.uint8),
                                  frame=color_im.frame)

        # Visualize.
        if self.cfg["vis"]["color_image"]:
            vis.imshow(color_im)
            vis.show()
        if self.cfg["vis"]["depth_image"]:
            vis.imshow(depth_im)
            vis.show()
        if self.cfg["vis"]["segmask"] and segmask is not None:
            vis.imshow(segmask)
            vis.show()

        # Aggregate color and depth images into a single
        # BerkeleyAutomation/autolab_core `RgbdImage`.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)

        # Mask bounding box.
        if bounding_box is not None:
            # Calc bb parameters.
            min_x = int(bounding_box.minX)
            min_y = int(bounding_box.minY)
            max_x = int(bounding_box.maxX)
            max_y = int(bounding_box.maxY)

            # Contain box to image->don't let it exceed image height/width
            # bounds.
            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > rgbd_im.width:
                max_x = rgbd_im.width
            if max_y > rgbd_im.height:
                max_y = rgbd_im.height

            # Mask.
            bb_segmask_arr = np.zeros([rgbd_im.height, rgbd_im.width])
            bb_segmask_arr[min_y:max_y, min_x:max_x] = 255
            bb_segmask = BinaryImage(bb_segmask_arr.astype(np.uint8),
                                     segmask.frame)
            segmask = segmask.mask_binary(bb_segmask)

        # Visualize.
        if self.cfg["vis"]["rgbd_state"]:
            masked_rgbd_im = rgbd_im.mask_binary(segmask)
            vis.figure()
            vis.subplot(1, 2, 1)
            vis.imshow(masked_rgbd_im.color)
            vis.subplot(1, 2, 2)
            vis.imshow(masked_rgbd_im.depth)
            vis.show()

        # Create an `RgbdImageState` with the cropped `RgbdImage` and
        # `CameraIntrinsics`.
        rgbd_state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

        # Execute policy.
        try:
            result = self.execute_policy(rgbd_state, self.grasping_policy,
                                       self.grasp_pose_publisher,
                                       camera_intr.frame)
            if self._crop_min is not None:
                result.center_px = np.array(result.center_px) + self._crop_min
            return result
        except NoValidGraspsException:
            rospy.logerr(
                ("While executing policy found no valid grasps from sampled"
                 " antipodal point pairs. Aborting Policy!"))
            raise rospy.ServiceException(
                ("While executing policy found no valid grasps from sampled"
                 " antipodal point pairs. Aborting Policy!"))
        

    def execute_policy(self, rgbd_image_state, grasping_policy,
                       grasp_pose_publisher, pose_frame):
        """Executes a grasping policy on an `RgbdImageState`.

        Parameters
        ----------
        rgbd_image_state: :obj:`RgbdImageState`
            `RgbdImageState` to encapsulate
            depth and color image along with camera intrinsics.
        grasping_policy: :obj:`GraspingPolicy`
            Grasping policy to use.
        grasp_pose_publisher: :obj:`Publisher`
            ROS publisher to publish pose of planned grasp for visualization.
        pose_frame: :obj:`str`
            Frame of reference to publish pose in.
        """
        # Execute the policy"s action.
        grasp_planning_start_time = time.time()
        grasp = grasping_policy(rgbd_image_state)
        print('type(grasp) = {}'.format(type(grasp)))
        # Create `GQCNNGrasp` return msg and populate it.
        gqcnn_grasp = GQCNNGrasp()
        gqcnn_grasp.q_value = grasp.q_value
        gqcnn_grasp.pose = grasp.grasp.pose().pose_msg
        if isinstance(grasp.grasp, Grasp2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.PARALLEL_JAW
        elif isinstance(grasp.grasp, SuctionPoint2D):
            gqcnn_grasp.grasp_type = GQCNNGrasp.SUCTION
        else:
            rospy.logerr("Grasp type not supported!")
            raise rospy.ServiceException("Grasp type not supported!")

        # Store grasp representation in image space.
        gqcnn_grasp.center_px[0] = grasp.grasp.center[0]
        gqcnn_grasp.center_px[1] = grasp.grasp.center[1]
        gqcnn_grasp.angle = grasp.grasp.angle
        gqcnn_grasp.depth = grasp.grasp.depth
        gqcnn_grasp.thumbnail = grasp.image.rosmsg

        # Create and publish the pose alone for easy visualization of grasp
        # pose in Rviz.
        pose_stamped = PoseStamped()
        pose_stamped.pose = grasp.grasp.pose().pose_msg
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = pose_frame
        pose_stamped.header = header
        grasp_pose_publisher.publish(pose_stamped)

        # Return `GQCNNGrasp` msg.
        rospy.loginfo("Total grasp planning time: " +
                      str(time.time() - grasp_planning_start_time) + " secs.")

        return gqcnn_grasp

    def color_img_cb(self, data):
        if self._subscribe_img_topic:
            self._color_img = data

    def depth_img_cb(self, data):
        if self._subscribe_img_topic:
            self._depth_img = data

    def camera_inf_cb(self, data):
        if self._subscribe_img_topic:
            self._camera_info = data

    def set_bounding_box(self):
        self._color_img = None
        rate = rospy.Rate(10)
        mouse_event = self.MouseEvent()
        while (self._color_img is None or self._depth_img is None) and not rospy.is_shutdown():
            rate.sleep()

        cv2.namedWindow('Area Bounding', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Area Bounding', mouse_event.extract_coordinates)
        while not rospy.is_shutdown():
            bounding_img = self.cv_bridge.imgmsg_to_cv2(self._color_img, COLOR_IMG_TYPE)
            bbc = mouse_event.get_bbc()
            if len(bbc) == 2:
                cv2.rectangle(bounding_img, bbc[0], bbc[1], (36,255,12), 2)
            elif len(bbc) == 3:
                cv2.rectangle(bounding_img, bbc[0], bbc[2], (36,255,12), 2)
            cv2.imshow('Area Bounding', bounding_img)
            key = cv2.waitKey(1)

            if key == ord('s') and len(bbc) == 3:
                self._bounding_box = BoundingBox(bbc[0][0], bbc[0][1], bbc[2][0], bbc[2][1])
                print('top left: {}, bottom right: {}'.format(bbc[0], bbc[1]))
                cv2.destroyAllWindows()
                return

            # Close program with keyboard 'q'
            if key == ord('q'):
                cv2.destroyAllWindows()
                print('close without save')
                return
            rate.sleep()

    def create_depth_background(self):
        depth_sum = self.cv_bridge.imgmsg_to_cv2(self._depth_img, DEPTH_IMG_TYPE)
        rate = rospy.Rate(10)
        for _ in range(9):
            rate.sleep()
            depth_sum += self.cv_bridge.imgmsg_to_cv2(self._depth_img, DEPTH_IMG_TYPE)
        self._depth_bg = depth_sum / 10
        self._depth_bg, _, _ = self.crop_img(self._depth_bg, self._bounding_box)

    def create_segmask(self, depth_img, bb=None):
        begin = rospy.Time.now()
        if depth_img.shape != self._depth_bg.shape:
            print('depth_img.shap = {}, self._depth_bg.shape = {}'.format(depth_img.shap, self._depth_bg.shape))
            rospy.logerr('[create_segmask]: The shape of input img and depth background not match!')
            return
        segmask = ((depth_img - self._depth_bg) < -4).astype(np.uint8)
        kernel = np.ones((5,5), np.uint8)
        segmask = cv2.erode(segmask, kernel, iterations=2)
        segmask = cv2.dilate(segmask, kernel, iterations=2)
        if bb is not None:
            bb_segmask = np.zeros([depth_img.shape[0], depth_img.shape[1]])
            bb_segmask[int(bb.minY) : int(bb.maxY), int(bb.minX) : int(bb.maxX)] = 1
            segmask = np.logical_and(segmask, bb_segmask).astype(np.uint8)
        segmask = (255 * segmask).astype(np.uint8)
        print('[create_segmask]: create_segmask time = {}'.format((rospy.Time.now() - begin).to_sec()))
        return segmask

    def crop_img(self, img, bb_in=None, camera_intr=None):
        if bb_in is not None:
            bb = np.array([bb_in.minX, bb_in.minY, bb_in.maxX, bb_in.maxY])
            bb_resolution = [bb_in.maxX - bb_in.minX, bb_in.maxY - bb_in.minY]
        else:
            bb = np.array([0, 0, img.shape[1], img.shape[0]])
            bb_resolution = [img.shape[1], img.shape[0]]
        ratio = 1.0
        if bb_resolution[0] > self.roi_width or bb_resolution[1] > self.roi_height:
            if bb_resolution[0] / self.roi_width > bb_resolution[1] / self.roi_height:
                ratio = float(self.roi_width) / float(bb_resolution[0])
            else:
                ratio = float(self.roi_height) / float(bb_resolution[1])
            dim = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            bb = bb * ratio
            if camera_intr is not None:
                camera_intr.K = np.array(camera_intr.K) * ratio
                p = np.array(camera_intr.P).reshape(3, 4)
                p[:3, :3] *= ratio
                camera_intr.P = p.reshape(-1)
        center = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
        crop_min = [int(center[0] - self.roi_width / 2), int(center[1] - self.roi_height / 2)]
        crop_max = [crop_min[0] + self.roi_width, crop_min[1] + self.roi_height]
        img = img[crop_min[1] : crop_max[1], crop_min[0] : crop_max[0]]
        bb[0::2] -= crop_min[0]
        bb[1::2] -= crop_min[1]
        if camera_intr is not None:
            camera_intr.K = np.array(camera_intr.K)
            camera_intr.P = np.array(camera_intr.P)
            camera_intr.K[2] -= crop_min[0]
            camera_intr.K[5] -= crop_min[1]
            camera_intr.P[2] -= crop_min[0]
            camera_intr.P[5] -= crop_min[1]
        self._crop_min = np.array(crop_min)
        print('[crop_img]: ratio = {}, center = {}, crop_min = {}, img.shape = {}'.format(ratio, center, crop_min, img.shape))
        return img, BoundingBox(bb[0], bb[1], bb[2], bb[3]), camera_intr

    
    class MouseEvent(object):
        def __init__(self):
            self.bbc = []
            self.drawing = False

        def get_bbc(self):
            return self.bbc

        def extract_coordinates(self, event, x, y, flags, parameters):
            # Record starting (x,y) coordinates on left mouse button click
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.bbc = [(x,y)]

            # Record ending (x,y) coordintes on left mouse button release
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.bbc.append((x,y))

            # Clear drawing boxes on right mouse button click
            elif event == cv2.EVENT_MBUTTONUP:
                self.bbc = []
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if not self.drawing:
                    return
                if len(self.bbc) == 1:
                    self.bbc.append((x,y))
                elif len(self.bbc) == 2:
                    self.bbc[1] = (x,y)

if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node("Grasp_Sampler_Server")

    # Initialize `CvBridge`.
    cv_bridge = CvBridge()

    # Get configs.
    model_name = rospy.get_param("~model_name")
    model_dir = rospy.get_param("~model_dir")
    fully_conv = rospy.get_param("~fully_conv")
    color_img_topic = rospy.get_param("~color_img_topic")
    depth_img_topic = rospy.get_param("~depth_img_topic")
    camera_info_topic = rospy.get_param("~camera_info_topic")

    if model_dir.lower() == "default":
        model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 "../models")
    model_dir = os.path.join(model_dir, model_name)
    model_config = json.load(open(os.path.join(model_dir, "config.json"), "r"))
    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))

    # Set config.
    if (gripper_mode == GripperMode.LEGACY_PARALLEL_JAW
            or gripper_mode == GripperMode.PARALLEL_JAW):
        if fully_conv:
            config_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "cfg/examples/ros/fc_gqcnn_pj.yaml")
        else:
            config_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "cfg/examples/ros/gqcnn_pj.yaml")
    else:
        if fully_conv:
            config_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "cfg/examples/ros/fc_gqcnn_suction.yaml")
        else:
            config_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..",
                "cfg/examples/ros/gqcnn_suction.yaml")

    # Read config.
    cfg = YamlConfig(config_filename)
    policy_cfg = cfg["policy"]
    policy_cfg["metric"]["gqcnn_model"] = model_dir

    # Create publisher to publish pose of final grasp.
    grasp_pose_publisher = rospy.Publisher("/gqcnn_grasp/pose",
                                           PoseStamped,
                                           queue_size=10)

    # Create a grasping policy.
    rospy.loginfo("Creating Grasping Policy")
    if fully_conv:
        # TODO(vsatish): We should really be doing this in some factory policy.
        if policy_cfg["type"] == "fully_conv_suction":
            grasping_policy = \
                FullyConvolutionalGraspingPolicySuction(policy_cfg)
        elif policy_cfg["type"] == "fully_conv_pj":
            grasping_policy = \
                FullyConvolutionalGraspingPolicyParallelJaw(policy_cfg)
        else:
            raise ValueError(
                "Invalid fully-convolutional policy type: {}".format(
                    policy_cfg["type"]))
    else:
        policy_type = "cem"
        if "type" in policy_cfg:
            policy_type = policy_cfg["type"]
        if policy_type == "ranking":
            grasping_policy = RobustGraspingPolicy(policy_cfg)
        elif policy_type == "cem":
            grasping_policy = CrossEntropyRobustGraspingPolicy(policy_cfg)
        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))

    # Create a grasp planner.
    grasp_planner = GraspPlanner(cfg, cv_bridge, grasping_policy,
                                 grasp_pose_publisher)

    # Initialize the ROS service.
    grasp_planning_service = rospy.Service("grasp_planner", GQCNNGraspPlanner,
                                           grasp_planner.plan_grasp)
    grasp_planning_service_img = rospy.Service("grasp_planner_img", GQCNNGraspPlannerImg,
                                           grasp_planner.plan_grasp_img)
    grasp_planning_service_bb = rospy.Service("grasp_planner_bounding_box",
                                              GQCNNGraspPlannerBoundingBox,
                                              grasp_planner.plan_grasp_bb)
    grasp_planning_service_segmask = rospy.Service(
        "grasp_planner_segmask", GQCNNGraspPlannerSegmask,
        grasp_planner.plan_grasp_segmask)

    rospy.Subscriber(color_img_topic, Image, grasp_planner.color_img_cb)
    rospy.Subscriber(depth_img_topic, Image, grasp_planner.depth_img_cb)
    rospy.Subscriber(camera_info_topic, CameraInfo, grasp_planner.camera_inf_cb)

    rospy.loginfo("Grasping Policy Initialized")
    grasp_planner.set_bounding_box()
    grasp_planner.create_depth_background()
    rate = rospy.Rate(3)
    while not rospy.is_shutdown():
        rate.sleep()
        depth_img = grasp_planner.cv_bridge.imgmsg_to_cv2(grasp_planner._depth_img, DEPTH_IMG_TYPE)
        depth_img, bb, _ = grasp_planner.crop_img(depth_img, grasp_planner._bounding_box)
        print('bb = {}'.format(bb))
        segmask = grasp_planner.create_segmask(depth_img, bb)
        cv2.namedWindow('Segmask show', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Segmask show', segmask)
        key = cv2.waitKey(10)
        if key == ord('q'):
            cv2.destroyAllWindows()
            print('close segmask show')
            break

    # Spin forever.
    rospy.spin()
