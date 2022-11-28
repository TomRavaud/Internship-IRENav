#!/usr/bin/env python3

import rospy
import cv2

import actionlib
import cv_bridge

from camera.msg import TemplateTrackingAction, TemplateTrackingGoal


# Initialize the node
rospy.init_node("template_tracking_action_client")

# Initialize the template tracking action client
template_tracking_client = actionlib.SimpleActionClient(
    "TemplateTracking", TemplateTrackingAction)

# Block until the action server is ready
template_tracking_client.wait_for_server()

# Set the action goal
template_tracking_goal = TemplateTrackingGoal()

# Use to convert OpenCV format to ROS Image
bridge = cv_bridge.CvBridge()
template_cv = cv2.imread(
    "./camera/src/correspondences_2d3d/Images/template.png")

template_tracking_goal.template = bridge.cv2_to_imgmsg(template_cv, encoding="passthrough")

# Send the goal to the server
template_tracking_client.send_goal(template_tracking_goal)
