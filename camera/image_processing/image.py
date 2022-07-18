#!/usr/bin/env python3

import rospy
import cv2, cv_bridge
from sensor_msgs.msg import Image

class Test:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        cv2.namedWindow("window", 1)
        self.image_sub = rospy.Subscriber("camera1/image_raw",
                                          Image, self.image_callback)
        
    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("window", image)
        cv2.waitKey(3)
        
rospy.init_node("test")
test = Test()
rospy.spin()
