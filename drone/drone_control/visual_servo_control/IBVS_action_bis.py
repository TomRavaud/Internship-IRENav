#!/usr/bin/env python3

# ROS - Python librairies
from json.encoder import INFINITY
import rospy
import actionlib
from drone_control.msg import IBVSAction, IBVSGoal, IBVSResult, IBVSFeedback

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, TwistStamped

# Python librairies
import numpy as np
import cv2

# My modules
from image_processing import optical_flow as of, drawing as dw
import SIFT_detection_matching as sift
import IBVS_velocity_command as vc


class IBVS:
    def __init__(self):
        """Constructor of the class
        """
        # Declare the image and time variables currently published on the
        # image topic
        self.image = None
        self.time = None
        
        # Declare some attributes which will be used to compute optical flow
        # self.old_image = None
        # self.old_points = None
        
        # self.old_time = None
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Extract only one message from the camera_info topic as the camera
        # parameters do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        
        # Get the internal calibration matrix of the camera and reshape it to
        # more conventional form
        self.K = camera_info.K
        self.K = np.reshape(np.array(self.K), (3, 3))
        
        # Initialize the subscriber to the camera images topic
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                      self.callback_image, queue_size=1)

        # Initialize a publisher to the drone velocity topic
        self.pub_velocity = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        
        self.IBVS_server = actionlib.SimpleActionServer("drone/IBVS", IBVSAction, self.do_IBVS, False)
        self.IBVS_server.start()
        
        # self.error_norm = 1e4
        # self.is_stabilized = False


    def do_IBVS(self, goal):
        target_image = self.bridge.imgmsg_to_cv2(goal.target_image)
        # cv2.imshow("Target", target_image)
        # cv2.waitKey()        
        
        old_image = self.image
        old_time = self.time
        
        # Convert the current image to grayscale
        old_image_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
        
        # Detect and match SIFT points on the current and target images
        target_points, old_points = sift.detect_and_match(
            target_image, old_image_gray, nb_points=40) 
        
        image_to_display = np.copy(old_image)
        
        # # Draw key-points on the image
        # image_to_display = dw.draw_points(image_to_display, old_points)
        # # Draw key-points' target position on the image
        # image_to_display = dw.draw_points(image_to_display, target_points,
        #                                   color=(0, 255, 0))
        # cv2.imshow("Test", image_to_display)
        # cv2.waitKey()
        
        # Compute the (constant) interaction matrix
        L = vc.compute_interaction_matrix(target_points, self.K)
        
        velocity_norm = 1
        
        #TODO: do not forget to change this
        while velocity_norm >= 0:
        # while velocity_norm >= 0.01:
            # Get the current image and time
            image = self.image
            time = self.time
            
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, old_points = of.points_next_position_lk(image,
                                                            old_image,
                                                            old_points)

            dt = time - old_time

            velocity = vc.velocity_command(L, points, old_points, target_points, dt)
            
            velocity_norm = np.linalg.norm(velocity)
            # print(f"Velocity norm : {velocity_norm}")

            twist = TwistStamped()
            twist.twist.linear.x = velocity[0]
            twist.twist.linear.y = -velocity[1]
            twist.twist.linear.z = -velocity[2]
            twist.twist.angular.z = -velocity[3]

            self.pub_velocity.publish(twist)

            # self.error_norm = np.linalg.norm(points - self.target_points)
            # print(f"The norm of the error is {self.error_norm} pixels"
            #       "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            
            old_points = points
            old_image = image
            old_time = time
            
            IBVS_feedback = IBVSFeedback()
            IBVS_feedback.drone_linear_velocity = np.linalg.norm(velocity[:3])
            IBVS_feedback.drone_yaw_velocity = np.abs(velocity[3])
            self.IBVS_server.publish_feedback(IBVS_feedback)
            
            image_to_display = np.copy(image)
            # Draw key-points on the image
            image_to_display = dw.draw_points(image_to_display, points)
            # Draw key-points' target position on the image
            image_to_display = dw.draw_points(image_to_display, target_points,
                                              color=(0, 255, 0))

            # Display the image
            dw.show_image(image_to_display)
            
            # Set a 10Hz frequency
            rospy.Rate(30).sleep()
        
        IBVS_result = IBVSResult()
        IBVS_result.is_stabilized = True
        self.IBVS_server.set_succeeded(IBVS_result)

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        # Set old image and time
        # self.old_image = self.image
        # self.old_time = self.time
        
        # Get the time the message is published
        self.time = msg.header.stamp
        
        # Convert the ROS Image into the OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVSbisbis")

    # Instantiate an object
    IBVS_action = IBVS()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 