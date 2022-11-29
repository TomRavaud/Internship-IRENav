#!/usr/bin/env python3

# ROS - Python librairies
import rospy

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
        self.TARGET_IMAGE = cv2.imread(
            "/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/Images/target_image_1475_wall.png",
            flags=cv2.IMREAD_GRAYSCALE)
        
        self.target_points = None
        
        # Set a boolean attribute to identify the first frame
        self.is_first_image = True
        
        # Declare some attributes which will be used to compute optical flow
        self.old_image = None
        self.old_points = None
        
        self.old_time = None
        
        # self.depth_image = None
        
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
        # Initialize the subscriber to the depth image topic
        # self.sub_depth = rospy.Subscriber("camera1/image_raw_depth", Image,
        #                                   self.callback_depth)
        
        # Initialize a publisher to the drone velocity topic
        self.pub_velocity = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
                

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        # Get the time the message was published
        time = msg.header.stamp
        
        # Convert the ROS Image into the OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Find the Harris' corners on the first frame
        if self.is_first_image:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            self.target_points, points = sift.detect_and_match(
                self.TARGET_IMAGE, image_gray, nb_points=40)
            
            self.L = vc.compute_interaction_matrix(self.target_points, self.K)
            
            self.is_first_image = False
            
        # Compute the optical flow from the second frame and estimate the
        # platform pose in the camera coordinate system
        else:
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, self.old_points = of.points_next_position_lk(image,
                                                              self.old_image,
                                                              self.old_points)
            
            dt = time - self.old_time
            
            velocity = vc.velocity_command(self.L, points, self.old_points, self.target_points, dt)
            
            twist = TwistStamped()
            
            # twist.twist.linear.x = 0
            # twist.twist.linear.y = 0
            # twist.twist.linear.z = 0
            # twist.twist.angular.z = 0.2
            twist.twist.linear.x = velocity[0]
            twist.twist.linear.y = -velocity[1]
            twist.twist.linear.z = -velocity[2]
            twist.twist.angular.z = -velocity[3]
            
            # print(twist)
            
            error_norm = np.linalg.norm(points - self.target_points)
            print(f"The norm of the error is {error_norm} pixels"
                  "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            
            self.pub_velocity.publish(twist)
            
        # Update old image and points
        self.old_image = image
        self.old_points = points
        
        # Update the time the last message was published
        self.old_time = time
         
        image_to_display = np.copy(image)
        # Draw key-points on the image
        image_to_display = dw.draw_points(image_to_display, points)
        # Draw key-points' target position on the image
        image_to_display = dw.draw_points(image_to_display, self.target_points,
                                          color=(0, 255, 0))
        
        # cv2.imwrite("image_drone.png", image_to_display)
        
        # imagebis = cv2.imread("/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/drone/drone_control/visual_servo_control/target_image_1475_wall.png")
        # imagebis = dw.draw_points(imagebis, self.target_points,
        #                           color=(0, 255, 0))
        # cv2.imwrite("imagebis.png", imagebis)
        
        # Display the image
        dw.show_image(image_to_display)
        
    def callback_depth(self, msg):
        """Function called each time a new ROS Image is received on
        the camera1/image_raw_depth topic
        Args:
            msg (sensor_msgs/Image): a ROS depth image sent by the camera
        """
        # Convert the ROS Image into the OpenCV format
        # They are encoded as 32-bit float (32UC1) and each pixel is a depth
        # along the camera Z axis in meters
        # self.dstamp = msg.header.stamp
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough")

 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVS")

    # Instantiate an object
    IBVS = IBVS()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 