#!/usr/bin/env python3

# ROS - Python librairies
from json.encoder import INFINITY
import rospy
import actionlib
from drone_control.msg import IBVSAction, IBVSGoal, IBVSResult

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


class IBVSbis:
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
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Extract only one message from the camera_info topic as the camera
        # parameters do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        
        # Get the internal calibration matrix of the camera and reshape it to
        # more conventional form
        self.K = camera_info.K
        self.K = np.reshape(np.array(self.K), (3, 3))
         
        # Initialize a publisher to the drone velocity topic
        self.pub_velocity = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        
        self.server = actionlib.SimpleActionServer("drone/IBVS", IBVSAction, self.do_stabilize, False)
        self.server.start()
        
        self.error_norm = 1e4
        
    
    def do_stabilize(self, goal):
        
        # TODO: If an image has been given
        if goal.stabilize:
            
            # Throw away all the messages older than 1 second from now
            # It is not the best way to do it, it would be better changing the
            # system architecture
            current_time = rospy.get_time()  # Get the current simulation time
            image = rospy.wait_for_message("camera1/image_raw", Image)
            
            while image.header.stamp.secs + 1 < current_time:
                image = rospy.wait_for_message("camera1/image_raw", Image)
            
            # Initialize the subscriber to the camera images topic
            self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                          self.callback_image, queue_size=1)

            while self.error_norm > 20:
                continue

        result = IBVSResult()
        result.state = "done"
        self.server.set_succeeded(result)
        

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
        
        # cv2.imshow(str(self.i), image)
        # cv2.waitKey()
        # self.i += 1

        # Find the Harris' corners on the first frame
        if self.is_first_image:
            # cv2.imshow("First", image)
            # cv2.waitKey()
            
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
            twist.twist.linear.x = velocity[0]
            twist.twist.linear.y = -velocity[1]
            twist.twist.linear.z = -velocity[2]
            twist.twist.angular.z = -velocity[3]
            
            self.pub_velocity.publish(twist)           
            
            self.error_norm = np.linalg.norm(points - self.target_points)
            # print(f"The norm of the error is {self.error_norm} pixels"
            #       "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            
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
        
        # Display the image
        dw.show_image(image_to_display)
        
 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVSbis")

    # Instantiate an object
    IBVS = IBVSbis()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 