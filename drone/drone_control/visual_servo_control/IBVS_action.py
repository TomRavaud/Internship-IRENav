#!/usr/bin/env python3

# ROS - Python librairies
import rospy
import actionlib

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import TransformStamped, TwistStamped

# Custom messages
from drone_control.msg import IBVSAction, IBVSGoal, IBVSResult, IBVSFeedback,\
    DecisionAction, DecisionGoal, DecisionResult, DecisionFeedback

# Python librairies
import numpy as np
import cv2

# My modules
from image_processing import optical_flow as of, drawing as dw, conversions, SIFT_detection_matching as sift
import IBVS_velocity_command as vc


class DroneControl:
    def __init__(self):
        """Constructor of the class
        """
        # Declare the image and time variables currently published on the
        # image topic
        self.image = None
        self.time = None
        
        self.platform_velocity = np.zeros(4)
        self.measured_velocity = np.zeros((4,))
        self.drone_platform_angle = None
        
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
        
        # Initialize the subscriber to the estimated velocity (ArduPilot EKF)
        # topic
        self.sub_velocity = rospy.Subscriber(
            "mavros/local_position/velocity_body", TwistStamped,
            self.callback_velocity, queue_size=1)
        
        # Initialize the subscriber to the platform joint states topic
        self.sub_platform_joints = rospy.Subscriber(
            "mobile_platform/joint_states", JointState,
            self.callback_platform_joints, queue_size=1)
        
        # Initialize the subscriber to the pose_estimate topic
        # self.sub_pose = rospy.Subscriber("pose_estimate", TransformStamped,
        #                                  self.callback_pose, queue_size=1)

        # Initialize a publisher to the drone velocity topic
        self.pub_velocity = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        
        # Initialize the publisher to the IBVS_error topic
        self.pub_IBVS_error = rospy.Publisher(
            "IBVS_error", Float32, queue_size=1)
        
        # Initialize the IBVS server
        self.IBVS_server = actionlib.SimpleActionServer("drone/IBVS",
                                                        IBVSAction,
                                                        self.do_IBVS,
                                                        False)
        # Start the server
        self.IBVS_server.start()
        

    def do_IBVS(self, goal):
        rospy.Rate(0.3).sleep()
        
        TARGET_IMAGE = self.bridge.imgmsg_to_cv2(goal.target_image)
        # cv2.imshow("Target", target_image)
        # cv2.waitKey()
        
        old_image = self.image
        # old_time = rospy.get_time()
        # old_time = self.time
        # dt = 1/50
        
        # Convert the current image to grayscale
        old_image_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
        
        # Detect and match SIFT points on the current and target images
        TARGET_POINTS, old_points = sift.detect_and_match(
            TARGET_IMAGE, old_image_gray, nb_points=20)
        
        # image_to_display = np.copy(old_image)
        
        # # Draw key-points on the image
        # image_to_display = dw.draw_points(image_to_display, old_points)
        # # Draw key-points' target position on the image
        # image_to_display = dw.draw_points(image_to_display, TARGET_POINTS,
        #                                   color=(0, 255, 0))
        # cv2.imshow("Test", image_to_display)
        # cv2.waitKey()
        
        # Compute the (constant) interaction matrix
        L = vc.compute_interaction_matrix(TARGET_POINTS, self.K)
        
        while not self.IBVS_server.is_preempt_requested():
            
            # Get the current image, time and measured velocity
            image = self.image
            # time = rospy.get_time()
            # time = self.time
            # measured_velocity = self.measured_velocity
            platform_velocity = self.platform_velocity
            
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, old_points, status = of.points_next_position_lk(image,
                                                            old_image,
                                                            old_points)

            # dt = time - old_time
            
            if len(points) < len(TARGET_POINTS):
                status = np.array(status)
                # Keep only target points corresponding to points still
                # currently tracked
                TARGET_POINTS = TARGET_POINTS[status[:, 0] == 1]
            
            velocity = vc.velocity_command(L, points, old_points,
                                           TARGET_POINTS,
                                           platform_velocity)
            
            # velocity_norm = np.linalg.norm(velocity)

            # Set the drone velocity as a Twist message
            twist = TwistStamped()
            twist.twist.linear.x = velocity[0]
            twist.twist.linear.y = -velocity[1]
            twist.twist.linear.z = -velocity[2]
            twist.twist.angular.z = -velocity[3]

            # Publish the velocity on the mavros velocity command topic
            self.pub_velocity.publish(twist)

            # Compute the norm of the error we try to minimize
            # and send it to the IBVS_error topic
            error_norm = np.linalg.norm(points - TARGET_POINTS)
            self.pub_IBVS_error.publish(error_norm)
            
            # Send action feedback
            IBVS_feedback = IBVSFeedback()
            # IBVS_feedback.drone_linear_velocity = np.linalg.norm(velocity[:3])
            # IBVS_feedback.drone_yaw_velocity = np.abs(velocity[3])
            IBVS_feedback.error = error_norm
            self.IBVS_server.publish_feedback(IBVS_feedback)
            
            image_to_display = np.copy(image)
            # Draw key-points on the image
            image_to_display = dw.draw_points(image_to_display, points)
            # Draw key-points' target position on the image
            image_to_display = dw.draw_points(image_to_display, TARGET_POINTS,
                                              color=(0, 255, 0))

            # Display the image
            dw.show_image(image_to_display, "IBVS")
            
            old_points = points
            old_image = image
            # old_time = time
            
            # Set a 30Hz frequency
            rospy.Rate(50).sleep()
        
        IBVS_result = IBVSResult()
        IBVS_result.is_drone_stabilized = True
        self.IBVS_server.set_preempted(IBVS_result)
        
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
        self.time = msg.header.stamp.to_sec()
        
        # Convert the ROS Image into the OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
    def callback_velocity(self, msg):
        self.measured_velocity[0] = msg.twist.linear.x
        self.measured_velocity[1] = -msg.twist.linear.y
        self.measured_velocity[2] = -msg.twist.linear.z
        self.measured_velocity[3] = -msg.twist.angular.z
        
    def callback_platform_joints(self, msg):
        self.platform_velocity[0] = msg.velocity[2]
        self.platform_velocity[1] = msg.velocity[3]
        self.platform_velocity[2] = msg.velocity[4]
        self.platform_velocity[3] = msg.velocity[5]
        
    def callback_pose(self, msg):
        camera_platform_transform = conversions.tf_to_transform_matrix(msg)
        angles = cv2.Rodrigues(camera_platform_transform[:3, :3])[0]
        self.drone_platform_angle = np.linalg.norm(angles[:2])


# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("IBVSbisbis")

    # Instantiate an object
    drone_control = DroneControl()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 