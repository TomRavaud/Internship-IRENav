#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# A library to deal with actions
import actionlib

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

# Custom messages
from drone_control.msg import PoseEstimationAction, PoseEstimationResult,\
    PoseEstimationFeedback

# Python librairies
import numpy as np
import cv2


# My modules
from image_utils import features_detection as fd, optical_flow as of,\
    switch_frame as sf, drawing as dw, conversions,\
        SIFT_detection_matching as sift


def linear_interpolation(x1, min1, max1, min2, max2):
    """Map a number from one range to another

    Args:
        x1 (float): A number in the first range
        min1 (float): Lower bound of the first range
        max1 (float): Upper bound of the first range
        min2 (float): Lower bound of the second range
        max2 (float): Upper bound of the second range

    Returns:
        float: Mapped number in the second range
    """
    A = (max2-min2) / (max1-min1)
    B = (min2*max1 - min1*max2) / (max1 - min1)
    x2 = A*x1 + B
    
    return x2

class PoseEstimation:
    def __init__(self):
        """Constructor of the class
        """
        self.image = None
        
        # Initialize the set of 4 points which will be used to draw the axes
        # These values are expressed in the platform coordinate system
        self.AXES_POINTS_PLATFORM = np.array([[0.3, 0, 0], [0, 0.3, 0],
                                              [0, 0, 0.3], [0, 0, 0]], dtype="float32")
        
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
        
        # Initialize the publisher to the pose_estimate topic
        self.pub_pose = rospy.Publisher("pose_estimate", TransformStamped,
                                        queue_size=1)
        
        # Initialize the pose estimation action server
        self.pose_estimation_server = actionlib.SimpleActionServer("PoseEstimation",
                                                                   PoseEstimationAction,
                                                                   self.do_pose_estimation,
                                                                   False)
        # Start the server
        self.pose_estimation_server.start()
        

    def do_pose_estimation(self, goal):
                
        # Read the platform template and set it to grayscale
        PLATFORM_TEMPLATE = cv2.imread(
                "/media/tom/Shared/Stage-EN-2022/quadcopter_landing_ws/src/mobile_platform/platform_gazebo/media/materials/textures/texture_wall_cropped_and_resized.jpg",
                flags=cv2.IMREAD_GRAYSCALE)
        
        # Get the current image
        old_image = self.image
        
        # Convert the current image to grayscale
        old_image_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
        
        # Detect and match SIFT points on the current image and the template
        TEMPLATE_POINTS, old_points = sift.detect_and_match(
            PLATFORM_TEMPLATE, old_image_gray, nb_points=20)
        
        # Get the 3D position of those points in the platform frame
        # Indicate the component on the z axis of the platform's points
        # (all the points are on the same plane z = 0 in the platform
        # coordinate system)
        TEMPLATE_POINTS_3D = np.zeros((np.shape(TEMPLATE_POINTS)[0], 3))
        
        template_width, template_height = np.shape(PLATFORM_TEMPLATE)
        assert template_width == template_height, "The image must be square !"
        
        # Convert points coordinates to real platform dimensions
        platform_side = 1
        TEMPLATE_POINTS_3D[:, :-1] = linear_interpolation(TEMPLATE_POINTS,
                                                          0, template_width,
                                                          -platform_side/2,
                                                          platform_side/2)
        
        # Swap columns and take their opposite to have the right coordinates in
        # the platform frame
        column1 = np.copy(TEMPLATE_POINTS_3D[:, 0])
        TEMPLATE_POINTS_3D[:, 0] = -TEMPLATE_POINTS_3D[:, 1]
        TEMPLATE_POINTS_3D[:, 1] = -column1
        
        while not self.pose_estimation_server.is_preempt_requested():
            
            # print(self.pose_estimation_server.is_preempt_requested())
            
            # Get the current image
            image = self.image
            
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, _, status = of.points_next_position_lk(image,
                                                   old_image,
                                                   old_points)
            
            if len(points) < len(TEMPLATE_POINTS_3D):
                status = np.array(status)
                # Keep only template points corresponding to points still
                # currently tracked
                TEMPLATE_POINTS_3D = TEMPLATE_POINTS_3D[status[:, 0] == 1]
                # print(status)
            
            # Solve the P-n-P problem (with RANSAC), ie find the transform
            # between the camera's frame and the platform's frame
            _, theta, T = cv2.solvePnP(TEMPLATE_POINTS_3D,
                                       points, self.K,
                                       np.zeros((4, 1)),
                                       flags=0)
            
            # Get the translation vector and the rotation matrix from the
            # output of the pnp solving function
            R = cv2.Rodrigues(theta)[0]
            T = T[:, 0]
            
            # Gather the rotation and the translation in a transform matrix
            HTM = sf.transform_matrix(R, T)
            # Convert the transform matrix to a ROS Transform object
            camera_platform_transform = conversions.transform_matrix_to_tf(HTM)
            
            # Publish the transform to the pose_estimate topic
            self.pub_pose.publish(camera_platform_transform)
            
            # Send action feedback
            pose_estimation_feedback = PoseEstimationFeedback()
            pose_estimation_feedback.transform = camera_platform_transform
            self.pose_estimation_server.publish_feedback(pose_estimation_feedback)
            
            # Update axes points coordinates
            axes_points_camera = sf.apply_rigid_motion(self.AXES_POINTS_PLATFORM,
                                                       HTM)
            
            # Compute those points coordinates in the image plan
            axes_points_image = sf.camera_frame_to_image(axes_points_camera,
                                                         self.K)
            
            #TODO: Display the pose estimation result on the image
            image_to_display = np.copy(image)
            # Draw the platform axes on the image
            image_to_display = dw.draw_axes(image_to_display,
                                            axes_points_image)
            # Draw key-points on the image
            image_to_display = dw.draw_points(image_to_display, points)
            # Draw template's key-points on the image
            image_to_display = dw.draw_points(image_to_display, TEMPLATE_POINTS,
                                              color=(0, 255, 0))

            # Display the image
            dw.show_image(image_to_display, "Pose estimation")
            
            old_points = points
            old_image = image
            
            # Set a 30Hz frequency
            rospy.Rate(10).sleep()
        
        pose_estimation_result = PoseEstimationResult()
        pose_estimation_result.is_finished = True
        self.pose_estimation_server.set_preempted(pose_estimation_result)

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
        # Get the time the message is published
        self.time = msg.header.stamp.to_sec()
        
        # Convert the ROS Image into the OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("pose_estimation")

    # Instantiate an object
    pose_estimation = PoseEstimation()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 