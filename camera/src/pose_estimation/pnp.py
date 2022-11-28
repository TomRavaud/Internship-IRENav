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
from camera.msg import PoseEstimationAction, PoseEstimationResult,\
    PoseEstimationFeedback, Correspondences

# Python librairies
import numpy as np
import cv2

# My modules
from image_utils import switch_frame as sf, drawing as dw, conversions


class PoseEstimation:
    def __init__(self):
        """Constructor of the class
        """
        self.image = None
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Extract only one message from the camera_info topic as the camera
        # parameters do not change
        camera_info = rospy.wait_for_message("camera1/camera_info", CameraInfo)
        
        # Get the internal calibration matrix of the camera and reshape it to
        # more conventional form
        self.K = camera_info.K
        self.K = np.reshape(np.array(self.K), (3, 3))
        
        # Define variables for 2D-3D correspondences
        self.coordinates_2d = None
        self.coordinates_3d = None
        
        # Initialize the set of 4 points which will be used to draw the axes
        # These values are expressed in the platform coordinate system
        self.AXES_POINTS_PLATFORM = np.array([[0.3, 0, 0], [0, 0.3, 0],
                                              [0, 0, 0.3], [0, 0, 0]], dtype="float32")

        
        # Initialize the subscriber to the 2D-3D correspondences topic
        self.sub_correspondences = rospy.Subscriber("correspondences",
                                                    Correspondences,
                                                    self.callback_correspondences,
                                                    queue_size=1)
        
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
                
        while not self.pose_estimation_server.is_preempt_requested():
            
            # Get the current image for visualization purpose
            image = self.image
            # time = self.time
            
            
            # Solve the P-n-P problem (with RANSAC), ie find the transform
            # between the camera's frame and the platform's frame
            _, theta, T = cv2.solvePnP(self.coordinates_3d,
                                       self.coordinates_2d, self.K,
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
            # print(f"Time image : {time}")
            # print(f"Time points : {time_points}")
            # print("")
            image_to_display = np.copy(image)
            # Draw the platform axes on the image
            image_to_display = dw.draw_axes(image_to_display,
                                            axes_points_image)
            # Draw key-points on the image
            image_to_display = dw.draw_points(image_to_display, self.coordinates_2d)

            # Display the image
            dw.show_image(image_to_display, "Pose estimation")
            
            # Set a 30Hz frequency
            rospy.Rate(30).sleep()
        
        pose_estimation_result = PoseEstimationResult()
        pose_estimation_result.is_finished = True
        self.pose_estimation_server.set_preempted(pose_estimation_result)


    def callback_correspondences(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="bgr8")
        self.coordinates_2d = np.asarray(msg.coordinates_2d).reshape(-1, 2)
        self.coordinates_3d = np.asarray(msg.coordinates_3d).reshape(-1, 3)

 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("pnp")

    # Instantiate an object
    pose_estimation = PoseEstimation()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 