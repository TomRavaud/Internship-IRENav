#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

# Python librairies
import numpy as np


# My modules
import features_detection as fd
import optical_flow as of
# import velocity as vel
import infinitesimal_motion as im
import switch_frame as sf
import drawing as dw
import conversions

class PoseEstimation:
    def __init__(self):
        """Constructor of the class
        """
        # Set a boolean attribute to identify the first frame
        self.is_first_image = True
        
        # Declare some attributes which will be used to compute optical flow
        self.old_image = None
        self.old_points = None
        self.old_time = None
        
        # Initialize the set of 4 points which will be used to draw the axes
        # These values are expressed in the platform coordinate system
        self.AXES_POINTS_PLATFORM = np.array([[0.3, 0, 0], [0, 0.3, 0],
                                              [0, 0, 0.3], [0, 0, 0]], dtype="float32")
        
        # Declare a depth image attribute
        self.depth_image = None
        
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
        self.sub_depth = rospy.Subscriber("camera1/image_raw_depth", Image,
                                          self.callback_depth)
        
        # Initialize the publisher to the pose_estimate topic
        self.pub_pose = rospy.Publisher("pose_estimate", TransformStamped,
                                        queue_size=1)
        
        # Initialize the rotation and the translation
        # These values correspond to the initial transformation between the camera
        # frame and the platform frame
        self.T = np.array([0., 0., 1.475])
        self.R = np.array([[0, -1, 0],
                           [-1, 0, 0],
                           [0, 0, -1]])
        
        # Initialize the subscriber to the pose_true topic
        self.sub_pose_true = rospy.Subscriber("pose_true", TransformStamped,
                                              self.callback_pose_true,
                                              queue_size=1)
        
        # Declare an attribute to store the true transform between the camera
        # and the platform's frames
        self.HTM_true = None
        
        # initial_true_transform = rospy.wait_for_message("pose_true", TransformStamped)
        # HTM0 = conversions.tf_to_transform_matrix(initial_true_transform.transform)
        # self.T = HTM0[:3, 3]
        # self.R = HTM0[:3, :3]
        

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
            # Compute the Harris' score map
            harris_score = fd.compute_harris_score(image)
            
            # Set a threshold to avoid having too many corners,
            # its value depends on the image
            threshold = 0.4
            
            # Extract corners coordinates
            points = fd.corners_detection(harris_score, threshold)
            
            self.is_first_image = False
            
        # Compute the optical flow from the second frame and estimate the
        # platform pose in the camera coordinate system
        else:
            # The optical flow might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, self.old_points, _ = of.points_next_position_lk(image,
                                                              self.old_image,
                                                              self.old_points)

            # Get the duration between two published messages
            # dt = (time - self.old_time).to_sec()
            
            # Compute the displacement of the corners between the 2 images
            points_displacement = of.sparse_displacement(self.old_points,
                                                         points)
            
            # Compute the optical flow at the corners
            # optical_flow = of.sparse_optical_flow(self.old_points, points, dt)
            
            # Estimate the rotational relative velocities
            # rot_velocities = vel.velocity_estimation(optical_flow,
            #                                          self.old_points, self.f,
            #                                          self.mu0, self.nu0)
            
            # Estimate the infinitesimal motion of the platform between two
            # successive images
            dmotion = im.compute_infinitesimal_rigid_motion(
                points_displacement, self.old_points, self.K, self.depth_image)
            
            # Extract infinitesimal rotation and translation
            dT = dmotion[:3]
            dtheta = dmotion[-3:]

            # Compute the infinitesimal rotation matrix from the angles
            dR = im.infinitesimal_rotation_matrix(dtheta)
            
            # Update R and T values
            self.R = np.dot(dR, self.R)
            self.T = np.dot(dR, self.T) + dT
            
        # Update old image and points
        self.old_image = image
        self.old_points = points
         
        # Update the time the last message was published
        self.old_time = time

        # Gather the rotation and the translation in a transform matrix
        HTM = sf.transform_matrix(self.R, self.T)
        # Convert the transform matrix to a ROS Transform object
        camera_platform_transform = conversions.transform_matrix_to_tf(HTM)
        # Publish the transform to the pose_estimate topic
        self.pub_pose.publish(camera_platform_transform)
            
        # Update axes points coordinates
        axes_points_camera = sf.apply_rigid_motion(self.AXES_POINTS_PLATFORM,
                                                HTM)
        axes_points_camera_true = sf.apply_rigid_motion(self.AXES_POINTS_PLATFORM,
                                                        self.HTM_true)
        # Compute those points coordinates in the image plan
        axes_points_image = sf.camera_frame_to_image(axes_points_camera,
                                                    self.K)
        axes_points_image_true = sf.camera_frame_to_image(axes_points_camera_true,
                                                          self.K)
            
        # Draw the platform axes on the image
        image_to_display = np.copy(image)
        image_to_display = dw.draw_axes(image_to_display, axes_points_image)
        image_to_display = dw.draw_axes(image_to_display, axes_points_image_true,
                             colors = [(255, 0, 255), (255, 0, 255), (150, 0, 150)])
            
        # Draw key-points on the image
        image_to_display = dw.draw_points(image_to_display, points)
            
        # Display the image
        dw.show_image(image_to_display)
        
    def callback_pose_true(self, msg):
        """Function caller each time a new ROS TransformStamped is received
        on the pose_true topic

        Args:
            msg (geometry_msgs/TransformStamped): the true transform between
            the camera and the platform's frames
        """
        self.HTM_true = conversions.tf_to_transform_matrix(msg.transform)
        
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
    rospy.init_node("pose_estimation")

    # Instantiate an object
    pose_estimation = PoseEstimation()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 