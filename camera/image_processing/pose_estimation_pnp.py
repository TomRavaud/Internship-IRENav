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
import cv2


# My modules
import features_detection as fd
import optical_flow as of
import switch_frame as sf
import drawing as dw
import conversions


class PoseEstimationPnP:
    def __init__(self):
        """Constructor of the class
        """
        # Set a boolean attribute to identify the first frame
        self.is_first_image = True
        
        # Declare some attributes which will be used to compute optical flow
        self.old_image = None
        self.old_points = None
        
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
        
        # Initialize the first 2D-3D correspondences of 4 points belonging
        # to the platform, it allows us to compute the homography between
        # the platform plane and the first image, and thus to deduce all the
        # other 2D-3D correspondences at the initial time instant
        corners_image_2D = np.array([[561, 561],
                                     [239, 561],
                                     [239, 239],
                                     [561, 239]], dtype="double")
        CORNERS_PLATFORM_2D = np.array([[-0.5, -0.5],
                                        [-0.5, 0.5],
                                        [0.5, 0.5],
                                        [0.5, -0.5]])
        # The homography matrix is defined up to a scale factor
        self.H, _ = cv2.findHomography(CORNERS_PLATFORM_2D, corners_image_2D)
        
        # Indicate the component on the z axis of the platform's corners
        # (all the corners are on the same plane z = 0 in the platform
        # coordinate system)
        self.CORNERS_PLATFORM_3D = np.zeros((4, 3))
        self.CORNERS_PLATFORM_3D[:, :-1] = CORNERS_PLATFORM_2D
        
        # Make 2D platform corners homogeneous vectors
        self.HOMOGENEOUS_CORNERS_PLATFORM = np.ones((4, 3))
        self.HOMOGENEOUS_CORNERS_PLATFORM[:, :-1] = CORNERS_PLATFORM_2D
        
        # Initialize the subscriber to the pose_true topic
        self.sub_pose_true = rospy.Subscriber("pose_true", TransformStamped,
                                              self.callback_pose_true,
                                              queue_size=1)
        
        # Declare an attribute to store the true transform between the camera
        # and the platform's frames
        self.HTM_true = None
        

    def callback_image(self, msg):
        """Function called each time a new ros Image message is received on
        the camera1/image_raw topic
        Args:
            msg (sensor_msgs/Image): a ROS image sent by the camera
        """
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
            # print(points)
            
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

            # Compute the homography (with RANSAC) between the old and the
            # current image. The homography matrix is defined up to a scale
            # factor
            H_old_current, _ = cv2.findHomography(self.old_points, points,
                                                  cv2.RANSAC, 3.0)
            
            # Compute the homography between the platform plane and the current
            # image. The homography matrix is defined up to a a scale factor
            self.H = np.dot(H_old_current, self.H)
            
            # Get the 2D-3D correspondences of the platform's corners
            corners_image_computed = np.dot(self.HOMOGENEOUS_CORNERS_PLATFORM,
                                            np.transpose(self.H))
            # Set the scale so that the third coordinate of the platform's
            # corners on the image is equal to 1
            corners_image_computed /= corners_image_computed[:, -1, None]
            
            # Keep only the 2D platform's corners coordinates on the image
            corners_image_2D = corners_image_computed[:, :-1].astype("double")
            
            # Solve the P-n-P problem (without RANSAC), ie find the transform
            # between the camera's frame and the platform's frame
            _, theta, T = cv2.solvePnP(self.CORNERS_PLATFORM_3D,
                                            corners_image_2D, self.K,
                                            np.zeros((4, 1)), flags=0)
            
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
            image_to_display = dw.draw_axes(image_to_display,
                                            axes_points_image)
            image_to_display = dw.draw_axes(image_to_display,
                                            axes_points_image_true,
                                            colors = [(255, 0, 255),
                                                      (255, 0, 255),
                                                      (150, 0, 150)])

            # Draw key-points on the image
            image_to_display = dw.draw_points(image_to_display, points)
            # Draw platform's corners on the image
            image_to_display = dw.draw_points(image_to_display,
                                              corners_image_2D,
                                              (0, 255, 0))

            # Display the image
            dw.show_image(image_to_display)
            
        # Update old image and points
        self.old_image = image
        self.old_points = points
        
    def callback_pose_true(self, msg):
        """Function caller each time a new ROS TransformStamped is received
        on the pose_true topic

        Args:
            msg (geometry_msgs/TransformStamped): the true transform between
            the camera and the platform's frames
        """
        self.HTM_true = conversions.tf_to_transform_matrix(msg.transform)
        
 
# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("pose_estimation_pnp")

    # Instantiate an object
    pose_estimation_pnp = PoseEstimationPnP()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 