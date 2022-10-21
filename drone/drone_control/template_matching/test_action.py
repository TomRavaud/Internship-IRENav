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
from drone_control.msg import PoseEstimationAction, PoseEstimationGoal, PoseEstimationResult, PoseEstimationFeedback

# Python librairies
import numpy as np
import cv2


# My modules
from image_processing import features_detection as fd, optical_flow as of, switch_frame as sf, drawing as dw, conversions, SIFT_detection_matching as sift


class TemplateTracking:
    def __init__(self):
        """Constructor of the class
        """
        self.image = None
        
        # Load the pre-computed linear predictor
        self.A = np.load("pre_computed_A_right.npy")
        # print(self.A)
        
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
        
        # Initialize the pose estimation action server
        self.template_tracking_server = actionlib.SimpleActionServer("drone/TemplateTracking",
                                                                   PoseEstimationAction,
                                                                   self.do_template_tracking,
                                                                   False)
        # Start the server
        self.template_tracking_server.start()
        

    def do_template_tracking(self, goal):
        
        # Set the current image as the reference image
        REFERENCE = self.image
        REFERENCE = cv2.cvtColor(REFERENCE, cv2.COLOR_BGR2GRAY)
        
        # The four 2D corner points of the template region in the image reference
        x1_i, x2_i = 239, 561
        y1_i, y2_i = 239, 561
        
        mu = np.array([[x1_i, y1_i],
                       [x2_i, y1_i],
                       [x2_i, y2_i],
                       [x1_i, y2_i]], dtype=np.float32)
        
        mu_previous = np.copy(mu)
        mu_previous_column = mu_previous.reshape(8, 1)
        
        # In the reference region
        x1_r, x2_r = 0, 322
        y1_r, y2_r = 0, 322
        
        MU = np.array([[x1_r, y1_r],
                       [x2_r, y1_r],
                       [x2_r, y2_r],
                       [x1_r, y2_r]])
        
        MU_homogeneous = np.ones((4, 3))
        MU_homogeneous[:, :-1] = MU
        
        # First homography
        F0, _ = cv2.findHomography(MU, mu_previous)
        F = np.copy(F0)
        
        # Sample the image to a reduced grid of points
        NB_POINTS_1D = 10
        
        X = np.linspace(x1_r, x2_r, NB_POINTS_1D)
        Y = np.linspace(y1_r, y2_r, NB_POINTS_1D)
        
        NB_POINTS_2D = 64 # (10-2)**2
        
        # Grid inside the template in the region reference
        X_grid_r, Y_grid_r = np.meshgrid(X, Y)
        X_grid_r = X_grid_r[1:-1, 1:-1]
        Y_grid_r = Y_grid_r[1:-1, 1:-1]
        
        # X_grid_flat = X_grid.ravel()
        # Y_grid_flat = Y_grid.ravel()
        
        # points = np.zeros((NB_POINTS_2D, 2))
        
        # for k in range(NB_POINTS_2D):
        #     points[k, 0] = X_grid_flat[k]
        #     points[k, 1] = Y_grid_flat[k]
        
        def grid_homography(X_grid, Y_grid, F):
            # Update grid points coordinates
            f11, f12, f13 = F[0]
            f21, f22, f23 = F[1]
            f31, f32, f33 = F[2]
        
            X_grid_new = np.int32((f11*X_grid + f12*Y_grid + f13) / (f31*X_grid + f32*Y_grid + f33))
            Y_grid_new = np.int32((f21*X_grid + f22*Y_grid + f23) / (f31*X_grid + f32*Y_grid + f33))
            
            return X_grid_new, Y_grid_new
            
        X_grid_i, Y_grid_i = grid_homography(X_grid_r, Y_grid_r, F0)
        
        # Get the intensity of the points on the grid and store them in a column array
        I_REF = np.float32(REFERENCE[X_grid_i, Y_grid_i].reshape(NB_POINTS_2D, 1))

        rospy.Rate(30).sleep()
        
        
        while not self.template_tracking_server.is_preempt_requested():
            print(mu_previous)
            
            # Get the current image
            image = self.image
            
            # Convert the current image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract intensities from the current image using the previously
            # computed grid points
            i_current = np.float32(image_gray[X_grid_i, Y_grid_i].reshape(NB_POINTS_2D, 1))
            
            # Image differences
            di = i_current - I_REF
            
            # Compute the small disturbance using the pre-computed
            # linear predictor
            dmu = np.dot(self.A, di)
            
            # Deduce the new corners position
            # mu_current_column = MU_REF_COLUMN + dmu
            mu_current_column = mu_previous_column + dmu
            
            # Reshape the corners position vector
            mu_current = mu_current_column.reshape(4, 2)
            
            # FIXME:
            # F1, _ = cv2.findHomography(MU, mu_current)
            F10, _ = cv2.findHomography(mu + dmu.reshape(4, 2), mu)
            
            # FIXME:
            F = np.dot(np.dot(F, np.linalg.inv(np.dot(F0, F10))), F0)
            print(F)
            
            # FIXME: update mu
            mu_previous_homogeneous = np.ones((4, 3))
            mu_previous_homogeneous[:, :-1] = mu_previous
            
            mu_current_homogeneous = np.dot(MU_homogeneous, np.transpose(F))
            
            # Scale
            mu_current_homogeneous /= mu_current_homogeneous[:, -1, None]
            
            mu_current = mu_current_homogeneous[:, :-1]
            
            mu_previous = np.copy(mu_current)
            mu_previous_column = mu_previous.reshape(8, 1)
            
            
            # Update grid points coordinates
            X_grid_i, Y_grid_i = grid_homography(X_grid_r, Y_grid_r, F)
            
            # Send action feedback
            # pose_estimation_feedback = PoseEstimationFeedback()
            # pose_estimation_feedback.transform = camera_platform_transform
            # self.pose_estimation_server.publish_feedback(pose_estimation_feedback)
            
            image_to_display = np.copy(image)
            
            dw.draw_points(image_to_display, mu_current)
            
            # dw.draw_points(image_to_display, np.array([[x1, y1]]), color=(0, 255, 0))
            # dw.draw_points(image_to_display, cornershow, color=(0, 255, 0))
            
            X_grid_flat = X_grid_i.ravel()
            Y_grid_flat = Y_grid_i.ravel()

            points = np.zeros((NB_POINTS_2D, 2))

            for k in range(NB_POINTS_2D):
                points[k, 0] = X_grid_flat[k]
                points[k, 1] = Y_grid_flat[k]
                
            dw.draw_points(image_to_display, points, color=(255, 0, 0))
            
            # Display the image
            dw.show_image(image_to_display, "Template tracking")
            
            # cv2.waitKey()
            
            # Set a 30Hz frequency
            rospy.Rate(30).sleep()
        
        pose_estimation_result = PoseEstimationResult()
        pose_estimation_result.is_finished = True
        self.template_tracking_server.set_preempted(pose_estimation_result)

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
    rospy.init_node("template_tracking")

    # Instantiate an object
    template_tracking = TemplateTracking()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 