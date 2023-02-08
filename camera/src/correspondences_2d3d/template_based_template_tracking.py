#!/usr/bin/env python3

# ROS - Python librairies
import rospy

# A library to deal with actions
import actionlib

# cv_bridge is used to convert ROS Image message type into OpenCV images
import cv_bridge

# Import useful ROS types
from sensor_msgs.msg import Image

# Custom messages
from camera.msg import TemplateTrackingAction, TemplateTrackingResult,\
    TemplateTrackingFeedback, Correspondences

# Python librairies
import numpy as np
import cv2
import time

# My modules
from image_utils import drawing as dw, sift_detection_matching as sift


def grid_homography(X_grid, Y_grid, H):
    """Apply a homography to a grid

    Args:
        X_grid (ndarray (m, n)): grid X coordinates
        Y_grid (ndarray (n, m)): grid Y coordinates
        H (ndarray (3, 3)): homography matrix

    Returns:
        (ndarray (m, n), ndarray (n, m)): grid X and Y transformed coordinates
    """
    # Update grid points coordinates
    h11, h12, h13 = H[0]
    h21, h22, h23 = H[1]
    h31, h32, h33 = H[2]

    X_grid_new = np.int32(np.round((h11*X_grid + h12*Y_grid + h13) / (h31*X_grid + h32*Y_grid + h33)))
    Y_grid_new = np.int32(np.round((h21*X_grid + h22*Y_grid + h23) / (h31*X_grid + h32*Y_grid + h33)))
    
    return X_grid_new, Y_grid_new

def apply_homography(src, H):
    """Apply a homography to a set of points

    Args:
        src (ndarray (n, 2)): set of 2D points
        H (ndarray (3, 3)): homography matrix

    Returns:
        ndarray (n, 2): transformed set of points
    """
    return (cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H)).reshape(-1, 2)


class TBTT:
    def __init__(self):
        """Constructor of the class
        """
        # Declare the image currently published on the image topic
        self.image = None
        
        # Load the pre-computed linear predictor
        # Be careful, A must be computed for the right template
        self.A = np.load("./camera/src/correspondences_2d3d/template_learning/Actf.npy")
       
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Initialize the subscriber to the camera images topic
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                          self.callback_image, queue_size=1)
        
        # Initialize a publisher to the correspondences topic
        self.pub_correspondences = rospy.Publisher(
            "correspondences", Correspondences, queue_size=1)

        # Initialize the pose estimation action server
        self.template_tracking_server = actionlib.SimpleActionServer(
            "TemplateTracking",
            TemplateTrackingAction,
            self.do_template_tracking,
            False)
        # Start the server
        self.template_tracking_server.start()
        

    def do_template_tracking(self, goal):
        # Get the template image
        TEMPLATE = self.bridge.imgmsg_to_cv2(goal.template)
        TEMPLATE = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)
        
        #FIXME: Corners order has been changed
        # 3D coordinates of the template corners in the platform coordinate frame
        TEMPLATE_CORNERS_3D = np.array([[ 0.310559, 0.310559, 0.],
                                        [-0.1552795,  0.310559, 0.],
                                        [-0.1552795, -0.1552795, 0.],
                                        [ 0.310559,  -0.1552795, 0.]]).flatten()
        
        # Get the current image
        image = self.image
        
        # Convert the current image to grayscale
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect and match SIFT points in the current and target images
        TEMPLATE_POINTS, points = sift.detect_and_match(
            TEMPLATE, image_gray, nb_points=200)
        
        # Find the homography between the template image and
        # the current image, with RANSAC to deal with outliers
        Fstar, _ = cv2.findHomography(TEMPLATE_POINTS, points, cv2.RANSAC, 3.0)
        
        # Get the position of the template corners in the current image
        TEMPLATE_WIDTH, TEMPLATE_HEIGH = np.shape(TEMPLATE)
        TEMPLATE_CORNERS_2D = np.array([[0, 0],
                                        [TEMPLATE_WIDTH, 0],
                                        [TEMPLATE_WIDTH, TEMPLATE_HEIGH],
                                        [0, TEMPLATE_HEIGH]])
        mu0 = apply_homography(TEMPLATE_CORNERS_2D, Fstar)
        
        # Sample the template into a reduced grid of points
        NB_POINTS_1D = 20
        
        X = np.linspace(0, TEMPLATE_WIDTH-1, NB_POINTS_1D, dtype=np.int32)
        Y = np.linspace(0, TEMPLATE_HEIGH-1, NB_POINTS_1D, dtype=np.int32)
        
        NB_POINTS_2D = NB_POINTS_1D**2
        
        # Grid inside the template image
        X_grid_star, Y_grid_star = np.meshgrid(X, Y)
        
        # Get the intensity of the points on the grid and store them in a column array
        I_REF = np.float32(TEMPLATE[X_grid_star, Y_grid_star].reshape(NB_POINTS_2D, 1))
        
        # Initialize mu
        mu = np.copy(mu0)
        
        # Initialize the grid in the reference image
        X_grid0, Y_grid0 = grid_homography(X_grid_star, Y_grid_star, Fstar)
        X_grid = np.copy(X_grid0)
        Y_grid = np.copy(Y_grid0)
        
        # Initialize the homography between the reference image and the current image
        F = np.eye(3)

        # Coarse-to-fine approach parameters
        NB_LEVEL = 5
        NB_IT = 3
        
        
        while not self.template_tracking_server.is_preempt_requested():
            
            # Get the current image
            image = self.image
            
            # Publish the computed correspondences
            correspondences = Correspondences()
            mu_copy = np.copy(mu)
            # mu_copy[1, :], mu_copy[2, :] = mu_copy[2, :], mu_copy[1, :]
            correspondences.coordinates_2d = mu_copy[:, ::-1].flatten()
            # print(mu_copy[:, ::-1])
            correspondences.coordinates_3d = TEMPLATE_CORNERS_3D
            self.pub_correspondences.publish(correspondences)
            
            # Convert the current image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # start = time.time()
            
            for l in range(NB_LEVEL):
                # Get the pre-computed matrix associated with the current level
                Al = self.A[:, l*NB_POINTS_2D:(l+1)*NB_POINTS_2D]
                
                for it in range(NB_IT):
                    # Extract intensities from the current image using the previously
                    # computed grid points
                    i_current = np.float32(image_gray[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
                    
                    # Image differences
                    di = i_current - I_REF
                    
                    # Compute the small disturbance using the pre-computed
                    # linear predictor
                    dmu0 = np.dot(Al, di)
                    dmu0 = dmu0.reshape(4, 2).astype(np.float32)
                    
                    # Compute the current template corners positions
                    mu = apply_homography(mu0 - dmu0, F)

                    # Compute the homography between the position of the template in
                    # the first frame and in the current frame
                    F, _ = cv2.findHomography(mu0, mu)

                    # Update the grid
                    X_grid, Y_grid = grid_homography(X_grid0, Y_grid0, F)
            
            # stop = time.time()
            
            # print(stop-start)
            
            # Display the image
            image_to_display = np.copy(image)
            dw.draw_quadrilateral(image_to_display, mu[:,::-1])
            dw.show_image(image_to_display, "Template tracking")
            
            
            # Set a 30Hz frequency
            rospy.Rate(50).sleep()
            
            
            # Send action feedback
            template_tracking_feedback = TemplateTrackingFeedback()
            template_tracking_feedback.nb_correspondences = 4
            self.template_tracking_server.publish_feedback(template_tracking_feedback)
        
        
        # Send action result
        template_tracking_result = TemplateTrackingResult()
        template_tracking_result.is_complete = True
        self.template_tracking_server.set_preempted(template_tracking_result)

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
    rospy.init_node("template_based_template_tracking")

    # Instantiate an object
    tbtt = TBTT()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 