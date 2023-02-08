#!/usr/bin/env python3

# ROS - Python librairies
import rospy
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
from image_utils import optical_flow as of, drawing as dw,\
    sift_detection_matching as sift,\
        harris_corners_detection as harris


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

def apply_homography(src, H):
    """Apply a homography to a set of points

    Args:
        src (ndarray (n, 2)): set of 2D points
        H (ndarray (3, 3)): homography matrix

    Returns:
        ndarray (n, 2): transformed set of points
    """
    return (cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H)).reshape(-1, 2)


class FBTT:
    def __init__(self):
        """Constructor of the class
        """
        # Declare the image currently published on the image topic
        self.image = None
        
        # Initialize the bridge between ROS and OpenCV images
        self.bridge = cv_bridge.CvBridge()
        
        # Initialize the subscriber to the camera images topic
        self.sub_image = rospy.Subscriber("camera1/image_raw", Image,
                                      self.callback_image, queue_size=1)

        # Initialize a publisher to the correspondences topic
        self.pub_correspondences = rospy.Publisher(
            "correspondences", Correspondences, queue_size=1)
        
        # Initialize the correspondences server
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
        
        # 3D coordinates of the template corners in the platform coordinate frame
        TEMPLATE_CORNERS_3D = np.array([[ 0.310559, 0.310559, 0.],
                                        [ 0.310559,  -0.1552795, 0.],
                                        [-0.1552795, -0.1552795, 0.],
                                        [-0.1552795,  0.310559, 0.]])
        
        # Get the current image
        old_image = self.image
        
        # Convert the current image to grayscale
        old_image_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
        
        # Detect and match SIFT points in the current and target images
        TEMPLATE_POINTS, old_points = sift.detect_and_match(
            TEMPLATE, old_image_gray, nb_points=200)
        
        # # Compute the homography between the real template region and its
        # # projection in the image
        # H, _ = cv2.findHomography(old_points, TEMPLATE_POINTS, cv2.RANSAC, 3.0)
        
        # # Compute the Harris' score map
        # harris_score = harris.compute_harris_score(old_image)
        
        # # Set a threshold to avoid having too many corners,
        # # its value depends on the image
        # threshold = 0.4
        
        # # Extract corners coordinates
        # old_points = harris.corners_detection(harris_score, threshold)
        
        # # Compute corresponding coordinates in the platform frame
        # TEMPLATE_POINTS = apply_homography(old_points, H)
        
        
        # Get the 3D position of those points in the platform frame
        # Indicate the component on the z axis of the platform's points
        # (all the points are on the same plane z = 0 in the platform
        # coordinate system)
        TEMPLATE_POINTS_3D = np.zeros((np.shape(TEMPLATE_POINTS)[0], 3))
        
        # Map points coordinates to real platform dimensions
        template_width, template_height = np.shape(TEMPLATE)  # (in pixels)
        assert template_width == template_height, "The template must be square !"
        
        TEMPLATE_POINTS_3D[:, 0] = linear_interpolation(TEMPLATE_POINTS[:, 1],
                                                          0, template_width,
                                                          TEMPLATE_CORNERS_3D[0, 0],
                                                          TEMPLATE_CORNERS_3D[2, 0])
        TEMPLATE_POINTS_3D[:, 1] = linear_interpolation(TEMPLATE_POINTS[:, 0],
                                                          0, template_width,
                                                          TEMPLATE_CORNERS_3D[0, 0],
                                                          TEMPLATE_CORNERS_3D[2, 0])
        
        # Swap columns and take their opposite to have the right coordinates in
        # the platform frame
        # column1 = np.copy(TEMPLATE_POINTS_3D[:, 0])
        # TEMPLATE_POINTS_3D[:, 0] = -TEMPLATE_POINTS_3D[:, 1]
        # TEMPLATE_POINTS_3D[:, 1] = -column1
        
        nb_correspondences = np.shape(old_points)[0]
        
        
        while not self.template_tracking_server.is_preempt_requested():
            
            # Get the current image
            image = self.image
            
            # start = time.time()
            
            # The displacement might have not been found for some points
            # we need to update old_points to compute the difference between
            # the current points and these
            points, _, status = of.points_next_position_lk(image,
                                                           old_image,
                                                           old_points)
            # If we lose some correspondences...
            if np.shape(points)[0] < nb_correspondences:
                
                nb_correspondences = np.shape(points)[0]
                
                # If we do not have enough points to compute the pose, abort
                if nb_correspondences < 4:
                    template_tracking_result = TemplateTrackingResult()
                    template_tracking_result.is_complete = False
                    self.template_tracking_server.set_aborted(template_tracking_result)
                    
                # Keep only template points corresponding to points still
                # currently tracked
                TEMPLATE_POINTS_3D = TEMPLATE_POINTS_3D[status[:, 0] == 1]
                
            # stop = time.time()
            
            # print(stop - start)
                
            
            # Publish the computed correspondences
            correspondences = Correspondences()
            correspondences.image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            correspondences.coordinates_2d = points.flatten()
            correspondences.coordinates_3d = TEMPLATE_POINTS_3D.flatten()
            self.pub_correspondences.publish(correspondences)
            
            
            # # Display the image
            # image_to_display = np.copy(image)
            # # Draw key-points on the image
            # image_to_display = dw.draw_points(image_to_display, points)
            # dw.show_image(image_to_display, "Features-based template tracking")
            
            
            # Update points' coordinates and the latest image
            old_points = points
            old_image = image
            
            # Set a 30Hz frequency
            rospy.Rate(30).sleep()
            
            
            # Send action feedback
            template_tracking_feedback = TemplateTrackingFeedback()
            template_tracking_feedback.nb_correspondences = nb_correspondences
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
        # Convert the ROS Image into the OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
       

# Main program
# The "__main__" flag acts as a shield to avoid these lines to be executed if
# this file is imported in another one
if __name__ == "__main__":
    # Declare the node
    rospy.init_node("features_based_template_tracking")

    # Instantiate an object
    fbtt = FBTT()

    # Run the node until Ctrl + C is pressed
    rospy.spin()
 