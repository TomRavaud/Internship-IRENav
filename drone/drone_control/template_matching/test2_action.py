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

def grid_homography(X_grid, Y_grid, H):
    # Update grid points coordinates
    h11, h12, h13 = H[0]
    h21, h22, h23 = H[1]
    h31, h32, h33 = H[2]

    X_grid_new = np.int32(np.round((h11*X_grid + h12*Y_grid + h13) / (h31*X_grid + h32*Y_grid + h33)))
    Y_grid_new = np.int32(np.round((h21*X_grid + h22*Y_grid + h23) / (h31*X_grid + h32*Y_grid + h33)))
    
    return X_grid_new, Y_grid_new

def apply_homography(src, H):
    return (cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), H)).reshape(-1, 2)


class TemplateTracking:
    def __init__(self):
        """Constructor of the class
        """
        self.image = None
        
        # Load the pre-computed linear predictor
        self.A = np.load("Aok.npy")
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
        x1, x2 = 300, 450
        y1, y2 = 300, 450
        # x1_i, x2_i = 239, 561
        # y1_i, y2_i = 239, 561
        
        mu0 = np.array([[x1, y1],
                       [x2, y1],
                       [x2, y2],
                       [x1, y2]], dtype=np.float32)
        
        # Sample the image into a reduced grid of points
        NB_POINTS_1D = 20
        
        X = np.linspace(x1, x2, NB_POINTS_1D, dtype=np.int32)
        Y = np.linspace(y1, y2, NB_POINTS_1D, dtype=np.int32)
        
        NB_POINTS_2D = NB_POINTS_1D**2
        
        # Grid inside the template in the region reference
        X_grid0, Y_grid0 = np.meshgrid(X, Y)
        
        # Get the intensity of the points on the grid and store them in a column array
        I_REF = np.float32(REFERENCE[X_grid0, Y_grid0].reshape(NB_POINTS_2D, 1))
        
        # Initialize mu
        mu = np.copy(mu0)
        
        # Initialize the grid
        X_grid = np.copy(X_grid0)
        Y_grid = np.copy(Y_grid0)
        
        # Initialize the homography
        F = np.eye(3)

        # NB_LEVEL = 5
        # NB_IT = 3
        
        # image_to_display = np.copy(REFERENCE)
            
        # X_grid_flat = X_grid.ravel()
        # Y_grid_flat = Y_grid.ravel()
        
        # points = np.zeros((NB_POINTS_2D, 2))
        
        # for k in range(NB_POINTS_2D):
        #     points[k, 0] = X_grid_flat[k]
        #     points[k, 1] = Y_grid_flat[k]
        
        # dw.draw_points(image_to_display, points, color=(255, 0, 0))
        # dw.draw_points(image_to_display, mu)
        
        # # Display the image
        # dw.show_image(image_to_display, "Template tracking")
        
        # cv2.waitKey()


        while not self.template_tracking_server.is_preempt_requested():
            
            # Get the current image
            image = self.image
            # Convert the current image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # for l in range(NB_LEVEL):
            #     Al = self.A[:, l*NB_POINTS_2D:(l+1)*NB_POINTS_2D]
                
                # for it in range(NB_IT):
                
            # Extract intensities from the current image using the previously
            # computed grid points
            i_current = np.float32(image_gray[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
            # Image differences
            di = i_current - I_REF
            # print("di : ", di)
            
            # Compute the small disturbance using the pre-computed
            # linear predictor
            # dmu = np.dot(self.A, np.zeros((NB_POINTS_2D, 1)))
            dmu0 = np.dot(self.A, di)
            # dmu = -np.int32(np.round(dmu))                       
            dmu0 = dmu0.reshape(4, 2).astype(np.float32)
            # print("dmu0' : ", dmu0)
            
            # Input : same F, dmu0
            #FIXME: 1
            # mu0bis = mu0 + dmu0
            # mu1 = apply_homography(mu0bis, F)
            mu = apply_homography(mu0 - dmu0, F)
            print(mu)
            
            #FIXME: 2
            # Change reference frame
            # dmu = apply_homography(dmu0, F)
            # print("dmu : ", dmu)
            # Update mu
            # mu = mu + dmu
            # The problem is that each of the two terms are normalized differently !
            # mu = apply_homography(mu0, F) + apply_homography(dmu0, F)
            
            # print("CASE 1 : ", mu1)
            # print("CASE 2 : ", mu)
            
            # Compute the homography between the position of the template in
            # the first frame and in the current frame
            F, _ = cv2.findHomography(mu0, mu)
            # print("F : ", F)
            # print("mu : ", mu)
            # print("Fmu0 : ", (cv2.perspectiveTransform(mu0.reshape(-1, 1, 2).astype(np.float32), F)).reshape(-1, 2))
            
            # Update the grid
            X_grid, Y_grid = grid_homography(X_grid0, Y_grid0, F)
            # print("X_grid : ", X_grid)
            
            # X_grid_bis, Y_grid_bis = grid_homography(X_grid, Y_grid, np.linalg.inv(F))
            # print("X_grid 0 computed : ", X_grid_bis)
            
            
            image_to_display = np.copy(image)
            
            # X_grid_flat = X_grid.ravel()
            # Y_grid_flat = Y_grid.ravel()
            # X_grid_flat_bis = X_grid_bis.ravel()
            # Y_grid_flat_bis = Y_grid_bis.ravel()
            
            # points = np.zeros((NB_POINTS_2D, 2))
            # points_bis = np.zeros((NB_POINTS_2D, 2))
            
            # for k in range(NB_POINTS_2D):
            #     points[k, 0] = Y_grid_flat[k]
            #     points[k, 1] = X_grid_flat[k]
                # points_bis[k, 0] = X_grid_flat_bis[k]
                # points_bis[k, 1] = Y_grid_flat_bis[k]
            
            # dw.draw_points(image_to_display, points, color=(255, 0, 0))
            # dw.draw_points(image_to_display, points_bis, color=(0, 255, 0))
            
            dw.draw_quadrilateral(image_to_display, mu[:,::-1])
            
            # Display the image
            dw.show_image(image_to_display, "Template tracking")
            
            # cv2.waitKey()
                    
            rospy.Rate(10).sleep()
        

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
 