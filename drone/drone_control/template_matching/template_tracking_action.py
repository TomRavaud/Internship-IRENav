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
        self.A = np.load("pre_computed_A.npy")
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
        # Load the first image and set it to grayscale
        # REFERENCE = cv2.imread(
        #     "drone/drone_control/template_matching/initial_frame.png",
        #     flags=cv2.IMREAD_GRAYSCALE)
        # REFERENCE = cv2.imread(
        #     "drone/drone_control/visual_servo_control/Images/target_image_1475_wall_cropped.png",
        #     flags=cv2.IMREAD_GRAYSCALE)
        
        REFERENCE = self.image
        REFERENCE = cv2.cvtColor(REFERENCE, cv2.COLOR_BGR2GRAY)
        
        # Get the current image
        # old_image = self.image
        
        # Convert the current image to grayscale
        # old_image_gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
        
        # The four 2D corner points of the template region in the reference frame
        x1, x2 = 239, 561
        y1, y2 = 239, 561
        
        # corner = np.array([[x1],
        #                    [y1],
        #                    [1]])

        MU_REF = np.array([[x2, y2],
                           [x1, y2],
                           [x1, y1],
                           [x2, y1]])
        
        mu_previous = MU_REF
        
        # Reshape the corners array
        # MU_REF_COLUMN = MU_REF.reshape(8, 1)
        
        # Set previous corners position
        mu_previous_column = mu_previous.reshape(8, 1)
        
        # Sample the image to a reduced grid of points
        NB_POINTS_1D = 10
        NB_POINTS_2D = NB_POINTS_1D**2

        X = np.linspace(x1, x2, NB_POINTS_1D, dtype=np.int32)
        Y = np.linspace(y1, y2, NB_POINTS_1D, dtype=np.int32)

        NB_POINTS_2D = 36

        # Grid inside the template
        X_grid, Y_grid = np.meshgrid(X, Y)
        X_grid = X_grid[2:-2, 2:-2]
        Y_grid = Y_grid[2:-2, 2:-2]
        
        # X_grid_flat = X_grid.ravel()
        # Y_grid_flat = Y_grid.ravel()
        
        # points = np.zeros((64, 2))

        # for k in range(64):
        #     points[k, 0] = X_grid_flat[k]
        #     points[k, 1] = Y_grid_flat[k]
        
        # ref_copy = np.copy(REFERENCE)
        # dw.draw_points(ref_copy, points)
        # dw.draw_points(ref_copy, TEMPLATE_CORNERS_REF)
        # cv2.imshow("Ref", ref_copy)
        # cv2.waitKey()
        
        # Get the intensity of the points on the grid and store them in a column array
        I_REF = np.int32(REFERENCE[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
        # print(I_REF)
        
        # Detect and match SIFT points on the current image and the template
        # TEMPLATE_POINTS, old_points = sift.detect_and_match(
        #     PLATFORM_TEMPLATE, old_image_gray, nb_points=20)
        
        # Get the 3D position of those points in the platform frame
        # Indicate the component on the z axis of the platform's points
        # (all the points are on the same plane z = 0 in the platform
        # coordinate system)
        # TEMPLATE_POINTS_3D = np.zeros((np.shape(TEMPLATE_POINTS)[0], 3))
        
        # template_width, template_height = np.shape(PLATFORM_TEMPLATE)
        # assert template_width == template_height, "The image must be square !"
        
        # Swap columns and take their opposite to have the right coordinates in
        # the platform frame
        # column1 = np.copy(TEMPLATE_POINTS_3D[:, 0])
        # TEMPLATE_POINTS_3D[:, 0] = -TEMPLATE_POINTS_3D[:, 1]
        # TEMPLATE_POINTS_3D[:, 1] = -column1
        
        # cv2.imshow("ref2", REFERENCE)
        # cv2.waitKey()
        
        rospy.Rate(10).sleep()
        
        while not self.template_tracking_server.is_preempt_requested():
            
            # Get the current image
            image = self.image
            
            # Convert the current image to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # print(image_gray)
            # print(REFERENCE)
            # print(np.shape(image_gray - REFERENCE))
            # cv2.imshow("Diff", image_gray - REFERENCE)
            # cv2.waitKey()
            
            # Extract intensities from the current image using the previously
            # computed grid points
            i_current = np.int32(image_gray[X_grid, Y_grid].reshape(NB_POINTS_2D, 1))
            
            # Image differences
            di = i_current - I_REF
            print(di)
            
            # Compute the small disturbance using the pre-computed
            # linear predictor
            dmu = np.round(np.dot(self.A, di))
            dmu = np.int32(dmu)
            # dmu = 10
            # print(type(dmu[1, 0]))
            print(dmu)
            
            # Deduce the new corners position
            # mu_current_column = MU_REF_COLUMN + dmu
            mu_current_column = mu_previous_column + dmu
            
            # Reshape the corners position vector
            mu_current = mu_current_column.reshape(4, 2)
            # print(mu_current)
            # print(f"previous : {mu_previous_column}")
            # print(f"current : {mu_current_column}")
            
            # Set the warping function to be a homography (= perspective transform)
            # F, _ = cv2.findHomography(mu_previous, mu_current)
            # print(mu_previous - mu_current)
            # F = cv2.getPerspectiveTransform(mu_previous, mu_current)
            
            # F /= F[2, 2]
            # print(F)
            
            # Update grid points coordinates
            # f11, f12, f13 = F[0]
            # f21, f22, f23 = F[1]
            # f31, f32, f33 = F[2]
            
            # X_temp = np.copy(X_grid)
            # X_grid = np.int32((f11*X_grid + f12*Y_grid + f13) / (f31*X_grid + f32*Y_grid + f33))
            # Y_grid = np.int32((f21*X_temp + f22*Y_grid + f23) / (f31*X_temp + f32*Y_grid + f33))
            Y_grid += dmu[1, 0]
            
            # xtemp = x1
            # x1 = np.int32((f11*x1 + f12*y1 + f13) / (f31*x1 + f32*y1 + f33))
            # y1 = np.int32((f21*xtemp + f22*y1 + f23) / (f31*xtemp + f32*y1 + f33))
            
            # corner = np.dot(F, corner)
            # corner /= corner[2]
            # # print(corner)
            # cornershow = corner[:2].reshape(1, 2)
            
            
            # Send action feedback
            # pose_estimation_feedback = PoseEstimationFeedback()
            # pose_estimation_feedback.transform = camera_platform_transform
            # self.pose_estimation_server.publish_feedback(pose_estimation_feedback)
            
            image_to_display = np.copy(image)
            # image_to_display = cv2.warpPerspective(image, F, (800, 800))
            
            dw.draw_points(image_to_display, mu_current)
            
            # dw.draw_points(image_to_display, np.array([[x1, y1]]), color=(0, 255, 0))
            # dw.draw_points(image_to_display, cornershow, color=(0, 255, 0))
            
            X_grid_flat = X_grid.ravel()
            Y_grid_flat = Y_grid.ravel()

            points = np.zeros((NB_POINTS_2D, 2))

            for k in range(NB_POINTS_2D):
                points[k, 0] = X_grid_flat[k]
                points[k, 1] = Y_grid_flat[k]
                
            # print(points[0])
            # print(mu_current_column[-1] - points[0, 1])
            
            dw.draw_points(image_to_display, points, color=(255, 0, 0))
            
            # Display the image
            dw.show_image(image_to_display, "Template tracking")
            
            # cv2.waitKey()
            
            # mu_previous = mu_current
            mu_previous_column = mu_current_column
            mu_previous = mu_previous_column.reshape(4, 2)
            
            # old_points = points
            # old_image = image
            
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
 