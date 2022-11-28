// ROS-C++ framework
#include <ros/ros.h>

#include <iostream>
#include <bits/stdc++.h>

// OpenCV
#include <opencv2/opencv.hpp>

// OpenCV Aruco markers module
#include <opencv2/aruco.hpp>
// To work with different image transports
#include <image_transport/image_transport.h>
// To convert ROS Images into OpenCV images
#include <cv_bridge/cv_bridge.h>

// Useful ROS messages
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TransformStamped.h>

class PoseEstimationAruco
{
private:
    // Allows us to create topics, services and actions (implicit with Python)
    ros::NodeHandle node;

    // Create a dictionary by choosing one of the predefined dictionaries in
    // the aruco module (here a dictionary composed of 50 markers and a marker
    // size of 5x5 bits)
    const cv::Ptr<cv::aruco::Dictionary> DICTIONARY = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

    // Declare a vector of integers to store aruco markers' ids
    std::vector<int> ids;

    // Declare a vector of 2D points to store markers' corners
    std::vector<std::vector<cv::Point2f> > corners;

    // Declare the internal calibration matrix and distortion coefficients
    cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 476.703, 0., 400.5, 0., 476.703, 400.5, 0., 0., 1.);
    cv::Mat distCoeffs = (cv::Mat1d(1, 5) << 0, 0, 0, 0, 0);

    // Set the size of the marker side in meters (necessary to estimate
    // the camera pose)
    const double MARKER_SIZE = 1.;

    // Declare rotation and translation vectors (define the 3d transformation
    //from the marker coordinate system to the camera coordinate system)
    std::vector<cv::Vec3d> rvecs, tvecs;

    // Subscribers and publishers
    // image_transport::Subscriber sub_image;
    // ros::Subscriber sub_camera_info;
    ros::Subscriber sub_image;
    ros::Publisher pub_pose;

    // Declare callback function(s)
    // void callbackInfo(const sensor_msgs::CameraInfoConstPtr& msg);
    void callbackImage(const sensor_msgs::ImageConstPtr& msg);

public:
    PoseEstimationAruco();
};

PoseEstimationAruco::PoseEstimationAruco()
{
    // It is better practice to use image_transport instead of a simple subscriber
    // (allows to change image transport at run-time)
    // image_transport::ImageTransport imageTransport(node);

    // Initialize a subscriber to the camera info topic
    // sub_camera_info = node.subscribe("camera1/camera_info", 1, &PoseEstimationAruco::callbackInfo, this);
    
    // Initialize a subscriber to the camera topic
    // sub_image = imageTransport.subscribe("camera1/image_raw", 1, &PoseEstimationAruco::callbackImage, this);
    sub_image = node.subscribe("camera1/image_raw", 1, &PoseEstimationAruco::callbackImage, this);

    // Initialize a publisher to the pose_estimate topic
    pub_pose = node.advertise<geometry_msgs::TransformStamped>("pose_estimate", 1);
}

void PoseEstimationAruco::callbackImage(const sensor_msgs::ImageConstPtr& msg)
{   
    // To convert ROS Image type into a CvImage
    cv_bridge::CvImagePtr cvPtr;

    // Convert and copy the ROS Image into a CvImage
    cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

    std::clock_t t0, tf;
    t0 = std::clock();

    // Detect Aruco markers in the image (returns for each detected marker its id
    // and the position of its four corners)
    cv::aruco::detectMarkers(cvPtr->image, DICTIONARY, corners, ids);
    tf = std::clock();

    // If at least one marker detected
    if (ids.size() > 0)
        // Draw markers' boundaries, indicate the top-left corner and markers' id
        cv::aruco::drawDetectedMarkers(cvPtr->image, corners, ids);

        // Estimate the pose of the camera with solvePnP from 2D-3D correspondences
        cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

        std::cout << "Time taken : " << double(tf-t0)/double(CLOCKS_PER_SEC) << " s" << std::endl; 

        // Draw axes for each marker
        for(int i=0; i<ids.size(); i++)
        {
            // std::cout << rvecs[i] << std::endl;
            cv::drawFrameAxes(cvPtr->image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
        }

    cv::imshow("Preview", cvPtr->image);
    cv::waitKey(5);
}


// argc is the number of command line arguments and argv the list of arguments
int main(int argc, char **argv)
{
    // Initialize the node
    ros::init(argc, argv, "pose_estimation_aruco");

    PoseEstimationAruco poseEstimationAruco;
    
    ros::spin();

    return 0;
}
