# Modelling and simulation of the automatic landing of a UAV on a mobile platform

This repository contains the source code for the application I worked on during my six months internship at the French Naval Academy Research Institute, in 2022.

<!-- ## TL;DR

Quick start


Here I detail the procedure for installing and running the project quickly. I worked with ROS noetic and Gazebo 11 on Ubuntu 20.04, and coded the nodes mainly in Python3, but for a quick start you just need to have Docker installed, I take care of the rest.

1. Clone [this](https://) repository to get the files which will help us work with Docker. Or simply execute the following command : 
   
   ```console
   $ git clone #to fill
   ```

2. In the directory created by the previous command, create a new directory for the ROS workspace :
   
   ```console
   $ cd ROS_simulation
   $ mkdir quacopter_landing_ws
   ```

3. Clone the source code of the project inside this new empty directory :
   
   ```console
   $ cd quadcopter_landing_ws
   $ git clone #to fill
   ```

4. Then, it depends if you want to install ROS, Gazebo and the other necessary tools directly on your machine, or if you prefer to use my Docker environment instead.
   
   * Manual installation : Be sure to have the correct versions of the software installed and the ROS setup script sourced. I am not responsible for any malfunctions that may be caused by different software versions. Next, compile the project to get setup files, executable programs, and so on stored into the two newly created directories, devel and build, at the root of the workspace, and run setup files to configure your system to use this workspace :
     
     ```console
     $ catkin_make
     $ source devel/setup.bash
     ```
   
   * Using Docker : I let you refer to the README of my docker files repository I mentioned earlier to make use of my ROS Docker image and other helpful Docker tools. -->

## About

The whole study is the subject of a thesis at the French Naval Academy Research Institute. The real-life problem to which it is trying to find an answer is the automatic landing of a helicopter on the deck of a ship. While this is a highly technical manoeuver for humans, it is no less challenging for the autonomous robot. The uncertainty of the maritime environment, the absence of GPS and the presence of wind are all constraints that make the task more complex.

The problem will mainly be dealt with in simulation, using ROS and Gazebo, but tests may be carried out on a drone and a hexapod at the research institute.

The subject is broad; it involves both a phase of extracting information from the environment using various sensors, and a phase of task planning and decision-making during the mission.

In the context of my internship, I focused initially on the processing of images acquired by an onboard camera for the estimation of the pose of the mobile platform. I first had to design a simulated working environment in which to test algorithms. Then, to implement and validate them.
