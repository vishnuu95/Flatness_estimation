# Flatness_estimation
Project to estimate flatness of surface using the depth camera. 

## Requirements
* Numpy 1.10
* python 2
* realsense library
* ROS - kinetic

## Overview
* This project aims to estimate the flatness of a surface using a depth camera(using Intel realsense cameras right now).
* Project has code for flatness estimation using intel realsense D435 and L515.
* L515 is superior in terms of depth accuracy but still is limited to indoor environments only. 

## TODO
* The project estimates flatness for a single trigger capture given by a user. The future aim is to perform multi trigger capture to estimate depth values better.  
There are three paths forward as of date (27th October 2020).

* 1) **Capture multiple frames without moving the camera :**  Upon a single trigger by the user, the algorithm captures multiple frames, collects depth values of points of interest over time and gives a better estimate of the flatness. In case the flatness of the surface changes, we conclude that there is inherent noise in each capture and might have to do multiple capture at a location by default. 
* 2) **Capture multiple frames while slightly moving the camera :**  We encourage the user to move the camera around slightly so that we estimate depth values from different view points. This might give in more accurate depth values of the same points when averaging over multiple captures. This task will be performed with the help of rtabmap_ros package which performs ICP using the IMU and the Point cloud from the Lidar L515. The rtabmap_ros package gives out pose estimate after performing ICP, using which, we can transform the newly caputured pointcloud to original frame and thereby get more depth estimates of the same points of interest. 
* 3) **Capture multiple frames while moving the camera to a new location :** We encourage the user to move to completely new location while looking at the same point of interest. While moving the camera around, Rtabmap_ros will run in the background to continuosly estimate the pose of the camera. This way, when the user gives a trigger, the captured pointcloud at that instant can be transformed back to original frame by the pose estimate from rtabmap_ros. This path would benefit us from looking at points from a totally new perspective and estimate depth in a hopefully better way. 

## Backlog
* Add requirement download links and complete remaining requirements.
* Add product links.
* Add run instructions.
* Add modifications to realsense rs_camera launch files for running. 
