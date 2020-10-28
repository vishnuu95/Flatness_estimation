#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visualise_depth_l515 import *

if __name__=="__main__":

	# First user trigger
	
		# Capture once - returns img1, ROI1, roi-pcl1, Pose1
	# Visualise plane fit	

# Move camera around

	# Upon user trigger
		# Capture once - returns img2, pcl2, Pose2
	# Transform	pcl2 from pose 2 to pose 1
	# extract roi-pcl1 points range of points from pcl2
	# Fit plane


