#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


def handler(data):

	rospy.loginfo("Image Message received!" )
	rospy.loginfo("Header: " +  str(data.header))
	rospy.loginfo("Height: " +  str(data.height))
	rospy.loginfo("Width: " +  str(data.width))
	# rospy.loginfo(type(data))
	# rospy.loginfo("step: " +  str(data.step))
	# rospy.loginfo( data.data)
	cvimg = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	rospy.loginfo("image details: " + str(cvimg.shape) )

	


if __name__=="__main__":

	bridge = CvBridge()

	rospy.init_node("visualise_depth", anonymous=True)	
	# rospy.Subscriber("camera/depth/image_rect_raw", Image, handler) # Color image subscribe
	rospy.Subscriber("camera/color/image_raw", Image, handler)
	rospy.spin()
