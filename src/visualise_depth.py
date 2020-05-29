#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge


def handler(data):

	rospy.loginfo("Image Message received!" )
	# rospy.loginfo("Header: " +  str(data.header))
	# rospy.loginfo("Height: " +  str(data.height))
	# rospy.loginfo("Width: " +  str(data.width))
	# rospy.loginfo("step: " +  str(data.step))
	# rospy.loginfo("data" + np.array(data.data))
	rospy.loginfo("image details: " + str(bridge.imgmsg_to_cv2(data).shape) )

	# filewriter.write(bridge.imgmsg_to_cv2(data, "bgr8"))
	cv2.imshow("im",bridge.imgmsg_to_cv2(data, "bgr8")) # Works !
	cv2.waitKey(50)

if __name__=="__main__":

	filewriter = cv2.VideoWriter("~/test_video", cv2.VideoWriter_fourcc(*'mp4v'), 24, (640,480))
	bridge = CvBridge()

	rospy.init_node("visualise_depth", anonymous=True)	
	rospy.Subscriber("camera/color/image_raw", Image, handler)

	rospy.spin()
