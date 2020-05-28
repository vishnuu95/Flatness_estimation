#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2 


def handler(data):

	rospy.loginfo("Image Message received!" )
	rospy.loginfo("Header: " +  str(data.header))
	rospy.loginfo("Height: " +  str(data.height))
	rospy.loginfo("Width: " +  str(data.width))
	rospy.loginfo("step: " +  str(data.step))
	rospy.loginfo("-----------------------")

if __name__=="__main__":
	filewriter = cv2.VideoWriter()
	rospy.init_node("visualise_depth", anonymous=True)
	rospy.Subscriber("camera/color/image_raw", Image, handler)	
	rospy.spin()
