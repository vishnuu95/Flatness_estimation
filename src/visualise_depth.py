#!/usr/bin/env python

import rospy
from std_msgs.msg import String

if __name__=="__main__":
	pub = rospy.Publisher('segmented_frames', String)
	rospy.init_node('visualise_depth', anonymous = True)
	rate = rospy.Rate(10)
