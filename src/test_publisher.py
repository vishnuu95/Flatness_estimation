#!/usr/bin/env python

import rospy
from std_msgs.msg import String


def publish_frames():
	while not rospy.is_shutdown():
		try_str = "it Works!"
		rospy.loginfo(try_str)
		pub.publish(try_str)
		rate.sleep()


if __name__=="__main__":
	pub = rospy.Publisher('segmented_frames', String)
	rospy.init_node('visualise_depth', anonymous = True)
	rate = rospy.Rate(1)
	try:
		publish_frames()
	except rospy.ROSInterruptException:
		pass	