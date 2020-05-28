#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def handler(data):
	rospy.loginfo("Message received: " + data.data)


if __name__=="__main__":
	rospy.init_node('test_subscriber', anonymous=True)
	rospy.Subscriber("test_string",String, handler)
	rospy.spin()
