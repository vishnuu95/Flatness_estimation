#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
import geometry_msgs

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(x=0,y=0,z=0)
def subs_callback(data):
	# print data.orientation
	# print "----------"
	
	msg = geometry_msgs.msg.Pose()
	# print msg.position
	msg.orientation = data.orientation
	subs_callback.x += 0.5*data.linear_acceleration.x*(0.0025)*(0.0025)
	subs_callback.y += 0.5*data.linear_acceleration.y*(0.0025)*(0.0025)
	subs_callback.z += 0.5*data.linear_acceleration.z*(0.0025)*(0.0025)
	msg.position.x = subs_callback.x
	msg.position.y = subs_callback.y
	msg.position.z = subs_callback.z
	pub.publish(msg)

	


if __name__=="__main__":
	rospy.init_node("est_motion", anonymous=True)
	rospy.wait_for_message("/imu/data", Imu)
	sub = rospy.Subscriber("imu/data", Imu, subs_callback)
	pub = rospy.Publisher('lidarPose', Pose) # Name of publishing topic, Message type
	
	rospy.spin()