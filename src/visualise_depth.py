#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import sys
global reset, refPt, ctrl, drawing, completed, img_held, img, prev_ctrl


def handler(data):
	global reset, refPt, ctrl, drawing, completed, img_held, img
	# rospy.loginfo("Image Message received!" )
	# rospy.loginfo("Header: " +  str(data.header))
	# rospy.loginfo("Height: " +  str(data.height))
	# rospy.loginfo("Width: " +  str(data.width))
	# rospy.loginfo("step: " +  str(data.step))
	# rospy.loginfo("data" + np.array(data.data))
	# rospy.loginfo("image details: " + str(bridge.imgmsg_to_cv2(data).shape) )

	# filewriter.write(bridge.imgmsg_to_cv2(data, "bgr8"))
	img = bridge.imgmsg_to_cv2(data, "bgr8")
	# param = [img]

	
def mouse_callback(event, x, y, flags, param):
	global reset, refPt, ctrl, drawing, completed, img_held, prev_ctrl, img

	if (completed != True):
		prev_ctrl = ctrl
		ctrl = False

		if (flags >> 3 & 0x01):
			ctrl = True

		# print ctrl - Works

		if(prev_ctrl==False and ctrl==True):
			img_held = img.copy()
			refPt = []

		
		if event == cv2.EVENT_LBUTTONDOWN:
			if (ctrl == False and drawing == True):
				img_held = cv2.line(img_held, refPt[-1], refPt[0], ( 0, 255, 255), 2)
				completed = True
			elif (ctrl == True and drawing == True):
				refPt.append((x,y))
				img_held = cv2.circle(img_held, refPt[-1], 3, (0,255,0))
				img_held = cv2.line(img_held, refPt[-2], refPt[-1], ( 0, 255, 255), 2)
			elif (ctrl == True):
				refPt.append((x,y))
				img_held = cv2.circle(img_held, refPt[-1], 3, (0,255,0))
				if len(refPt) == 2:
					cv2.line(img_held, refPt[-2], refPt[-1], ( 0, 255, 255), 2)
					drawing = True
				


if __name__=="__main__":
	np.set_printoptions(threshold=sys.maxsize)

	# filewriter = cv2.VideoWriter("~/test_video", cv2.VideoWriter_fourcc(*'MJPG'), 24, (640,480))
	bridge = CvBridge()
	img_held = np.array([], dtype=np.uint8)
	img = np.array([], dtype = np.uint8)
	cv2.namedWindow("img_window")

	rospy.init_node("visualise_depth", anonymous=True)

	rospy.Subscriber("camera/color/image_raw", Image, handler) # Color image subscribe
	rospy.wait_for_message("camera/color/image_raw", Image)
	cv2.setMouseCallback("img_window", mouse_callback)
	# rospy.Subscriber("camera/depth/image_rect_raw", Image, handler) # Depth image subscribe
	
	completed = False
	drawing = False
	ctrl = False
	prev_ctrl = False
	reset = False
	loop = True

	while(loop):
		key = cv2.waitKey(1) & 0xFF
		if (key == ord("r") or key == ord("R")):
			completed = False
			drawing = False
			ctrl = False
			prev_ctrl = False
			reset = False
			refPt = []
			continue

		if (key == ord("q") or key == ord("Q")):
			loop = False
			break
		if (ctrl == True or drawing == True):
			cv2.imshow("img_window", img_held) # Color image visualisation
			cv2.waitKey(5)	
		else:
			cv2.imshow("img_window", img) # Color image visualisation
			cv2.waitKey(5)
		
	

	if completed == True:
		# show the image with shaded part
		y = np.arange(0, img.shape[0])
		x = np.arange(0, img.shape[1])
		refPt_fill = []
		# print np.array(refPt)
		

		# refPt = np.array([refPt])
		# print refPt
		print "--------"
		img_held2 = img_held.copy()
		# hull = cv2.convexHull(refPt)
		# hull_list = [hull]
		# drawing = np.zeros((img_held.shape[0], img_held.shape[1], 3), dtype = np.uint8)
		# cv2.drawContours(drawing, refPt, 0, (0,0,255))
		# cv2.drawContours(drawing, hull_list, 0, (0,0,255))

		# cv2.imshow("contours",drawing)
		# cv2.waitKey(0)
		############ METHOD TO FIND ALL POINTS INSIDE POLYGON ###################
		for i in range(len(refPt)):
			# x = ((x2-x1)/(y2-y1))*y + c
			if (refPt[i][1] != refPt[i-1][1] ): # if slope not inf
				if (refPt[i-1][1] < refPt[i][1]):
					m = float(refPt[i][0] - refPt[i-1][0])/(refPt[i][1] - refPt[i-1][1])
					c = refPt[i-1][0] - m*refPt[i-1][1] 
					y_vec = range(refPt[i-1][1], refPt[i][1]+1)
					x_vec = [int(round(m*(y_vec_v) + c)) for y_vec_v in y_vec]
					for x_, y_ in zip(x_vec, y_vec):
						refPt_fill.append((x_, y_))
					x_vec_1 = range(min(refPt[i-1][0], refPt[i][0]), max(refPt[i-1][0], refPt[i][0]))
					y_vec_1 = [int(round((x_vec - c)/m)) for x_vec in x_vec_1]
					for x_, y_ in zip(x_vec_1, y_vec_1):
						refPt_fill.append((x_, y_))
											
				else:
					m = float(refPt[i-1][0] - refPt[i][0])/(refPt[i-1][1] - refPt[i][1])
					c = refPt[i-1][0] - m*refPt[i-1][1] 
					y_vec = range(refPt[i][1], refPt[i-1][1]+1)
					x_vec = [int(round(m*(y_vec_v) + c)) for y_vec_v in y_vec]
					for x_, y_ in zip(x_vec, y_vec):
						refPt_fill.append((x_, y_))
					x_vec_1 = range(min(refPt[i-1][0], refPt[i][0]), max(refPt[i-1][0], refPt[i][0]))
					y_vec_1 = [int(round((x_vec - c)/m)) for x_vec in x_vec_1]
					for x_, y_ in zip(x_vec_1, y_vec_1):
						refPt_fill.append((x_, y_))	
													
			else: # slope is inf
				y_vec = (abs(refPt[i][0]-refPt[i-1][0])*[refPt[i][1]])
				if refPt[i-1][0] < refPt[i][1]:
					x_vec = range(refPt[i-1][1], refPt[i][1])
				else: 	
					x_vec = range(refPt[i][1], refPt[i-1][1])

				for x_, y_ in zip(x_vec, y_vec):
					refPt_fill.append((x_, y_))
		#########################################################################


			# print ("every itr", refPt_fill_i)	
		img_held2 = img_held.copy()
		# print(refPt_fill)
		refPt_fill = np.array(refPt_fill)
		# print(refPt_fill.shape)
		# print(refPt_fill[0])
		# print(refPt_fill[1])
		# print(refPt_fill[2])
		# for j in range(	len(refPt_fill)):
		# 	img_held2 = cv2.circle(img_held2, tuple(refPt_fill[j]), 3, (0,255,0))
		# # # print ("end of itrs", refPt_fill)
		# cv2.imshow("next_window", img_held2)
		# cv2.waitKey(0)
		# print refPt_fill
		cv2.destroyAllWindows()
		img_mask = np.full((img_held.shape), True, dtype=np.bool)
		sortedPts = np.sort(refPt_fill.view('i8,i8'), order=['f0'], axis=0).view(np.int)
		# print sortedPts.shape
		sort_mask = np.full((sortedPts.shape), True, dtype = np.bool)
		anc = sortedPts[0]
		for i in range(1, sortedPts.shape[0]):
			if anc[0] == sortedPts[i][0] and anc[1] == sortedPts[i][1]:
				sort_mask[i] = np.array([False, False], dtype= np.bool)
			anc = sortedPts[i]
		
		sortedPts = sortedPts[sort_mask].reshape(-1,2)
		# for j in range(	len(sortedPts)):
		# 	img_held2 = cv2.circle(img_held2, tuple(sortedPts[j]), 1, (0,0,255),0)
		# # # # print ("end of itrs", refPt_fill)
		# cv2.imshow("next_window", img_held2)
		# cv2.waitKey(0)
		# print sortedPts.shape
		# print sortedPts
		# input()
		anc = sortedPts[0]
		# val = False
		lst = np.array([sortedPts[0]])
		for i in range(1, sortedPts.shape[0]):
		# for i in range(1, 100):
			if anc[0] == sortedPts[i][0]:
				lst = np.append(lst, np.array([sortedPts[i]]), axis = 0)
			else:
				lst = np.sort(lst.view('i8,i8'), order=['f0'], axis=0).view(np.int)
				val = False
				for j in range(lst.shape[0]-1):
					# print ("j: ",j)
					if (lst[j+1][1] - lst[j][1]) > 1:
						print(lst[0][0])
						y = np.arange(lst[j][1], lst[j+1][1])
						x = np.full_like(y, lst[0][0])
						# print(x,y)
						# input()
						img_mask[y,x,:] = (val, val, val) ## Why is it y,x ? suppossed to be x,y right?
						val = not val
						# print val

				lst = np.array([sortedPts[i]])	
			anc = sortedPts[i]

		img_held2[img_mask] = 255

		cv2.imshow("next_window", img_held2)
		cv2.waitKey(0)
		
		# anc = sortedPts[0][0]
		# clustPts = np.array([sortedPts[0]])
		# print clustPts
		# clust = np.array([[sortedPts[0]]])
		# # input()
		# # print(sortedPts[20][0], anc)
		# for i in range(1, 20):
		# 	if sortedPts[i][0] != anc:
		# 		print ("inside if clustPts ", clustPts)
		# 		clustPts = np.sort(clustPts, axis = 1)
		# 		print clust.shape
		# 		print np.array([clustPts]).shape	
		# 		clust = np.append(clust, np.array([clustPts]), axis = 0)
		# 		clustPts = np.array([sortedPts[i][0]])
		# 		anc = sortedPts[i][0]
		# 		print("inside if clust ", clust)
		# 	else:	# input()
		# 		print clustPts.shape
		# 		print np.array([sortedPts[i]]).shape
		# 		clustPts = np.append(clustPts, np.array([sortedPts[i]]), axis = 0)
		# 		# print (np.array([sortedPts[i]]))
		# 		# print ("outside if clustPts", clustPts)	
		# clust = np.append(clust, clustPts)
		# clust = clust[1][:][:]
		# print("outside if clust ", clust)
		# # print clust	
		

		
	

	cv2.destroyAllWindows()