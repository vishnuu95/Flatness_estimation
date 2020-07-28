#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
global reset, refPt, ctrl, drawing, completed, img_held, img, prev_ctrl, img_depth, img_depth_held


def handler_img(data):
	global reset, refPt, ctrl, drawing, completed, img_held, img, img_depth, img_depth_held
	# rospy.loginfo("Image Message received!" )
	# rospy.loginfo("Header: " +  str(data.header))
	# rospy.loginfo("Height: " +  str(data.height))
	# rospy.loginfo("Width: " +  str(data.width))
	# rospy.loginfo("step: " +  str(data.step))
	# rospy.loginfo("data" + np.array(data.data))
	# rospy.loginfo("image details: " + str(bridge.imgmsg_to_cv2(data).shape) )

	# filewriter.write(bridge.imgmsg_to_cv2(data, "bgr8"))
	img = bridge.imgmsg_to_cv2(data, "bgr8")
	# print(type(img))
	# param = [img]

def handler_depth(data):
	global reset, refPt, ctrl, drawing, completed, img_held, img, img_depth
	img_depth = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
	
def mouse_callback(event, x, y, flags, param):
	global reset, refPt, ctrl, drawing, completed, img_held, prev_ctrl, img, img_depth_held

	if (completed != True):
		prev_ctrl = ctrl
		ctrl = False

		if (flags >> 3 & 0x01):
			ctrl = True

		# print ctrl - Works

		if(prev_ctrl==False and ctrl==True):
			img_held = img.copy()
			img_depth_held = img_depth.copy()
			refPt = []

		
		if event == cv2.EVENT_LBUTTONDOWN:
			if (ctrl == False and drawing == True):
				img_held = cv2.line(img_held, (refPt[-1][1], refPt[-1][0]), (refPt[0][1], refPt[0][0]), ( 0, 255, 255), 2)
				completed = True
			elif (ctrl == True and drawing == True):
				refPt.append((y, x))
				img_held = cv2.circle(img_held, (refPt[-1][1], refPt[-1][0]), 3, (0,255,0))
				img_held = cv2.line(img_held, (refPt[-2][1], refPt[-2][0]), (refPt[-1][1], refPt[-1][0]), ( 0, 255, 255), 2)
			elif (ctrl == True):
				refPt.append((y, x))
				img_held = cv2.circle(img_held, (refPt[-1][1], refPt[-1][0]), 3, (0,255,0))
				if len(refPt) == 2:
					cv2.line(img_held, (refPt[-2][1], refPt[-2][0]), (refPt[-1][1], refPt[-1][0]), ( 0, 255, 255), 2)
					drawing = True
				
def find_contour_pts(refPt, refPt_fill):
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
				if refPt[i-1][0] < refPt[i][0]:
					x_vec = range(refPt[i-1][0], refPt[i][0])
				else: 	
					x_vec = range(refPt[i][0], refPt[i-1][0])

				for x_ in x_vec:
					refPt_fill.append((x_, refPt[i][1]))
		refPt_fill = np.array(refPt_fill)
		return refPt_fill			
		#########################################################################

def remove_duplicates(refPt_fill):
	sortedPts = np.sort(refPt_fill.view('i8,i8'), order=['f0'], axis=0).view(np.int) # sorts along 1st axis
	# print sortedPts.shape
	sort_mask = np.full((sortedPts.shape), True, dtype = np.bool) # Mask for sorted pts same shape. 
	anc = sortedPts[0] # Take the first point as anchor
	for i in range(1, sortedPts.shape[0]): # iterate over all pts
		if anc[0] == sortedPts[i][0] and anc[1] == sortedPts[i][1]: # if anc point is same as sorted pt
			sort_mask[i] = np.array([False, False], dtype= np.bool) # mark as false: duplicate
		anc = sortedPts[i] # Shift the anc by default. 
	sortedPts = sortedPts[sort_mask].reshape(-1,2) # Reshape after masking out.

	return sortedPts	
		
def find_interior_pts(sortedPts, img_mask):
	anc = sortedPts[0]
	# val = False
	lst = np.array([sortedPts[0]])
	for i in range(1, sortedPts.shape[0]):
	# for i in range(1, 100):
		if anc[0] == sortedPts[i][0]:
			lst = np.append(lst, np.array([sortedPts[i]]), axis = 0)
		else:
			lst = np.sort(lst.view('i8,i8'), order=['f0'], axis = 1).view(np.int)
			val = True
			for j in range(lst.shape[0]-1):
				# print ("j: ",j)
				if (lst[j+1][1] - lst[j][1]) > 1:
					# print(lst[0][0])
					y = np.arange(lst[j][1], lst[j+1][1])
					x = np.full_like(y, lst[0][0])
					# print(x,y)
					# input()
					img_mask[x,y,:] = (val, val, val) ## Why is it y,x ? suppossed to be x,y right?
					val = not val
					# print val

			lst = np.array([sortedPts[i]])	
		anc = sortedPts[i]

	return img_mask	

def visualize_mask(img_held2, img_mask, view_mask=False):
	if (view_mask):
		img_held2[img_mask] = 255
		cv2.imshow("next_window", img_held2)
		cv2.waitKey(0)

def visualize_depth_img(img_depth_held2, img_mask, view_d_mask=False):
	img_depth_held2 = img_depth_held.copy()
	img_depth_held2[img_mask[:,:,0]] = 0
	cv2.imshow("next_window", img_depth_held2)
	cv2.waitKey(0)

def create_pc(img_mask, img_depth_held):
	############## CREATE X,Y,Z ARRAY #################
	idx = np.where(img_mask[:,:,0] == True)
	idx = np.array(idx).T
	pcl = np.hstack((idx, img_depth_held[img_mask[:,:,0]][:,None]))

	return pcl, idx

def visualize_color_gradient(pcl, img_held, view_grad=False):
############## Visualise gradient #################
	idx_z_min = np.argmin(pcl[:,2])
	idx_z_max = np.argmax(pcl[:,2])
	z_min = np.amin(pcl[:,2])
	z_max = np.amax(pcl[:,2])
	scaling = float(255-50)/(z_max - z_min)
	# print(idx_z_min, idx_z_max, scaling)
	img_held3 = img_held.copy()
	for i in range(idx.shape[0]):
		shade = (pcl[i,2] - z_min)*scaling + 50
		img_held3[pcl[i,0], pcl[i,1], :] = (shade, 0, 0)
	cv2.imshow("gradient", img_held3)
	cv2.waitKey(0)

def fit_plane(pcl, idx, img_mask, img_depth_held):
	################ PLANE FIT ###################
	# ax + by + c = z equation of plane
	ones = np.ones((pcl.shape[0],1))
	A = np.hstack((idx, ones))
	# print A
	z = img_depth_held[img_mask[:,:,0]][:,None]
	# print z
	xTx_inv = np.linalg.inv(np.matmul(A.T, A))
	xTy = np.matmul(A.T, z)
	sol = np.matmul(xTx_inv, xTy)
	# print sol
	# print sol.shape
	##############################################
	return sol, z

def analyse_plane_fit(sol, pcl, idx, z):
	################ ANALYSIS ####################
	# Along Z
	dist = np.zeros((idx.shape[0],1))
	dist_shortest = np.zeros((idx.shape[0],1))
	denom = np.sqrt(sol[0][0]**2 + sol[1][0]**2 + 1)
	for i in range(idx.shape[0]):
		z_plane = sol[0][0]*idx[i][0] + sol[1][0]*idx[i][1] + sol[2][0]
		dist[i] = float(np.abs(z[i] - z_plane))
		dist_shortest[i] = float(np.abs(sol[0][0]*pcl[i][0] + sol[1][0]*pcl[i][1] + sol[2][0] - pcl[i][2]))/denom
	z_mean = np.mean(dist, axis=0)
	z_std = np.std(dist, axis=0)
	print ("######### Along Z direction #########")
	print ("mean: ", z_mean)
	print ("std_dev: ", z_std)
	# Shortest dist
	p_mean = np.mean(dist_shortest, axis = 0)
	p_std = np.std(dist_shortest, axis = 0)
	print ("######### Shortest Distance #########")
	print ("mean: ", p_mean)
	print ("std_dev: ", p_std)

	return z_mean, z_std, p_mean, p_std, dist	

def visualize_outliers(dist, idx, threshold, img_held2, view_outliers=False):
	######### VISUALISE OUTLIERS ########## 
		
	# print ("idx shape", idx.shape)
	idx_outliers = np.where(dist > threshold, True, False)
	# print ("outliers idx shape: ", idx_outliers.shape)
	outliers = idx[idx_outliers[:,0], :]
	# print ("outliers shape: ", outliers.shape)
	# outliers = outliers.reshape(-1,3)
	# print("outliers shape after reshape: ", outliers.shape)
	img_held2[outliers[:,0], outliers[:,1], :] = (0,255,255)
	cv2.imshow("next_window", img_held2)
	cv2.waitKey(0)

def visualize_3d_plot(pcl, idx, sol):
	############ 3D Visualisation ##########
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.plot(pcl[:,0], pcl[:,1], pcl[:,2])
	# Point on the plane
	z_plane = np.zeros((idx.shape[0]))
	x_plane = np.arange(np.amin(pcl[:,0]), np.amax(pcl[:,0]))
	y_plane = np.arange(np.amin(pcl[:,1]), np.amax(pcl[:,1]))
	xx, yy = np.meshgrid(x_plane, y_plane)
	z_plane = sol[0][0]*xx + sol[1][0]*yy + sol[2][0]
	ax.contour3D(xx, yy, z_plane, 100, cmap = 'binary')
	ax.set_xlabel("pixel x")
	ax.set_ylabel("pixel y")
	ax.set_zlabel("depth value")
	plt.show()


if __name__=="__main__":
	# To print the entire array. 
	np.set_printoptions(threshold=sys.maxsize)

	## Intialization 
	# Init for Cv bridge for ROS to OpenCV
	bridge = CvBridge()

	# Init for variables
	img_held = np.array([], dtype=np.uint8)
	img = np.zeros([], dtype = np.uint8)
	cv2.namedWindow("img_window")

	# Init ros nodes
	rospy.init_node("visualise_depth", anonymous=True)
	rospy.wait_for_message("camera/color/image_raw", Image)
	rospy.Subscriber("camera/color/image_raw", Image, handler_img)
	rospy.wait_for_message("camera/aligned_depth_to_color/image_raw", Image)
	rospy.Subscriber("camera/aligned_depth_to_color/image_raw", Image, handler_depth)
	# rospy.Subscriber("camera/depth/image_rect_raw", Image, handler) # Depth image subscribe
	# rospy.wait_for_message("camera/color/image_raw", Image)

	# Init mouse event capture handler - OpenCV
	cv2.setMouseCallback("img_window", mouse_callback)
	
	# Init variables for state machine tracking. 	
	completed = False # Completed drawing the polygon
	drawing = False # Var to track if user is still drawing the polygon
	ctrl = False # Var to track if user has ctrl key pressed
	prev_ctrl = False # Var to track if ctrl was previously held. This is used to find ctrl release
	reset = False # Flag to reset the state to initial state
	loop = True # Var to check that polygon selection is still on. 
	
	# Flags for post polygon capture processing.
	view_mask = True
	view_d_mask = True
	view_grad = True
	threshold = 15
	view_outliers = True
	
	## Application Running 
	# While loop to that resets, breaks and allows for interrupts to take place. 
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
		
	# if successfully drawn, not quit in between:
	if completed == True:

		# show the image with shaded part
		# y = np.arange(0, img.shape[0])
		# x = np.arange(0, img.shape[1])
		print (refPt)
		refPt_fill = []
		img_held2 = img_held.copy()
		refPt_fill = find_contour_pts(refPt, refPt_fill)
		# print (refPt_fill.shape)
		sortedPts = remove_duplicates(refPt_fill)
		# print(sortedPts.shape)
		img_mask = np.full((img_held.shape), False, dtype=np.bool)	 
		# print(img_mask.shape)
		img_mask = find_interior_pts(sortedPts, img_mask)

		
		#visualise regular image
		visualize_mask( img_held2, img_mask, view_mask)
		

		#visualise depth image cutout
		visualize_depth_img(img_depth_held, img_mask, view_mask)
		

		pcl, idx = create_pc(img_mask, img_depth_held)

		visualize_color_gradient(pcl, img_held, view_mask)

		sol, z = fit_plane(pcl, idx, img_mask, img_depth_held)
	
		z_mean, z_std, p_mean, p_std, dist = analyse_plane_fit(sol, pcl, idx, z)
		
		visualize_outliers(dist, idx, threshold, img_held2, view_outliers)

		visualize_3d_plot(pcl, idx, sol)



		







	cv2.destroyAllWindows()