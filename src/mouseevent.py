import cv2 
import numpy as np
import argparse
global drawing

refPt = []
drawing = False
reset = False

def handle_event(event, x, y, flags, param):
	global drawing, refPt, reset
	ctrl = False
	
	if (flags >> 3 & 0x01 == 1):
		ctrl = True
	print(ctrl)

	if event == cv2.EVENT_LBUTTONDOWN:
		if ctrl == True:
			refPt.append((x,y))
			param[0] = cv2.circle(param[0], refPt[-1], 3, (0,255,0))
			if len(refPt) >= 2:
				param[0] = cv2.line(param[0], refPt[-2], refPt[-1],(0,255,255), 2)
		elif len(refPt) >= 3:
			param[0] = cv2.line(param[0], refPt[-1], refPt[0], (0,255,255), 2)
			refPt = []
		else:
			reset = True			

	###########################################################################
	# if (flags >> 3 & 0x01 == 1):
	# 	ctrl = True
	# if event == cv2.EVENT_LBUTTONDBLCLK:
	# 	if drawing == False:
	# 		drawing = True
	# 		refPt = []
	# 	elif drawing == True:
	# 		drawing = False
	# 		if len(refPt) > 2:
	# 			param[0] = cv2.line(param[0], refPt[-1], refPt[0], (0, 255, 255), 2)
	# 		else:
	# 			reset = True	
			

	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	# print("left button pressed")
	# 	if drawing == True:
	# 		refPt.append((x,y))
	# 	param[0] = cv2.circle(param[0], refPt[-1], 3, (0,255,0))
	# 	if len(refPt) > 2:
	# 		param[0] = cv2.line(param[0], refPt[-2], refPt[-1], (0, 255, 255), 2)
	############################################################################		


	############################################################################		
	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	if drawing == False:
	# 		refPt.append((x,y))
	# 		drawing = True
	# 	else:
	# 		refPt[-1] = (x,y)

	# if event == cv2.EVENT_LBUTTONUP:
	# 	if drawing == True:
	# 		refPt.append((x,y))
	# 		param[0] = cv2.line(param[0], refPt[-2], refPt[-1], (0, 255, 255), 3 )
	#############################################################################

if __name__=="__main__":

	## LIST ALL MOUSE EVENTS

	# ['EVENT_FLAG_ALTKEY', 'EVENT_FLAG_CTRLKEY', 'EVENT_FLAG_LBUTTON', 
	# 'EVENT_FLAG_MBUTTON', 'EVENT_FLAG_RBUTTON', 'EVENT_FLAG_SHIFTKEY', 
	# 'EVENT_LBUTTONDBLCLK', 'EVENT_LBUTTONDOWN', 'EVENT_LBUTTONUP', 
	# 'EVENT_MBUTTONDBLCLK', 'EVENT_MBUTTONDOWN', 'EVENT_MBUTTONUP', 
	# 'EVENT_MOUSEHWHEEL', 'EVENT_MOUSEMOVE', 'EVENT_MOUSEWHEEL', 
	# 'EVENT_RBUTTONDBLCLK', 'EVENT_RBUTTONDOWN', 'EVENT_RBUTTONUP']
	# events = [i for i in dir(cv2) if 'EVENT' in i]
	# print( events )

	# input()

	ap = argparse.ArgumentParser()
	ap.add_argument("--image", required=True, help="Path to the image" )
	args = vars(ap.parse_args())


	img = cv2.imread(args["image"])
	img_orig = img.copy()

	param = [img]
	cv2.namedWindow("the_window")
	cv2.setMouseCallback("the_window", handle_event, param)


	while True:
		cv2.imshow("the_window", img)
		key = cv2.waitKey(1) & 0xFF
		
		if (key == ord("r") or reset==True):
			img = img_orig.copy()
			reset = False
			param = [img]
			cv2.setMouseCallback("the_window", handle_event, param)
			refPt = []


		if key == ord("q"):
			break

	cv2.destroyAllWindows()		