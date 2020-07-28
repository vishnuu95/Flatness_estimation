import cv2
import numpy as np
from visualise_depth import *
import fire

def test_draw_contour():
	
	test_cases = [[(300, 400), (420, 520), (470, 680), (220, 700)],
					[(300, 400), (420, 520), (420, 680), (300, 700)], 
					[(300, 400), (420, 400), (420, 700), (300, 700)], 
					[(300, 400), (500, 300), (500, 700), (300, 500)]]
	for test in test_cases:
		img = np.zeros((720, 1280, 3), dtype = np.uint8)
		cv2.imshow("contour", img)
		cv2.waitKey(0)
		filled_points = []
		for point in test:
			img = cv2.circle(img, (point[1],point[0]), 3, (0,255,0))
		cv2.imshow("contour", img)
		cv2.waitKey(0)			
		filled_points = find_contour_pts(test, filled_points)
		for point in filled_points:
			img[point[0], point[1],:] = (255, 255, 0)	
		cv2.imshow("contour", img)
		cv2.waitKey(0)

def test_remove_duplicates():
	
	start_test_cases = [(300, 400), (420, 520), (470, 680), (220, 800)]
	test_case = []
	for point in start_test_cases:
		test_case.append([point]*3)
	pts = remove_duplicates(test_frame)

	print (pts)


def test_draw_fill_polygon():

	test_cases = [[(300, 400), (420, 520), (470, 680), (220, 800)],
					[(300, 400), (420, 520), (420, 680), (300, 800)], 
					[(300, 400), (420, 400), (420, 800), (300, 800)]]
	for test in test_cases:
		img = np.zeros((720, 1280, 3), dtype = np.uint8)
		filled_points = []
		for point in test:
			img = cv2.circle(img, (point[1], point[0]), 3, (0,255,0))
		filled_points = find_contour_pts(test, filled_points)
		for point in filled_points:
			img[point[0], point[1],:] = (255, 255, 0)	
		cv2.imshow("contour", img)
		cv2.waitKey(0)

	sorted_pts = remove_duplicates(filled_points)
	img_mask = np.full((img.shape), False, dtype=np.bool)
	img_mask = find_interior_pts(sorted_pts, img_mask)
	cv2.imshow("img mask", img_mask)
	cv2.waitKey(0)

		

if __name__=="__main__":
	fire.Fire()