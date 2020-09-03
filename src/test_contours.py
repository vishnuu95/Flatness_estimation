import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/home/vishnuu/test2.png", 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 80, 100)
img_, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("con", img_)
cv2.waitKey(0)
pts =  np.squeeze(contours[0])
print pts.shape
plt.plot(pts[:,0], pts[:,1])
plt.show()
