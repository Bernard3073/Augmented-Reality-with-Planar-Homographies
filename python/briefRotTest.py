import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
from helper import plotMatches

#Q3.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('./data/cv_cover.jpg')
hist = np.zeros(36)

for i in range(36):
	#Rotate Image
	cv_cover_rot = cv2.rotate(cv_cover, cv2.cv2.ROTATE_90_CLOCKWISE)
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rot)
	plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
	#Update histogram
	hist.append(matches)

#Display histogram
plt.hist()
