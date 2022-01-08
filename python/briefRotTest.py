import numpy as np
import cv2
from matchPics import matchPics
import matplotlib.pyplot as plt
from helper import plotMatches
from tqdm import tqdm
import scipy

#Q3.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('./data/cv_cover.jpg')
hist = np.zeros(5)
angles = np.arange(0, 50, 10)

for i in tqdm(range(5)):
	#Rotate Image
	cv_cover_rot = scipy.ndimage.rotate(cv_cover, angles[i])
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, cv_cover_rot)
	# plotMatches(cv_cover, cv_cover_rot, matches, locs1, locs2)
	#Update histogram
	hist[i] = len(matches)

#Display histogram
plt.figure()
plt.bar(angles, hist)
plt.show()