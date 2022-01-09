import numpy as np
import cv2
import skimage.io 
import skimage.color

from matchPics import matchPics
from planarH import computeH_ransac, compositeH
#Import necessary functions



#Write script for Q3.9
def main():
    cv_cover = cv2.imread('./data/cv_cover.jpg')
    cv_desk = cv2.imread('./data/cv_desk.png')
    hp_cover = cv2.imread('./data/hp_cover.jpg')
    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
    bestH2to1, inliers = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])
    composite_img = compositeH(bestH2to1, hp_cover, cv_desk)
    cv2.imshow('r', composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
