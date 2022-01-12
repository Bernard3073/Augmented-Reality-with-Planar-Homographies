import numpy as np
import cv2
#Import necessary functions
import sys
sys.path.append('./python')
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import plotMatches



#Write script for Q4.2x
def main():
    pano_left = cv2.imread('./data/pano_left.jpg')
    pano_right = cv2.imread('./data/pano_right.jpg')
    matches, locs1, locs2 = matchPics(pano_left, pano_right)
    # plotMatches(pano_left, pano_right, matches, locs1, locs2)
    H, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])
    # Apply panaroma correction
    result_width = pano_left.shape[1] + pano_right.shape[1]
    result_height = pano_left.shape[0] + pano_right.shape[0]
    result = cv2.warpPerspective(pano_right, H, (result_width, result_height))
    result[0:pano_left.shape[0], 0:pano_left.shape[1]] = pano_left
    cv2.imshow('r', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()