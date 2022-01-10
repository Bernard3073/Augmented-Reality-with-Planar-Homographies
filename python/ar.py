import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import plotMatches


#Write script for Q4.1
def main():
    ar_source = cv2.VideoCapture('./data/ar_source.mov')
    book = cv2.VideoCapture('./data/book.mov')
    # get frame count
    n_frames = int(book.get(cv2.CAP_PROP_FRAME_COUNT))
    if(book.isOpened()==False):
        print("Error")
    _, prev = book.read()    
    for i in range(n_frames-2):
        ret, curr = book.read()
        _, frame2 = ar_source.read()
        if ret:
            matches, locs1, locs2 = matchPics(prev, curr)
            bestH2to1, inliers = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])
            plotMatches(prev, curr, matches, locs1, locs2)
            # cv2.imshow('f', frame)
            if cv2.waitKey(30) & 0xFF == ord("q"): 
                break 
        else:
            break
    book.release()
    ar_source.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()