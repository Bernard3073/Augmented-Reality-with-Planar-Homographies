import numpy as np
import cv2
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import plotMatches
from loadVid import *
from tqdm import tqdm
import multiprocessing
from pathlib import Path

#Write script for Q4.1
def cropBlackBar(img):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # invert gray image
    gray = 255 - gray
    # gaussian blur
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    # threshold
    res = cv2.threshold(blur, 235, 255, cv2.THRESH_BINARY)[1]
    # use morphology to fill holes at the boundaries
    kernel = np.ones((5,5), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    # invert back
    res = 255 - res
    # get contours and get bounding rectangle
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

def homography_warp(args):

    # f1 = ar_source[i%ar_source_frames, :, :, :]
    # f2 = book[i, :, :, :]
    # f2 = np.squeeze(f2)
    # ar_source_crop = f1[:, int((ar_source_width-book_width)/2):int((ar_source_width+book_width)/2)]
    ar_source_crop, cv_cover, book= args
    # adjusting ar_source frame
    ar_source_crop = cv2.resize(ar_source_crop, dsize=(cv_cover.shape[1], cv_cover.shape[0]))
    matches, locs1, locs2 = matchPics(cv_cover, book)
    bestH2to1, _ = computeH_ransac(locs1[matches[:, 0]], locs2[matches[:, 1]])
    # plotMatches(prev, curr, matches, locs1, locs2)
    composite_img = compositeH(bestH2to1, ar_source_crop, book)
    # cv2.imshow('f', composite_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("Frame %s Processed" % i)
    return composite_img

def main():
    # ar_source = loadVid('./data/ar_source.mov')
    # book = loadVid('./data/book.mov')

    # Read videos using cv2 or from previously saved .npy
    ar_source = Path('ar_source.npy')
    if ar_source.exists():
        ar_source = np.load('ar_source.npy')
    else:
        ar_source = loadVid('./data/ar_source.mov')    # 511 frames 360x640
        np.save('ar_source.npy', ar_source)
    book = Path('book.npy')
    if book.exists():
        book = np.load('book.npy')
    else:
        book = loadVid('./data/book.mov')           # 641 frames 480x640
        np.save('book.npy', book)
    cv_cover = cv2.imread('./data/cv_cover.jpg')
    # Crop the top & bottom black bar
    frame0 = ar_source[0]
    x, y, w, h = cropBlackBar(frame0)
    frame0 = frame0[y:y+h, x:x+w]
    # Crop left and right: only the central region is used for AR
    H, W = frame0.shape[:2]
    width = cv_cover.shape[1] * H / cv_cover.shape[0]
    wStart, wEnd = np.round([W/2 - width/2, W/2 + width/2]).astype(int)
    frame0 = frame0[:, wStart:wEnd]

    # Crop all frames in ar_source
    new_source = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])
    # ar_source_frames, ar_source_height, ar_source_width, _ = ar_source.shape
    # book_frames, book_height, book_width, _ = book.shape
    
    # Multiprocess
    args=[]
    for i in range(len(new_source)):
        # f1 = ar_source[i%ar_source_frames, :, :, :]
        # f2 = book[i, :, :, :]
        # f2 = np.squeeze(f2)
        # ar_source_crop = f1[:, int((ar_source_width-book_width)/2):int((ar_source_width+book_width)/2)]
        args.append([new_source[i], cv_cover, book[i]])
    p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    ar = p.map(homography_warp, args)
    p.close()
    p.join()  
    ar = np.array(ar)
    writer = cv2.VideoWriter('./result/ar.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (ar.shape[2], ar.shape[1]))
    for i, f in enumerate(ar):
        writer.write(f)

    writer.release()  
    # cv2.imshow('c', ar_source_crop)
    # cv2.imshow('f', composite_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    