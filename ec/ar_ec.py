import numpy as np
import cv2
#Import necessary functions
import loadVid
from pathlib import Path


#Write script for Q4.1x
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

def detect_feature(img1, img2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    matches = flann.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    return kp1, kp2, good

def main():
    ar_source = Path('./python/ar_source.npy')
    if ar_source.exists():
        ar_source = np.load('./python/ar_source.npy')
    else:
        ar_source = loadVid('./data/ar_source.mov')    # 511 frames 360x640
        np.save('./python/ar_source.npy', ar_source)
    book = Path('./python/book.npy')
    if book.exists():
        book = np.load('./python/book.npy')
    else:
        book = loadVid('./data/book.mov')           # 641 frames 480x640
        np.save('./python/book.npy', book)
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
    ar_source_crop = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])
    cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
    for i in range(len(ar_source_crop)):
        cv_book = book[i]
        cv_book_gray = cv2.cvtColor(cv_book, cv2.COLOR_BGR2GRAY)
        kp1, kp2, good = detect_feature(cv_cover_gray, cv_book_gray)
        feature_1 = []
        feature_2 = []

        for i, match in enumerate(good):
            feature_1.append(kp1[match.queryIdx].pt)
            feature_2.append(kp2[match.trainIdx].pt)
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        H, _ = cv2.findHomography(feature_1, feature_2, cv2.FM_RANSAC)
        # print(H)
        template = cv2.resize(ar_source_crop[i], (cv_cover.shape[1], cv_cover.shape[0]))
        #Create mask of same size as template
        mask = np.ones(template.shape)
        #Warp mask by appropriate homography
        mask_warp = cv2.warpPerspective(mask, H, (cv_book.shape[1], cv_book.shape[0]))
        #Warp template by appropriate homography
        template_warp = cv2.warpPerspective(template, H, (cv_book.shape[1], cv_book.shape[0]))
        #Use mask to combine the warped template and the image
        composite_img = template_warp + cv_book * np.logical_not(mask_warp)
        cv2.imshow('c', composite_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()