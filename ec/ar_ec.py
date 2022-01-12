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
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, 
                    key_size = 12,     
                    multi_probe_level = 1)
    search_params = dict(checks = 100)  # increase checks for better precision
    matches = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matches.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    # for m, n in matches:
    #     if m.distance < 0.8 * n.distance:
    #         good.append(m)
    for m in matches:
        if len(m) < 2:  # bad matches
            continue
        if m[0].distance < 0.6 * m[1].distance:
            good.append(m[0])
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
    frame0 = ar_source[0]
    x, y, w, h = cropBlackBar(frame0)
    frame0 = frame0[y:y+h, x:x+w]
    H, W = frame0.shape[:2]
    width = cv_cover.shape[1] * H / cv_cover.shape[0]
    wStart, wEnd = np.round([W/2 - width/2, W/2 + width/2]).astype(int)
    ar_source_crop = np.array([f[y:y+h, x:x+w][:, wStart:wEnd] for f in ar_source])

    cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)

    # orb = cv2.ORB_create()
    # FLANN_INDEX_LSH = 6
    # index_params= dict(algorithm = FLANN_INDEX_LSH,
    #                 table_number = 6, 
    #                 key_size = 12,     
    #                 multi_probe_level = 1)
    # search_params = dict(checks = 100)  # increase checks for better precision
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # cv_cover_gray = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
    # kp1, des1 = orb.detectAndCompute(cv_cover_gray, None)

    for i in range(len(ar_source_crop)):
        cv_book = book[i]
        cv_book_gray = cv2.cvtColor(cv_book, cv2.COLOR_BGR2GRAY)
        kp1, kp2, good = detect_feature(cv_cover_gray, cv_book_gray)
        
        # kp2, des2 = orb.detectAndCompute(cv_book_gray, None)
        # matches = matcher.knnMatch(des1, des2, k=2)
        # # Apply ratio test
        # good = []
        # for m in matches:
        #     if len(m) < 2:  # bad matches
        #         continue
        #     if m[0].distance < 0.6 * m[1].distance:
        #         good.append(m[0])

        feature_1 = []
        feature_2 = []

        # for i, match in enumerate(good):
        for match in good:
            feature_1.append([kp1[match.queryIdx].pt])
            feature_2.append([kp2[match.trainIdx].pt])
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        if len(good) >= 4:
            H, _ = cv2.findHomography(feature_1, feature_2, cv2.FM_RANSAC)
            # locs1 = np.array([[*kp1[m.queryIdx].pt] for m in good])
            # locs2 = np.array([[*kp2[m.trainIdx].pt] for m in good])
            # H, _ = cv2.findHomography(locs1, locs2, cv2.RANSAC, 0.7)
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