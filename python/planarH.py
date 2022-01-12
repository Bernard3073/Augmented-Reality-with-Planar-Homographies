import numpy as np
import cv2
import copy

# def order_points(pts):
#     # Reference: https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
#     # the order of the list: top-left -> top->right -> bottom->right -> bottom->left
#     ls = np.zeros((4,2),dtype="float32")
#     s = pts.sum(axis=1)
#     ls[0] = pts[np.argmin(s)]
#     ls[2] = pts[np.argmax(s)]
    
#     diff = np.diff(pts, axis=1)
#     ls[1] = pts[np.argmin(diff)]
#     ls[3] = pts[np.argmax(diff)]
#     return ls 

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	A = []
	# x2 = order_points(x2)

	for i in range(len(x1)):
		x2_1, y2_2 = x2[i][0], x2[i][1]
		x1_1, y1_2 = x1[i][0], x1[i][1]
		A.append([x2_1, y2_2, 1, 0, 0, 0, -x1_1 * x2_1, -x1_1 * y2_2, -x1_1])
		A.append([0, 0 , 0, x2_1, y2_2, 1, -y1_2 * x2_1, -y1_2 * y2_2, -y1_2])
    
	A = np.array(A)
	U, D, V_t = np.linalg.svd(A)
    # the solution will be the last column
    # (the eigenvector corresponding to the smallest eigenvalue) of the orthonormal matrix 
	# normalize by dividing by the element at (3,3) 
	# h = V_t[-1, :] / V_t[-1, -1]
	H2to1 = np.reshape(V_t[-1, :], (3, 3))

	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	n = len(x1)
	x1_cent_x, x1_cent_y = np.sum(x1[:, 0])/n, np.sum(x1[:, 1])/n
	x2_cent_x, x2_cent_y = np.sum(x2[:, 0])/n, np.sum(x2[:, 1])/n

	#Shift the origin of the points to the centroid
	x1_norm = x1 - [x1_cent_x, x1_cent_y]
	x2_norm = x2 - [x2_cent_x, x2_cent_y]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	scaling_factor_1 = np.sqrt(2) / np.max([i[0]**2 + i[1]**2 for i in x1_norm])
	scaling_factor_2 = np.sqrt(2) / np.max([i[0]**2 + i[1]**2 for i in x2_norm])
	x1_norm = x1_norm * scaling_factor_1
	x2_norm = x2_norm * scaling_factor_2

	#Similarity transform 1
	# normalization in matrix form
	T1 = np.array([[scaling_factor_1, 0, -scaling_factor_1 * x1_cent_x],
		[0, scaling_factor_1, -scaling_factor_1 * x1_cent_y],
		[0, 0, 1]])

	#Similarity transform 2
	T2 = np.array([[scaling_factor_2, 0, -scaling_factor_2 * x2_cent_x],
		[0, scaling_factor_2, -scaling_factor_2 * x2_cent_y],
		[0, 0, 1]])

	#Compute homography
	H2to1 = computeH(x1_norm, x2_norm)

	#Denormalization
	# H = inv(T1) * H_til * T2
	H2to1 = np.linalg.inv(T1) @ H2to1 @ T2

	return H2to1


def generateRandom(src_Pts, dest_Pts):
    r = np.random.choice(len(src_Pts), 4)
    src = [src_Pts[i] for i in r]
    dest = [dest_Pts[i] for i in r]
    return np.asarray(src, dtype=np.float32), np.asarray(dest, dtype=np.float32)


def computeH_ransac(locs1, locs2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	# https://stackoverflow.com/questions/61146241/how-to-stitch-two-images-using-homography-matrix-in-opencv
	
	# Swap columns in locs because they are in the form of [y, x] returned by matchPics
	locs1 = np.fliplr(locs1)
	locs2 = np.fliplr(locs2)
	
	N = np.inf
	sample_count = 0
	p = 0.99
	threshold = 10
	max_num_inliers = 0
	while N > sample_count:
		random_pts_1, random_pts_2 = generateRandom(locs1, locs2)
		H2to1 = computeH_norm(random_pts_1, random_pts_2)
		inlier_count = 0
		inliers = []
		for p1, p2 in zip(locs1, locs2):
			p2_homo = (np.append(p2, 1)).reshape(3, 1)
			p1_est = H2to1 @ p2_homo
			p1_est = (p1_est/p1_est[2])[:2].reshape(1, 2)
			if cv2.norm(p1 - p1_est) <= threshold:
				inlier_count += 1
				inliers.append(1)
			else:
				inliers.append(0)
		if inlier_count > max_num_inliers:
			max_num_inliers = inlier_count
			bestH2to1 = H2to1
			inliers.append(p1)

		inlier_ratio = inlier_count / len(locs2)
		if np.log(1 - (inlier_ratio**8)) == 0:
			continue
		N = np.log(1-p) / np.log(1 - (inlier_ratio**8))
		sample_count += 1

	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	H2to1_inv = np.linalg.pinv(H2to1)

	#Create mask of same size as template
	mask = np.ones(template.shape)
	#Warp mask by appropriate homography
	mask_warp = cv2.warpPerspective(mask, H2to1_inv, (img.shape[1], img.shape[0]))
	#Warp template by appropriate homography
	template_warp = cv2.warpPerspective(template, H2to1_inv, (img.shape[1], img.shape[0]))
	#Use mask to combine the warped template and the image
	composite_img = template_warp + img * np.logical_not(mask_warp)
	return composite_img


