import numpy as np
import cv2

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
	scaling_factor_1 = np.sqrt(2) / np.max(np.hypot(x1_norm))
	scaling_factor_2 = np.sqrt(2) / np.max(np.hypot(x2_norm))
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




def computeH_ransac(locs1, locs2):
	#Q3.8
	#Compute the best fitting homography given a list of matching points



	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	return composite_img


