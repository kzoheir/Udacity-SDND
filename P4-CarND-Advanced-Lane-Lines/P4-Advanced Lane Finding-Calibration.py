#!/usr/bin/env python
# coding: utf-8

# ## Advanced Lane Finding Project - Camera Calibration
#
# The goals / steps of this project are the following:
#
# * Rubric 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Store the calibration matrix and distortion coefficients

# # Rubric 1: Camera Calibration

# In[1]:


import numpy as np
import cv2
import glob

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from IPython import display
from random import randint
import pickle

# %matplotlib qt: needed for cv2.imshow to work properly. However, it may causes problem when using plt!
#KZOH get_ipython().run_line_magic('matplotlib', 'qt')

# variables for chesboard corners
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')


# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(2)
cv2.destroyAllWindows()

# Perform camera calibration
print ("\nGray image shape: ", gray.shape, gray.shape[::-1], "\tColored image shape", img.shape)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

pickle.dump( { 'mtx': mtx, 'dist': dist }, open('./camera_calibration.p', 'wb'))


# In[ ]:
