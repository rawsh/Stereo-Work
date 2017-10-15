import numpy as np
import cv2
import scipy.misc
from matplotlib import pyplot as plt

imgL = cv2.imread('data/d0.png',0)
imgR = cv2.imread('data/d1.png',0)

window_size = 3
min_disp = 16
num_disp = 112-min_disp

stereo = cv2.StereoBM_create(numDisparities=48, blockSize=31)

disparity = stereo.compute(imgL,imgR)
scipy.misc.imsave('res/opencv-bm.png', disparity)
