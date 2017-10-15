#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Dec 10, 2015

@author: Robert Washbourne
'''

import scipy.misc
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import multiprocessing
import cProfile


def CorrsWithEdgeNCLeft(y):
    global image
    global done

    windowSize = 21  # Change this depending on the size.
    lagDis = 25  # this is the size of the window. It must be odd to handle edge effects.

    # get a slit from both images

    slit0 = im0_small[y, :]
    slit1 = im1_small[y, :]

    if len(slit0.shape) != 1 or len(slit1.shape) != 1:
        raise Exception('array1 and array2 are not 1D')

    if len(slit0.shape) != len(slit1.shape):
        raise Exception('array1 and array2 are not the same length')

    if lagDis % 2 == 0:
        raise Exception('lag must be odd')

    length = len(slit0)  # lag length

    sideBar = windowSize / 2  # side bar to skip

    lag = xrange(0 - lagDis, lagDis + 1)  # lagdis * 2
    #lag = xrange(0 - lagDis, 0)
    lagLength = len(lag)

    slitCorrelations = []
    firstLag = lag[0]

    for x in xrange(sideBar + lagLength / 2, length - sideBar
                    - lagLength / 2):
        box0 = slit0[x - sideBar:x + sideBar]

        correl = np.zeros(lagLength)
        for l in lag:
            box1 = slit1[x - sideBar + l:x + sideBar + l]

            sum11 = np.inner(box0, box0)  # finding cross correlation
            sum22 = np.inner(box1, box1)
            sum12 = np.inner(box0, box1)

            corr = sum12 / (sum11 * sum22) ** 0.5
            correl[l - firstLag] = corr

        maxDis = np.argmax(correl)
        slitCorrelations.extend([255 * (lagDis*2 - maxDis)
                                / float(lagDis*2)])

    return slitCorrelations


def CorrsWithEdgeNCRight(y):
    global image
    global done

    windowSize = 21  # Change this depending on the size.
    lagDis = 25  # this is the size of the window. It must be odd to handle edge effects.

    # get a slit from both images

    slit0 = im1_small[y, :]
    slit1 = im0_small[y, :]

    if len(slit0.shape) != 1 or len(slit1.shape) != 1:
        raise Exception('array1 and array2 are not 1D')

    if len(slit0.shape) != len(slit1.shape):
        raise Exception('array1 and array2 are not the same length')

    if lagDis % 2 == 0:
        raise Exception('lag must be odd')

    length = len(slit0)  # lag length

    sideBar = windowSize / 2  # side bar to skip

    lag = xrange(0 - lagDis, lagDis + 1)  # lagdis * 2
    #lag = xrange(0, lagDis + 1)
    lagLength = len(lag)

    slitCorrelations = []
    firstLag = lag[0]

    for x in xrange(sideBar + lagLength / 2, length - sideBar
                    - lagLength / 2):
        box0 = slit0[x - sideBar:x + sideBar]

        correl = np.zeros(lagLength)
        for l in lag:
            box1 = slit1[x - sideBar + l:x + sideBar + l]

            sum11 = np.inner(box0, box0)  # finding cross correlation
            sum22 = np.inner(box1, box1)
            sum12 = np.inner(box0, box1)

            corr = sum12 / (sum11 * sum22) ** 0.5
            correl[l - firstLag] = corr

        maxDis = np.argmax(correl)
        slitCorrelations.extend([255 * (lagDis*2 - maxDis)
                                / float(lagDis*2)])

    return slitCorrelations


def finished1(res):

    # scipy.misc.imsave('res/'+imnames+'-small-thread.png', results)

    # Make the Pool of workers

    pool2 = multiprocessing.Pool(12)
    results2 = pool2.map(CorrsWithEdgeNCRight, imageLoopData)

    finished2(res, results2)

    # close the pool and wait for the work to finish

    pool2.close()
    pool2.join()


def finished2(res, res2):
    plt.subplot(1, 2, 1)
    plt.imshow(res, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(res2, cmap='gray')
    plt.axis('off')
    plt.show()


# image names

imnames = 'd'

# get the images

im0 = scipy.misc.imread('data/' + imnames + '0.png', 1)
im1 = scipy.misc.imread('data/' + imnames + '1.png', 1)
print 'Read images...'

# resize the images to make the code faster.

im0_small = scipy.misc.imresize(im0, 0.4, interp='bicubic',
                                mode=None).astype(float)
im1_small = scipy.misc.imresize(im1, 0.4, interp='bicubic',
                                mode=None).astype(float)

# im0_small = im0
# im1_small = im1

print 'Resized images...'
print np.shape(im0_small)

# find the means of the images, subtracting these gives better depthmaps.

mean0 = np.mean(im0_small)
mean1 = np.mean(im1_small)

# subtract the means that we found.

im0_small -= mean0
im1_small -= mean1
print 'Computed means...'

size = im0_small.shape[0]

imageLoopData = range(0, size)
progress = len(imageLoopData) - 1

# image1 = []

print '\nStarting N2 Depthmap...'

# image = np.empty(im0_small.shape[0], dtype=object);

image = [None] * size

# t1 = time.clock()

# Make the Pool of workers

pool1 = multiprocessing.Pool(12)
results = pool1.map(CorrsWithEdgeNCLeft, imageLoopData)

finished1(results)

# close the pool and wait for the work to finish

pool1.close()
pool1.join()

# plt.imshow(image, cmap = "binary", interpolation='nearest')
# plt.axis('off')
# plt.savefig('res/'+imnames+'.png', bbox_inches='tight')


			
