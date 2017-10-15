'''
Created on Dec 10, 2015

@author: rawas
'''
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def CorrsWithEdgeNC(slit0, slit1, windowSize, lagDis):
    '''input two 1D slices from stereo images and correlates 
    using block matching with blocks sized windowSize'''
    if (len(slit0.shape) != 1 or len(slit1.shape)!= 1):
        raise Exception("array1 and array2 are not 1D")
    
    if (len(slit0.shape) != len(slit1.shape)):
        raise Exception("array1 and array2 are not the same length")
    
    if (windowSize % 2 == 0):
        raise Exception("windowSize must be odd")
    
    length = len(slit0) #lag length
    
    sideBar = (windowSize) / 2 #side bar to skip
    
    lag  = xrange(0-lagDis,lagDis+1)
    lagLength = len(lag)
    
    slitCorrelations = []
    firstLag = lag[0]
    
    for x in xrange(sideBar + (lagLength / 2), length - sideBar - (lagLength / 2)):
        box0 = slit0[(x - sideBar) : (x + sideBar)]
        
        correl = np.zeros(lagLength)
        for l in lag:
            box1 = slit1[(x - sideBar + l) : (x + sideBar + l)]
            
            sum11 = np.inner(box0,box0) #finding cross correlation
            sum22 = np.inner(box1,box1)
            sum12 = np.inner(box0,box1)
#             corr = sum((box0-box1)**2)      
            
            corr = sum12 / (sum11*sum22)**0.5 #make correlation
            correl[l-firstLag] = corr 
        
	maxDis = np.argmax(correl)
        slitCorrelations.extend([255*(lagDis-maxDis)/float(lagDis)])
    return(slitCorrelations)
  
#image names
imnames = "d"

#get the images
im0 = scipy.misc.imread("data/"+imnames+"0.png", 1)
im1 = scipy.misc.imread("data/"+imnames+"1.png", 1)
print("Read images...")

#resize the images to make the code faster.
im0_small = scipy.misc.imresize(im0, 0.4, interp='bicubic', mode=None).astype(float)
im1_small = scipy.misc.imresize(im1, 0.4, interp='bicubic', mode=None).astype(float)
#im0_small = im0
#im1_small = im1
print("Resized images...")
#find the means of the images, subtracting these gives better depthmaps.
mean0 = np.mean(im0_small)
mean1 = np.mean(im1_small)

#subtract the means that we found.
im0_small -= mean0
im1_small -= mean1
print("Computed means...")

imageLoopData = range(0,im0_small.shape[0])
# progress = len(imageLoopData)-1
# image1 = []
lag = 25 # Change this depending on the size.
winsize = 21 # this is the size of the window. It must be odd to handle edge effects.  

print("\nStarting NC Depthmap...")
progress = len(imageLoopData)-1
image = []
t1 = time.clock()
#repeat the program for each line of the images
for y in imageLoopData:
    #get a slit from both images
    im0 = im0_small[y,:] #all of x, 500 down y
    im1 = im1_small[y,:]     
    image.extend([CorrsWithEdgeNC(im0, im1, winsize, lag)])
    percent = round((float(y) / progress)*100, 2)
    sys.stdout.write("\r%d%%" % percent)
    sys.stdout.flush()
    
t2 = time.clock()-t1

print("time took "+str(t2))
scipy.misc.imsave('res/'+imnames+'-small-nothread.png', image)

