import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

im0 = scipy.misc.imread("data/d0.png", 1)
im1 = scipy.misc.imread("data/d1.png", 1)

im0color = scipy.misc.imread("data/d0.png")
im1color = scipy.misc.imread("data/d1.png")

im0_small = scipy.misc.imresize(im0, 0.4, interp='bicubic', mode=None).astype(float)
im1_small = scipy.misc.imresize(im1, 0.4, interp='bicubic', mode=None).astype(float)

im0_small_color = scipy.misc.imresize(im0color, 0.4, interp='bicubic', mode=None).astype(float)
im1_small_color = scipy.misc.imresize(im1color, 0.4, interp='bicubic', mode=None).astype(float)

mid = 120
window=60
lag=17

left  =im0_small[220][mid-window:mid+window]
right =im1_small[220][mid-window:mid+window]

leftcolor  =im0_small_color[220][mid-window:mid+window]
rightcolor =im1_small_color[220][mid-window:mid+window]

leftloc = 65
winleft = right[leftloc-lag:leftloc+lag]

corrs = []
corrsimg = []

for x in range(0,window*2):
	if (x >= lag and x < window*2-lag):
		winright = left[x-lag:x+lag]

		sum11 = np.inner(winleft,winleft)
		sum22 = np.inner(winright,winright)
		sum12 = np.inner(winleft,winright)

		corr = sum12 / (sum11*sum22)**0.5

		corrs.append(corr)
		corrsimg.append([1-corr,1-corr,1-corr])
	else:
		corrs.append(0)
		corrsimg.append([1,1,1]);

corrsimg[np.argmax(corrs)] = [1,0,0]

rightcolor[leftloc-lag]=[1,0,0]
rightcolor[leftloc+lag]=[1,0,0]

leftcolor[np.argmax(corrs)-lag]=[1,0,0]
leftcolor[np.argmax(corrs)+lag]=[1,0,0]

plt.imshow([rightcolor,leftcolor,corrsimg], aspect='auto', interpolation="nearest")
plt.axis('off')
#plt.show()
plt.savefig('test.png', bbox_inches='tight')
