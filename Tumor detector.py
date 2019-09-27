import cv2
import sys
import SimpleITK as sitk
from skimage import io, color, util, draw
import  matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import feature
import numpy as np

sift = cv2.xfeatures2d.SIFT_create()
path = 'T2mni_biasfieldcorr_noskull.nii'
img = sitk.ReadImage(path, sitk.sitkUInt8)
img1= sitk.ReadImage(path, sitk.sitkUInt16)
im = sitk.GetArrayViewFromImage(img)
im1=sitk.GetArrayViewFromImage(img1)
##print(im.shape)
fig = plt.figure("Neural image")
slice_no = 90
Z   = []
imgList = []
plt.subplot(2,2,1)
for i in range(180):
    Z.append(im1[i, ...])
    imgList.append([plt.imshow(Z[i])])
ani = animation.ArtistAnimation(fig, imgList, interval=20, blit=True,repeat_delay=0)

#plt.imshow(Z[0])
##plt.show()

img_slice=im[slice_no, ...]
inputImg=im[slice_no, ...]

edges = feature.canny(img_slice, sigma=2.0)
print(im.shape)
plt.subplot(2,2,2)
plt.imshow(edges, cmap=plt.cm.gray)

##plt.subplot(2,2,3)
##plt.imshow(edges, cmap=plt.cm.gray)
####plt.imshow(Z[1])
##plt.show()

def very_close(a, b, tol = 4.0):
    """Checks if the points a, b are within
    tol distance of each other."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) < tol

def S(si, sj, sigma=1):
    """Computes the 'S' function mentioned in
    the research paper."""
    q = ((-abs(si-sj)) / (sigma*(si+sj)))
    return np.exp(q**2)

def reisfeld(phi, phj, theta):
    return 1-np.cos(phi + phj - 2*theta)

def midpoint(i, j):
    return (i[0]+j[0])/2, (i[1]+j[1])/2

def angle_with_x_axis(i, j):
    x, y = i[0]-j[0], i[1]-j[1]
    if x == 0:
        return np.pi/2
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle


def superm2(image):
    """Performs the symmetry detection on image.
    Somewhat clunky at the moment -- first you 
    must comment out the last two lines: the 
    call to `draw` and `cv2.imshow` and uncomment
    `hex` call. This will show a 3d histogram, where
    bright orange/red is the maximum (most voted for
    line of symmetry). Manually get the coordinates,
    and re-run but this time uncomment draw/imshow."""
    mimage = np.fliplr(image)
    
##    plt.imshow(mimage)
##    plt.show()
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(mimage, None)
    for p, mp in zip(kp1, kp2):
        p.angle = np.deg2rad(p.angle)
        mp.angle = np.deg2rad(mp.angle)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    houghr = np.zeros(len(matches))
    houghth = np.zeros(len(matches))
    weights = np.zeros(len(matches))
    i = 0
    good = []
    for match, match2 in matches:
        point = kp1[match.queryIdx]
        mirpoint = kp2[match.trainIdx]
        mirpoint2 = kp2[match2.trainIdx]
        mirpoint2.angle = np.pi - mirpoint2.angle
        mirpoint.angle = np.pi - mirpoint.angle
        if mirpoint.angle < 0.0:
            mirpoint.angle += 2*np.pi
        if mirpoint2.angle < 0.0:
            mirpoint2.angle += 2*np.pi
        mirpoint.pt = (mimage.shape[1]-mirpoint.pt[0], mirpoint.pt[1])
        if very_close(point.pt, mirpoint.pt):
            mirpoint = mirpoint2
            good.append(match2)
        else:
            good.append(match)
        theta = angle_with_x_axis(point.pt, mirpoint.pt)
        xc, yc = midpoint(point.pt, mirpoint.pt) 
        r = xc*np.cos(theta) + yc*np.sin(theta)
        Mij = reisfeld(point.angle, mirpoint.angle, theta)*S(point.size, mirpoint.size)
        houghr[i] = r
        houghth[i] = theta

        weights[i] = Mij
        i += 1
    matches = sorted(matches, key = lambda x:x[0].distance)
    good = sorted(good, key = lambda x: x.distance)
    
    def draw(r, theta):
        image1= image.copy()
        if np.pi/4 < theta < 3*(np.pi/4):
            for x in range(len(image1.T)):
                y = int((r-x*np.cos(theta))/np.sin(theta))
                if 0 <= y < len(image1.T[x]):
                    image1[y][x] = 255
        else:
            for y in range(len(image1)):
                x = int((r-y*np.sin(theta))/np.cos(theta))
                if 0 <= x < len(image1[y]):
                    image1[y][x] = 255
        plt.subplot(2,2,4)
        plt.imshow(image1)
##        plt.show()
    img3 = cv2.drawMatches(image, kp1, mimage, kp2, good[:15], None, flags=2)

    def hex():
        plt.subplot(2,2,3)
        plt.hexbin(houghr, houghth, bins=200)
##        plt.show()
    hex()
    draw(2.8, 2.4)
##    cv2.imshow('First image',img3); cv2.waitKey(0);
    
##    cv2.imshow('Second image', image); cv2.waitKey(0);
superm2(inputImg)
plt.show()
