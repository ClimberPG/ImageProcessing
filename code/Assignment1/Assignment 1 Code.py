#!/usr/bin/env python
# coding: utf-8

# Assignment 1
# Jian Zhang, student ID: 219012058
# Date: 19.9.29


# import necessary library
import numpy as np
import math
import cv2 # use opencv to read image
import matplotlib.pyplot as plt # use matplotlib to show image


# Exercise 1: Spatial and Intensity Resolution

# read gray image
img_face = cv2.imread('./face.png', 0)
img_cameraman = cv2.imread('./cameraman.png', 0)
img_crowd = cv2.imread('./crowd.png', 0)

# 1.(a)
# make a program to downsample an image
# implement a function to recover the original size by interpolation

# 1.(b)
# make a program to reduce the quantization level of an image
# compare the results with(a)


# to downsample an image, input "img", "downsample rate"
# parameter:  image, downsample rate
# return:  downsampled image
def downsample(img, dsrate):
    
    row = img.shape[0]
    col = img.shape[1]
    drow, dcol = math.floor(row / dsrate), math.floor(col / dsrate)
    
    # generate a blank downsampled image
    new_img = np.zeros((drow, dcol), int)
    
    p = q = 0
    for i in range(0, drow * dsrate, dsrate):
        for j in range(0, dcol * dsrate, dsrate):
            new_img[p][q] = img[i][j]
            q += 1
        q = 0
        p += 1
    
    return new_img

# to downsample an image, input "img", "downsample rate"
# parameter:  image, downsample rate
# return:  downsampled image
# a better way to downsample
def downsample2(img, dsrate):
    
    return img[::dsrate, ::dsrate]

# to resample an image, bilinear interpolation, using for loop
# parameter: image, resample rate
# return: resampled image
def resample(img, rsrate):
    
    row = img.shape[0]
    col = img.shape[1]
    
    rerow, recol = row * rsrate, row * rsrate
    
    # generate a blank upsampled image
    new_img = np.zeros((rerow, recol), int)
    
    for i in range(rerow):
        for j in range(recol):
            # calculate the origin coordinates
            ori_x = (i + 0.5) / rsrate - 0.5
            ori_y = (j + 0.5) / rsrate - 0.5
            
            Q11 = (math.floor(ori_x), math.floor(ori_y))
            Q12 = (Q11[0], min(Q11[1] + 1, col - 1))
            Q21 = (min(Q11[0] + 1, row - 1), Q11[1])
            Q22 = (min(Q11[0] + 1, row - 1), min(Q11[1] + 1, col - 1)) # use min() to avoid index error
            
            # x direction interpolate
            f_R1 = (Q21[0] - ori_x) * img[Q11[0], Q11[1]] + (ori_x - Q11[0]) * img[Q21[0], Q21[1]]
            f_R2 = (Q21[0] - ori_x) * img[Q12[0], Q12[1]] + (ori_x - Q11[0]) * img[Q22[0], Q22[1]]
            
            # combine x and y interpolation
            new_img[i, j] = f_R1 * (Q12[1] - ori_y) + f_R2 * (ori_y - Q11[1])
    
    return new_img


# to resample an image, bilinear interpolation, using numpy
# parameter: image, resample rate
# return: resampled image
# Notes: this function is much faster than using for loop
def bilinear_interpolate(im, rsrate):
    
    ori_x = []
    ori_y = []
    row = im.shape[0]
    col = im.shape[1]
    rerow, recol = row * rsrate, row * rsrate
    for i in range(rerow):
        for j in range(recol):
            # calculate the origin coordinates
            ori_x.append((i + 0.5) / rsrate - 0.5)
            ori_y.append((j + 0.5) / rsrate - 0.5)
            
    x = np.asarray(ori_x)
    y = np.asarray(ori_y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    A = wa*Ia + wb*Ib + wc*Ic + wd*Id
    new_img = np.reshape(A, (rerow, recol), order='F')

    return new_img


def quantization(img, rate):
    img = img // rate
    return img


# apply different rates to 3 images
# downsampe to 1/2, 1/4 and 1/8, then resample them by bilinear interpolation


img_face_downbytwo = downsample(img_face, 2)
img_face_rebytwo   = bilinear_interpolate(img_face_downbytwo, 2)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_face_downbytwo,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_face_rebytwo,cmap="gray")
plt.axis('off')
plt.show()


img_cameraman_downbyfour = downsample(img_cameraman, 4)
img_cameraman_rebyfour   = bilinear_interpolate(img_cameraman_downbyfour, 4)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_cameraman_downbyfour,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_cameraman_rebyfour,cmap="gray")
plt.axis('off')
plt.show()


img_crowd_downbyeight = downsample(img_crowd, 8)
img_crowd_rebyeight   = bilinear_interpolate(img_crowd_downbyeight, 8)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_crowd_downbyeight,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_crowd_rebyeight,cmap="gray")
plt.axis('off')
plt.show()


# compare downsample results and reduce quantization level results

img_face_quantization = quantization(img_face, 2)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_face_downbytwo,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_face_quantization,cmap="gray")
plt.axis('off')
plt.show()


img_camera_quantization = quantization(img_cameraman, 4)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_cameraman_downbyfour,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_camera_quantization,cmap="gray")
plt.axis('off')
plt.show()


img_crowd_quantization = quantization(img_crowd, 8)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_crowd_downbyeight,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_crowd_quantization,cmap="gray")
plt.axis('off')
plt.show()


# Exercise 2: Image Enhance and Denoising

# 2.(a) Histogram Equalization
# Implement the histogram equalization technique, used image 'Histogram_Equalization.png'

# parameter: image to be equalized, type numpy.ndarray
# return: equalized image, type numpy.ndarray
def hist_equalize(img):
    
    hist = []
    for i in range(256):
        hist.append(0)
    row, col  = img.shape
    new_img = np.zeros((row, col))
    
    # count the total number for each pixel value
    for i in range(row):
        for j in range(col):
            hist[img[i, j]] += 1
            
    # normalization
    for i in range(256):
        hist[i] /= row * col
    
    # calculate accumulation
    for i in range(1, 256):
        hist[i] += hist[i - 1]
    
    # equalization
    for i in range(256):
        hist[i] = (np.uint8)(255 * hist[i] + 0.5)
    
    for i in range(row):
        for j in range(col):
            new_img[i, j] = hist[img[i, j]]
    
    return new_img


img_hist = cv2.imread('./Histogram_Equalization.png', 0)
img_hist_equalized = hist_equalize(img_hist)


# plot the results, compare the original image and equalized image
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_hist,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_hist_equalized,cmap="gray")
plt.axis('off')
plt.show()


# 2.(b)
# implement the sharpening spatial filters, used image 'Spatial_Filtering.png'

def spatial_filter(img):
    
    row, col = img.shape
    img_pad = np.pad(img, ((1,1), (1,1)), 'edge')
    img_laplacian = np.zeros((row, col))

    laplacian = np.array([[0,1,0], [1,-4,1], [0,1,0]])
    #laplacian = np.array([[1,1,1], [1,-8,1], [1,1,1]])
    
    for i in range(row - 2):
        for j in range(col - 2):
            img_laplacian[i, j] = np.sum(img_pad[i:i+3, j:j+3] * laplacian)
    
    new_img = img - img_laplacian
    new_img = np.clip(new_img, 0, 255)
    
#     gmin = np.min(img_laplacian)
#     img_laplacian = img_laplacian - gmin
#     gmax = np.max(img_laplacian)
#     img_laplacian = img_laplacian / gmax * 255.0
#     img_laplacian = np.uint8(img_laplacian + 0.5)
    
    return new_img


img_before_filtering = cv2.imread('./Spatial_Filtering.png', 0)
img_spatial_filtered = spatial_filter(img_before_filtering)


# plot the results, compare the original image and spacial filtered image
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_before_filtering,cmap="gray")
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img_spatial_filtered,cmap="gray")
plt.axis('off')
plt.show()


# 2.(c)
# Apply frequency filtering
# Denoise the given image 'Frequency_Filtering.png'

img_freq = cv2.imread('./Frequency_Filtering.png', 0)
img_freq_gaussian = cv2.GaussianBlur(img_freq, (7,7), 0)


plt.imshow(img_freq,cmap="gray")
plt.axis('off')
plt.show()


plt.imshow(img_freq_gaussian,cmap="gray")
plt.axis('off')
plt.show()
