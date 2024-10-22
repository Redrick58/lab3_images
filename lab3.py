# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:06:33 2024

@author: User
"""

import sys
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

def pprint_image(image, title = 'Заяц'):
    gs = plt.GridSpec(2, 2)
    plt.figure(figsize=(8, 10))
    plt.subplot(gs[0])
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.show()

def unsharp_mask(image_in, sigma=3.2, strength=1.3):
    
    # print(sigma)
    image = image_in.copy()
    
    # Apply Gaussian blur
    blurred = cv.GaussianBlur(image, (0, 0), sigma)
    # Subtract the blurred image from the original
    sharpened = cv.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    # plt.figure(figsize=(8, 10))
    # plt.imshow(sharpened, cmap = 'gray')
    return sharpened

def sharpen(image_in):
    
    
    image = image_in.copy() 
    image = cv.medianBlur(image, 3)
    kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
   
    image = cv.filter2D(image, -1, kernel1)
    
    return image

def eq(image):
    
    channels = [0]
    histSize = [256]
    qqq = [0, 256]
    
    hist1 = cv.calcHist([image], channels, None, histSize, qqq)
    
    lut = np.zeros([256, 1]) 
    
    hsum = hist1.sum()
    for i in range(256):
        lut[i] = np.uint8(255 * hist1[:i].sum()/hsum)
        
    image2 = image.copy()
        
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image2[i][j] = lut[image2[i][j]]
            
    return image2

def deconvolve(image):
    
    psf = np.ones((4, 4)) / 16
    deconvolved = restoration.wiener(image, psf, 1, clip=False)
    return deconvolved

# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]
image1 = cv.imread('./krishna.jpg')
image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
pprint_image(image1, 'Исходное')

r, g, b = cv.split(image1)
image1 = cv.merge([255-r, 255-g, 255-b])
pprint_image(image1, 'Переставить каналы')

# image1 = unsharp_mask(image1);
# pprint_image(image1, "Нерезкая маска")

# s = 0.1
# while s < 10:
#     image_cand = unsharp_mask(image1, sigma = s);
#     #plt.figure(figsize=(8, 10))
#     #plt.imshow(image_cand, cmap = 'gray')
#     pprint_image(image_cand, str(s))
#     s += 0.1

image1 = unsharp_mask(image1, sigma = 4.3);
pprint_image(image1, "Нерезкая маска")
image1 = sharpen(image1);
pprint_image(image1, "Увеличение резкости")





