"""
This files contains helpers function for the machine learning project on road segmentation.
"""

import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image

import code

import numpy as np


def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h, c=0):
    """ Personalized version of the img_crop incorporating the option of getting the context (c) around the patches"""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    
    # padding the image to access the context of border patches
    if is_2d:
        pad_im = np.pad(im,((c,c),(c,c)), 'reflect')
    else:
        pad_im = np.pad(im,((c,c),(c,c),(0,0)), 'reflect')

    # cropping the image
    for i in range(c,imgheight+c,h):
        for j in range(c,imgwidth+c,w):
            if is_2d:
                im_patch = pad_im[j-c:(j+w)+c, i-c:(i+h)+c]
            else:
                im_patch = pad_im[j-c:(j+w)+c, i-c:(i+h)+c, :]

            list_patches.append(im_patch)
    return list_patches