import cv2 as cv
import numpy as np
import sys, getopt, os, copy, math

def pad_img(img):       
    img_pad = np.asarray([[ 0 for x in range(0,img.shape[1] + 2)] for y in range(0,img.shape[0] + 2)], dtype =np.uint8)
    img_pad[1:(img_pad.shape[0]-1), 1:(img_pad.shape[1]-1)] = img 
    return img_pad 

def is_img_file(path):
    return os.path.isfile(path) and \
            (path.endswith(".tif") or \
            path.endswith(".png") or \
            path.endswith(".bmp") or \
            path.endswith(".jpeg") or path.endswith(".jpg"))
