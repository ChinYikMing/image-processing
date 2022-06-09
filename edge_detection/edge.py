import cv2 as cv
import numpy as np
import sys, getopt, os, copy, math
import basis as bs
import sobel
from matplotlib import pyplot as plt

def usage(script_name):
    print("Usage: python %s  -i image_path " % script_name)
    exit()

if __name__ == '__main__':
    script_name = sys.argv[0]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:i:h")
    except getopt.GetoptError as err:
        print(err)
        print("\n")
        usage(script_name)

    img_file = None
    for opt, arg in opts:
        if opt in ['-i']:
            if not bs.is_img_file(arg):
                print("Invalid image file")
                exit()
            img_file = arg
        elif opt in ['-h']:
            usage(script_name)

    if img_file == None:
        print("Required an image file")
        usage(script_name)

    sobel.Sobel(cv.imread(img_file))
    exit()