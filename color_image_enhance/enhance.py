import cv2 as cv
import numpy as np
import sys, getopt, os, copy, math
import histequal as he
from matplotlib import pyplot as plt
import brightness_contrast as bc

def usage(script_name):
    print("Usage: python %s -c color_space_for_enhancement -i image_path " % script_name)
    print("       valid alpha range: [1 - 500], default is 100 if not specified")
    print("       valid beta range: [0 - 200], default is 100 if not specified")
    exit()

def is_img_file(path):
    return os.path.isfile(path) and \
            (path.endswith(".tif") or \
            path.endswith(".png") or \
            path.endswith(".bmp") or \
            path.endswith(".jpeg") or path.endswith(".jpg"))

def valid_color_space(cs):
    if cs == "rgb" or cs == "hsi" or cs == "lab":
        return True
    return False

def valid_alpha_range(alpha):
    alpha = int(alpha)
    if(alpha < 1 or alpha > 500):
        return False
    return True

def valid_beta_range(beta):
    beta = int(beta)
    if(beta < 0 or alpha > 200):
        return False
    return True

if __name__ == '__main__':
    script_name = sys.argv[0]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:i:k:c:a:b:h")
    except getopt.GetoptError as err:
        print(err)
        print("\n")
        usage(script_name)

    img_file = None
    color_space = None
    has_k = None
    alpha = 100
    beta = 100
    for opt, arg in opts:
        if opt in ['-i']:
            if not is_img_file(arg):
                print("Invalid image file")
                exit()
            img_file = arg
        elif opt in ['-c']:
            if not valid_color_space(arg):
                print("Invalid color space. Check valid color space in usage(python %s -h)" % (script_name))
                exit()
            color_space = arg
        elif opt in ['-a']:
            if not valid_alpha_range(arg):
                print("Invalid alpha range. Check valid alpha range in usage(python %s -h)" % (script_name))
                exit()
            alpha = int(arg)
        elif opt in ['-b']:
            if not valid_beta_range(arg):
                print("Invalid beta range. Check valid beta range in usage(python %s -h)" % (script_name))
                exit()
            beta = int(arg)
        elif opt in ['-h']:
            usage(script_name)

    if color_space == None:
        print("Required a color space for color image enhancement")
        usage(script_name)

    if img_file == None:
        print("Required an image file")
        usage(script_name)

    image = cv.imread(img_file)
    if(color_space == "rgb" or color_space == "hsi"):
        print("Enhancing %s with histogram equalization method in %s color space" % (img_file, color_space))
        res = he.histequal(image, color_space)
    else:
        print("Enhancing %s with brightness contrast method with alpha= %d and beta= %d in L*a*b* color space" % (img_file, alpha,  beta))
        res = bc.brightness_contrast(image, alpha, beta)

    img_stack = np.hstack((image, res))
    cv.imshow("Left is original, Right is processed", img_stack)
    cv.waitKey(0)
    cv.destroyAllWindows()
    exit()