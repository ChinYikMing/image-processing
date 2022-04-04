import cv2 as cv
import numpy as np
import sys, getopt, os, copy, math
from matplotlib import pyplot as plt

def usage(script_name):
    print("Usage: python %s -m mode -i image_path [-k coefficient_of_mask]" % script_name)
    print("       valid mode are: 'laplacian' or 'highboost'")
    print("       note: -k option is optional and only available to highboost mode")
    print("       where coefficient_of_mask can be {2,3,4, ...}, default is 3")
    exit()

def is_img_file(path):
    return os.path.isfile(path) and \
            (path.endswith(".tif") or \
            path.endswith(".png") or \
            path.endswith(".bmp") or \
            path.endswith(".jpeg") or path.endswith(".jpg"))

def valid_mode(mode):
    if mode == "laplacian" or mode == "highboost":
        return True
    return False

def show_imgs(img_list):
    for i in range(0, len(img_list)):
        cv.imshow(img_list[i][0], img_list[i][1])
    cv.waitKey(0)
    cv.destroyAllWindows()

def pad_img(img, dim):
    img_row = img.shape[0]
    img_col = img.shape[1]

    dim_row = dim[0]
    dim_col = dim[1]

    top_pad = 0
    bottom_pad = 0
    left_pad = 0
    right_pad = 0

    # find gcd(dim_row or dim_col, 2), so we can equally padding top, bottom and left and right
    top_pad = int(((math.ceil(img_row / (dim_row * 2)) * (dim_row * 2)) - img_row) / 2)
    bottom_pad = int(((math.ceil(img_row / (dim_row * 2)) * (dim_row * 2)) - img_row) / 2)
    left_pad = int(((math.ceil(img_col / (dim_col * 2)) * (dim_col * 2)) - img_col) / 2)
    right_pad = int(((math.ceil(img_col / (dim_col * 2)) * (dim_col * 2)) - img_col) / 2)

    img_pad = cv.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv.BORDER_REPLICATE)

    return (img_pad, top_pad, bottom_pad, left_pad, right_pad)

def blur(pad_img, pad_top_left, weight):
    total_weight = np.asarray(weight).sum()

    img_pad_row = pad_img.shape[0]
    img_pad_col = pad_img.shape[1]

    img_pad_top = pad_top_left[0]
    img_pad_left = pad_top_left[1]

    img_pad_blur = copy.deepcopy(pad_img)
    for i in range(0, img_pad_row - 2):
        for j in range(0, img_pad_col - 2):
            subimage = np.asarray(pad_img[i:i+3, j:j+3, :])

            _sum = 0
            idx = 0
            for k in range(0, 3):
                for p in range(0, 3):
                    subimage[k][p] *= weight[idx]
                    _sum += subimage[k][p][0]
                    idx += 1
            
            avg = round(_sum / total_weight)

            # replace the center pixel intensity with the avg intensity
            img_pad_blur[i + img_pad_top][j + img_pad_left] = np.asarray([avg] * 3)

    return img_pad_blur

def laplacian(img_path, kernel=[0, 1, 0, 1, -4, 1, 0, 1, 0]):
    img = cv.imread(img_path)
    ret_tuple = pad_img(img, (3,3))

    img_pad = ret_tuple[0]
    img_pad_row = img_pad.shape[0]
    img_pad_col = img_pad.shape[1]
    img_pad_top = ret_tuple[1]
    img_pad_bottom = ret_tuple[2]
    img_pad_left = ret_tuple[3]
    img_pad_right = ret_tuple[4]

    # blur the padding image to remove noise
    img_pad_blur = blur(img_pad, (img_pad_top, img_pad_left), [1, 1, 1, 1, 1, 1, 1, 1, 1])

    # lapla image result
    img_lapla = copy.deepcopy(img_pad_blur)

    # sharp image result
    img_sharp = copy.deepcopy(img_pad_blur)
    
    # convert image data type to 16 bit signed integer to prevent overflow
    img_sharp = img_sharp.astype(np.int16)
    img_sharp_row = img_sharp.shape[0]
    img_sharp_col = img_sharp.shape[1]

    for i in range(0, img_pad_row - 2):
        for j in range(0, img_pad_col - 2):
            lapla = 0
            idx = 0
            subimage = np.asarray(img_pad_blur[i:i+3, j:j+3, :], np.int16)

            for k in range(0, 3):
                for p in range(0, 3):
                    lapla += subimage[k][p][0] * kernel[idx]
                    idx += 1

            # change the sign depends on the lapla's sign
            if lapla < 0:
                img_lapla[i + img_pad_top][j + img_pad_left] = np.asarray([-lapla] * 3)
                img_sharp[i + img_pad_top][j + img_pad_left] -= lapla
            else:
                img_lapla[i + img_pad_top][j + img_pad_left] = np.asarray([lapla] * 3)
                img_sharp[i + img_pad_top][j + img_pad_left] += lapla
 
    # remove padding
    img_sharp = np.asarray(img_sharp[img_pad_top:img_sharp_row - img_pad_bottom, \
                                     img_pad_left:img_sharp_col - img_pad_right, :], np.uint8) 

    # convert image data type back to 8 bit unsigned integer
    img_sharp = cv.convertScaleAbs(img_sharp)
    img_lapla = cv.convertScaleAbs(img_lapla)

    show_imgs([('original', img), ('processed', img_sharp), ('laplacian', img_lapla)])

def highboost(img_path, k):
    img = cv.imread(img_path)
    img_row = img.shape[0]
    img_col = img.shape[1]

    ret_tuple = pad_img(img, (3,3))
    img_pad = ret_tuple[0]
    img_pad_top = ret_tuple[1]
    img_pad_bottom = ret_tuple[2]
    img_pad_left = ret_tuple[3]
    img_pad_right = ret_tuple[4]
    img_pad_row = img_pad.shape[0]
    img_pad_col = img_pad.shape[1]

    # blur the padding image
    img_pad_blur = blur(img_pad, (img_pad_top, img_pad_left), [1, 1, 1, 1, 1, 1, 1, 1, 1])

    # remove padding
    img_blur = np.asarray(img_pad_blur[img_pad_top:img_pad_row  - img_pad_bottom, \
                                     img_pad_left:img_pad_col - img_pad_right, :], np.uint8) 
    img_blur_row = img_blur.shape[0]
    img_blur_col = img_blur.shape[1]

    # convert image data type to 16 bit signed integer to prevent overflow
    img_sharp = copy.deepcopy(img)
    img_sharp = img_sharp.astype(np.int16)

    for i in range(0, img_row):
        for j in range(0, img_col):
            # get the mask by subtract the blurred image from the original image
            mask = img[i][j] - img_blur[i][j]

            # add the mask with k to the original and produce the sharpend image
            img_sharp[i][j] = img[i][j] + np.multiply(k, mask)

    # convert image data type back to 8 bit unsigned integer
    img_sharp = cv.convertScaleAbs(img_sharp)

    show_imgs([('original', img), ('processed', img_sharp)])

if __name__ == '__main__':
    script_name = sys.argv[0]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:i:k:h")
    except getopt.GetoptError as err:
        print(err)
        print("\n")
        usage(script_name)

    img_file = None
    mode = None
    k = 3
    has_k = False
    for opt, arg in opts:
        if opt in ['-i']:
            if not is_img_file(arg):
                print("Invalid image file")
                exit()
            img_file = arg
        elif opt in ['-m']:
            if not valid_mode(arg):
                print("Invalid operation mode. Check valid mode in usage(python %s -h)" % (script_name))
                exit()
            mode = arg
        elif opt in ['-k']:
            k = int(arg)
            has_k = True
        elif opt in ['-h']:
            usage(script_name)

    if mode == 'laplacian' and has_k != False:
        print("-k option only available for highboost filtering mode")
        usage(script_name)

    if mode == None:
        print("Required a mode")
        usage(script_name)

    if img_file == None:
        print("Required an specified image file")
        usage(script_name)

    print("Sharpening %s with %s filtering" % (img_file, mode))

    if mode == 'laplacian':
        laplacian(img_file)
    else:
        highboost(img_file, k)

    exit()