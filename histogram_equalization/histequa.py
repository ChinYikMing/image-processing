import cv2 as cv
import numpy as np
import os
import sys, getopt, os, copy
from matplotlib import pyplot as plt

def usage(script_name):
    print("Usage: python %s -m mode -i image_path" % script_name)
    print("       valid mode are: 'global' or 'local'")
    exit()

def is_img_file(path):
    return os.path.isfile(path) and \
            (path.endswith(".tif") or \
            path.endswith(".png") or \
            path.endswith(".bmp") or \
            path.endswith(".jpeg") or path.endswith(".jpg"))

def valid_mode(mode):
    if mode == "global" or mode == "local":
        return True
    return False

def local_show_orig_equa_img_and_hist(img, img_name, equa, blocks, equa_blocks):
    cv.imshow('original', img)
    cv.imshow('equalized', equa)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print("Saving blocks and equalized blocks histogram figures")
    print("It will generate 8 figures at total, please wait for it...")

    # show 256 16x16 block histogram and their euqualized block histogram
    fig, blocks_axs = plt.subplots(8, 8, sharey=True, tight_layout=True)
    fig2, equa_blocks_axs = plt.subplots(8, 8, sharey=True, tight_layout=True)

    idx = 0
    # save 4 8x8 blocks figure, so we need to save 4 such figure to get 256 subplot
    for i in range(0, 4):
        for row in blocks_axs:
            for col in row:
                col.hist(blocks[idx].flatten(), 256, [0, 255], color='blue', edgecolor='black')
                idx += 1
        fig.savefig('original block histogram figure(' + img_name + ') ' + str(i))
        print("saved")

    # save 4 8x8 equa_blocks figure, so we need to save 4 such figure to get 256 subplot
    idx = 0
    for i in range(0, 4):
        for row in equa_blocks_axs:
            for col in row:
                col.hist(equa_blocks[idx].flatten(), 256, [0, 255], color='brown', edgecolor='black')
                idx += 1
        fig2.savefig('equalized block histogram figure(' + img_name + ')' + str(i))
        print("saved")

    print("done")

def global_show_orig_equa_img_and_hist(img, equa):
    cv.imshow('original', img)
    cv.imshow('equalized', equa)
    cv.waitKey(0)
    cv.destroyAllWindows()

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(img.flatten(), 256, [0, 255], color='blue', edgecolor='black')
    axs[0].set_title("original")
    axs[1].hist(equa.flatten(), 256, [0, 256], color='brown', edgecolor='black')
    axs[1].set_title("equalized")
    plt.show()

def get_cdf(pmf, k_level_max):
    cdf = []
    for i in range(0, len(pmf)):
        x = 0
        for j in range(0, i):
            x += pmf[j]
        cdf.append(round(k_level_max * x))
    return cdf

def get_pmf(hist, img_size):
    pmf = []
    total_pixel = img_size
    for x in hist:
        pmf.append(x / total_pixel)
    return pmf

def hist_equalize(img, block_row=-1, block_col=-1):
    img_row = img.shape[0]
    img_col = img.shape[1]
    img_size = img_row * img_col
    gray_level_cnt = 256

    if block_row == -1 and block_col == -1: # global approach
        img_flat = img.flatten()
        # calculate # of pixel at gray level k 
        n = [0] * gray_level_cnt
        for i in range(0, img_size * 3, 3):  # note: every three element represent a pixel intensity
            n[img_flat[i]] += 1

        # calculate the pmf 
        pmf = get_pmf(n, img_size)

        # calculate the s = T(r)
        s = get_cdf(pmf, k_level_max=255)
        s = np.array(s, dtype="uint8")
        equa = s[img]
        return equa
    else:                                    # local approach
        block_size = block_row * block_col
        blocks = []
        for i in range(0, img_row, 16):
            for j in range(0, img_col, 16):
                blocks.append(np.asarray(img[i:i+16, j:j+16, :]))
        
        # calculate # of pixel at gray level k in each block
        n = [] * img_row # 256x256 / 16x16 = 256(img_row) => we need 256 16x16 blocks
        for i in range(0, len(blocks)):
            block_flat = blocks[i].flatten()
            n_tmp = [0] * gray_level_cnt
            for j in range(0, block_size * 3, 3):  # note: every three element represent a pixel intensity
                n_tmp[block_flat[j]] += 1
            n.append(n_tmp)

        # calculate the pmf 
        block_pmf = []
        for i in range(0, len(blocks)):
            block_pmf.append(get_pmf(n[i], block_size))
        
        # calculate the s = T(r)
        blocks_s = []
        for i in range(0, len(blocks)):
            s_tmp = get_cdf(block_pmf[i], k_level_max=255)
            s_tmp = np.array(s_tmp, dtype="uint8")
            blocks_s.append(s_tmp)

        # collect all equa_block for display
        equa_blocks = []

        # combine all the histogram equilized block
        equa = copy.deepcopy(img)
        for i in range(0, img_row, 16):
            for j in range(0, img_col, 16):
                equa[i:i+16, j:j+16, :] = blocks_s[i][img[i:i+16, j:j+16, :]]
                equa_blocks.append(np.asarray(equa[i:i+16, j:j+16, :]))

        return (equa, blocks, equa_blocks)

def global_hist_equa(img_path):
    img = cv.imread(img_path)
    equa = hist_equalize(img)
    global_show_orig_equa_img_and_hist(img, equa)

def local_hist_equa(img_path, block_row=16, block_col=16):
    img = cv.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    equa, blocks, equa_blocks = hist_equalize(img, block_row, block_col)
    local_show_orig_equa_img_and_hist(img, img_name, equa, blocks, equa_blocks)

if __name__ == '__main__':
    script_name = sys.argv[0]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:i:h")
    except getopt.GetoptError as err:
        print(err)
        print("\n")
        usage(script_name)

    img_file = None
    mode = None
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
        elif opt in ['-h']:
            usage(script_name)

    if mode == None:
        print("Required a mode")
        usage(script_name)

    if img_file == None:
        print("Required an specified image file")
        usage(script_name)

    print("%s approach on %s" % (mode, img_file))

    if mode == 'global':
        global_hist_equa(img_file)
    else:
        local_hist_equa(img_file, 16, 16)

    exit()