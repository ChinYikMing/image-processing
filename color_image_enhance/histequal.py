import cv2 as cv
import numpy as np
import sys, getopt, os, copy, math
from matplotlib import pyplot as plt
import rgb2hsi_hsi2rgb as rh

def get_cdf(pmf, k_level_max):
    cdf = []

    for i in range(0, len(pmf)):
        x = 0
        for j in range(0, i + 1):
            x += pmf[j]
        cdf.append(round(k_level_max * x))

    cdf = np.asarray(cdf)
    cdf = (cdf - cdf.min()) * k_level_max / (cdf.max()-cdf.min())
    cdf = np.ma.filled(cdf,0).astype('uint8')
    return cdf

def get_pmf(hist, img_size):
    pmf = []
    total_pixel = img_size
    for x in hist:
        pmf.append(x / total_pixel)
    return pmf

def _histequal(img, k_level_max=255):
    img_row = img.shape[0]
    img_col = img.shape[1]
    img_size = img_row * img_col
    gray_level_cnt = k_level_max + 1

    # calculate # of pixel at gray level k 
    n = [0] * gray_level_cnt
    for i in range(0, img_row):
        for j in range(0, img_col):
            n[img[i][j]] += 1

    # calculate the pmf 
    pmf = get_pmf(n, img_size)

    # calculate the cdf
    cdf = get_cdf(pmf, k_level_max=k_level_max)
    equa = cdf[img]
    return equa

def histequal(img, color_space):
    if color_space == "rgb":
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]
        # cv.imshow('B', B)
        # cv.imshow('G', G)
        # cv.imshow('R', R)

        R_histequal = _histequal(R)
        G_histequal = _histequal(G)
        B_histequal = _histequal(B)

        res = np.copy(img)
        res[:, :, 0] = B_histequal
        res[:, :, 1] = G_histequal
        res[:, :, 2] = R_histequal

        return res
    elif color_space == "hsi":
        hsi = rh.rgb2hsi(img)
        # H = hsi[:, :, 0]
        # S = hsi[:, :, 1]
        I = hsi[:, :, 2]
        # cv.imshow('H', H)
        # cv.imshow('S', S)
        # cv.imshow('I', I)

        I_histequal = _histequal((I * 255).astype('uint8'))
        hsi[:, :, 2] = (I_histequal / 255.0).astype('float')

        return rh.hsi2rgb(hsi)
