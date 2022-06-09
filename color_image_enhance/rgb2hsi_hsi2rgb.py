import cv2
import numpy as np
import math

def __Rgb2Hsi(R, G, B):
    R /= 255
    G /= 255
    B /= 255
    eps =  1e-8
    H, S, I = 0, 0, 0
    sumRGB = R + G + B
    Min = min(R,G,B)
    S = 1 - 3 * Min / (sumRGB + eps)
    H = np.arccos((0.5 * (R + R - G - B)) / np.sqrt((R - G) * (R - G) + (R - B) * (G - B) + eps))
    if B > G:
        H = 2 * np.pi - H
    H = H / (2 * np.pi)
    if S == 0:
        H = 0
    I = sumRGB / 3
    return np.array([H, S, I], dtype = float)

def rgb2hsi(img):
    HSIimg = np.zeros(img.shape, dtype = float)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            HSIimg[w,h,:] = __Rgb2Hsi(img[w,h,0],img[w,h,1],img[w,h,2])

    return HSIimg

def __Hsi2Rgb(H, S, I):
    pi3 = np.pi / 3

    H *= 2 * np.pi
    if H >= 0 and H < 2 * pi3:
        B = I * (1 - S)
        R = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        G = 3 * I - (R + B)
    elif H >= 2 * pi3 and H <= 4 * pi3:
        H = H - 2 * pi3
        R = I * (1 - S)
        G = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        B = 3 * I - (R + G)
    else:
        H = H - 4 * pi3
        G = I * (1 - S)
        B = I * (1 + S * np.cos(H) / np.cos(pi3 - H))
        R = 3 * I - (B + G)
    return (np.array([R,G,B]) * 255).astype(np.uint8)

def hsi2rgb(img):
    RGBimg = np.zeros(img.shape, dtype = np.uint8)
    width, height = img.shape[:2]
    for w in range(width):
        for h in range(height):
            RGBimg[w,h,:] = __Hsi2Rgb(img[w,h,0],img[w,h,1],img[w,h,2])
    return RGBimg