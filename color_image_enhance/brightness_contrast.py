import numpy as np
import rgb2lab_lab2rgb as rl
import cv2 as cv

def brightness_contrast(img, alpha, beta):
        alpha /= 100
        beta -= 100

        w = img.shape[0]
        h = img.shape[1]
        img_new = np.zeros((w,h,3))
        lab = np.zeros((w,h,3))
        for i in range(w):
            for j in range(h):
                Lab = rl.rgb2lab(img[i,j])
                lab[i, j] = (Lab[0], Lab[1], Lab[2])

        L = lab[:, :, 0].astype('uint8')

        # enhancement
        new_image = np.copy(L)
        for y in range(L.shape[0]):
            for x in range(L.shape[1]):
                new_image[y,x] = np.clip(alpha * L[y,x] + beta, 0, 100)

        # new_image = np.clip(cv.convertScaleAbs(L, alpha=alpha, beta=beta), 0, 100)

        lab[:, :, 0] = new_image.astype('float')

        for i in range(w):
            for j in range(h):
                rgb = rl.lab2rgb(lab[i,j])
                img_new[i, j] = (rgb[2], rgb[1], rgb[0])

        cv.imwrite('res.jpg', img_new)

        return cv.imread('res.jpg')
            
    
    