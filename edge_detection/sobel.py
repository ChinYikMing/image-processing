import numpy as np
import basis as bs
import cv2 as cv
import copy

def _Sobel(src, kernel):
    img_pad = bs.pad_img(src)
    res = np.zeros(src.shape)

    for i in range(1, img_pad.shape[0] - 1):
        for j in range(1, img_pad.shape[1] - 1):
            sum = 0            
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    sum += kernel[m][n]*img_pad[i+m-1][j+n-1]
            res[i-1][j-1] = sum                           
    
    return res

def Sobel_sum(x, y):
    ret = np.zeros(x.shape)    
    list = []
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            tmp = (x[i][j]**2 + y[i][j]**2)**(1/2)            
            ret[i][j] = tmp
            list.append(tmp)
    ret /= max(list)
    return ret

def Sobel(src):
    B = src[:, :, 0]
    G = src[:, :, 1]
    R = src[:, :, 2]
    
    # kernel_x = np.asarray([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_x = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    # kernel_y = np.asarray([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernel_y = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])

    # sobel on B
    B_res_x = _Sobel(B, kernel_x)
    B_res_y = _Sobel(B, kernel_y)
    B_res = Sobel_sum(B_res_x, B_res_y)

    # sobel on G
    G_res_x = _Sobel(G, kernel_x)
    G_res_y = _Sobel(G, kernel_y)
    G_res = Sobel_sum(G_res_x, G_res_y)

    # sobel on R
    R_res_x = _Sobel(R, kernel_x)
    R_res_y = _Sobel(R, kernel_y)
    R_res = Sobel_sum(R_res_x, R_res_y)

    res = copy.deepcopy(src)

    res = cv.merge((B_res, G_res, R_res))
    orig = cv.merge((B, G, R))

    cv.imshow('original', orig)
    cv.imshow('edged', res)
    cv.waitKey(0)
    cv.destroyAllWindows()
