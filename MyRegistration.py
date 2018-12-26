import cv2
import numpy as np

# from imgregiscls import ImageRegistration
from test import ImageRegistration
# 读入图片
img1 = cv2.imread('1.jpg') # 左边的图
img2 = cv2.imread('2.jpg') # 右边的图,进行透视变换

iregis = ImageRegistration(meth='surf', ratio=0.6, reprojThresh=5.0, show_lines=True, Hessian_Thresh=700, opti_method='g')
# iregis = ImageRegistration()
img_result = iregis.registration([img1, img2])
imglist = [img1, img2]
# result = iregis.guass_lup(imglist)

cv2.imwrite('g_img_result.jpg',img_result)
# cv2.imshow('', img_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()