import cv2
import numpy as np 


def GaussiPyramidProcess(img, num):
    """
    将img进行高斯降采样处理，总共num级处理，resol↓
        @.img 输入图像
        @.num 高斯金字塔层数
        >.返回值为高斯金字塔列表
    """
    copy_img = img.copy()
    guassi_img_pyramid = [copy_img]
    for i in np.arange(num):
        copy_img = cv2.pyrDown(copy_img)
        guassi_img_pyramid.append(copy_img)
    return guassi_img_pyramid


def LupPyramidProcess(gua_img_pyramid, num=5):
    """
    TODO
    对高斯金字塔的第五层（默认）进行拉普拉斯升采样处理，总共num级处理，高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
        @.img 输入图像高斯金字塔
        @.num 高斯金字塔层数
        >.返回值为高斯金字塔列表
    """
    lup_img_pyramid = [gua_img_pyramid[num]]
    for i in np.arange(num,0,-1):
        gua_img_E = cv2.pyrUp(gua_img_pyramid[i])
        L_img = cv2.subtract(gua_img_pyramid[i-1],gua_img_E)
        lup_img_pyramid.append(L_img)
    return lup_img_pyramid


def gauss_lup(imglist):
    """
    高斯金字塔融合
    """
    LeftImg, RightImg = imglist[0], imglist[1]
    # 对图像进行金字塔运算前预处理
    LeftImg = transimg(LeftImg, 'l')
    RightImg = transimg(RightImg, 'r')
    # 分别获取左右图像高斯金字塔
    gua_LeftImg_pyramid = GaussiPyramidProcess(LeftImg, 6)
    gua_RightImg_pyramid = GaussiPyramidProcess(RightImg, 6)

    # 拉普拉斯金字塔，总共5级处理，高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
    lup_LeftImg_pyramid = LupPyramidProcess(gua_LeftImg_pyramid)
    lup_RightImg_pyramid = LupPyramidProcess(gua_RightImg_pyramid)

    # 存放拼接后的Lup图,将LeftImg和RightImg的 （高斯N层-N-1层升采样）的差值  进行拼接，存入列表
    # 差值实际就是gauss升采样的处理结果
    LS = [] 
    for la,lo in zip(lup_LeftImg_pyramid,lup_RightImg_pyramid):
        rows,cols,dpt = la.shape
        # hstack 和 vstack分别：水平拼接和竖直拼接
        # ls = np.vstack((la[:rows//2,:],lo[rows//2:,:]))
        # TODO
        ls = np.hstack((la[:,:int(0.5*cols)],lo[:,int(0.5*cols):]))    
        LS.append(ls)

    ls_ = LS[0] # LS[0]是最小的一张图
    for i in np.arange(1,len(gua_LeftImg_pyramid)-1):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


    src_h = int(src_img.shape[0])
    src_w = int(src_img.shape[1])
    add_h = 64-src_h%64
    add_w = 64-src_w%64
    h = src_h+add_h
    w = src_w+add_w
    img = np.zeros((h,w,3),np.float32)

    return img


def transimg(src_img, flag):   
    src_h = int(src_img.shape[0])
    src_w = int(src_img.shape[1])
    add_h = 64-src_h%64
    add_w = 64-src_w%64
    h = src_h+add_h
    w = src_w+add_w
    img = np.zeros((h,w,3),np.float32)
    if flag == 'l':
        img[:src_h, add_w:w] = src_img[:,:]
    elif flag == 'r':
        img[:src_h, :src_w] = src_img[:,:]
    else:
        print('缺少参数')
    return img


left_img = cv2.imread('gu2.jpg')
right_img = cv2.imread('gu1.jpg')
imglist = [left_img, right_img]

result = gauss_lup(imglist)
cv2.imwrite('result.jpg', result)
cv2.waitKey(0)
cv2.destroyAllWindows()