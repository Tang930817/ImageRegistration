import cv2
import numpy as np


def drawMatchesKnn_cv2(img1_gray,keypoint_img1,img2_gray,keypoints_img2,goodMatch):
    
    h1, w1 = img1_gray.shape[:2]
    h2, w2 = img2_gray.shape[:2]
 
    img_draw_lines = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_draw_lines[:h1, :w1] = img1_gray
    img_draw_lines[:h2, w1:w1 + w2] = img2_gray
 
    pointIdx_of_img1 = [point_Dmath.queryIdx for point_Dmath in goodMatch]
    pointIdx_of_img2 = [point_Dmath.trainIdx for point_Dmath in goodMatch]
 
    img1_points_coordi = np.int32([keypoint_img1[idx].pt for idx in pointIdx_of_img1])
    img2_points_coordi = np.int32([keypoints_img2[idx].pt for idx in pointIdx_of_img2]) + (w1, 0)
 
    for (x1, y1), (x2, y2) in zip(img1_points_coordi, img2_points_coordi):
        cv2.line(img_draw_lines, (x1, y1), (x2, y2), (0,0,255))

    # cv2.namedWindow("match",cv2.WINDOW_NORMAL)
    # cv2.imshow('match',img_draw_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 读入两幅图
img1 = cv2.imread('gg2.jpg') # 右边的图，进行透视变换
img2 = cv2.imread('gu2.jpg') # 左边的图

# 灰度图转换
img1gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# 提取特征点
surf_1 = cv2.xfeatures2d.SURF_create(400)  # 400 是Hessian矩阵的阈值，阈值越大能检测的特征就越少
surf_2 = cv2.xfeatures2d.SURF_create(400)

"""
Keypoints类包含关键点位置、方向等属性信息：
    @.pt(2f):位置坐标；
    @.size(float):特征点邻域直径
    @.angle(float):特征点方向(0~360°)，负值表示不使用
    @.octave(int):特征点所在图像金字塔组
    @.class_id(int)：用于聚类的id
"""
keypoints_surf_img1,desc1 = surf_1.detectAndCompute(img1gray,None) # None为mask参数
keypoints_surf_img2,desc2 = surf_2.detectAndCompute(img2gray,None)

"""
使用FLANN匹配需要传入两个字典参数：
    @1.一个参数是IndexParams，对于SIFT和SURF，可以传入参数:
    index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)。
    对于ORB，可以传入参数index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    @2.第二个参数是SearchParams，可以传入参数:
    search_params=dict(checks=100)，
    它来指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多.
"""
matcher = cv2.DescriptorMatcher_create('BruteForce')
matches = matcher.knnMatch(desc1, desc2, 2)    # 2为KNN算法中的k值
# matches为一对对Dmath类对象组成的列表
"""
Dmath类包含匹配对应的特征描述子索引、欧式距离等属性
    @.queryIdx(int):该匹配对应的查询图像的特征描述子索引  
    @.trainIdx(int):该匹配对应的训练(模板)图像的特征描述子索引  
    @.imgIdx(int):训练图像的索引(若有多个)  
    @.distance(float):两个特征向量之间的欧氏距离，越小表明匹配度越高
"""

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(desc1,desc2,k=2)

good_match_points = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good_match_points.append(m)
# print(good_match_points[0].trainIdx,good_match_points[0].queryIdx)

first_match = drawMatchesKnn_cv2(img1,keypoints_surf_img1,img2,keypoints_surf_img2,good_match_points)


imagePoints1=[]
imagePoints2=[]
for i in range(len(good_match_points)):
    imagePoints1.append(keypoints_surf_img1[good_match_points[i].queryIdx].pt)
    imagePoints2.append(keypoints_surf_img2[good_match_points[i].trainIdx].pt)
print(len(matches))
print(len(imagePoints1))
imagePoints1 = np.float32(imagePoints1).reshape(-1,2)
imagePoints2 = np.float32(imagePoints2).reshape(-1,2)

reprojThresh = 5.5 # 每个sample相对于model的成功阈值
M_trans,mask = cv2.findHomography(imagePoints1,imagePoints2,cv2.RANSAC,reprojThresh)
print(M_trans,mask.shape)
transl = cv2.warpPerspective(img1,M_trans,(img1.shape[1]+img2.shape[1],max(img1.shape[0],img2.shape[0])))
transl[:img2.shape[0],:img2.shape[1]] = img2

# cv2.imshow('dst',transl)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('translat.jpg',transl)