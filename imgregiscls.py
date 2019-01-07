import cv2
import numpy as np

# 2019/1/7 测试新的SSH
class ImageRegistration(object):
    """
    默认值：ratio=0.5, reprojThresh=5.0, show_lines=False, Hessian_Thresh=600
        @.ratio  用于筛选good_matches,0~1.0；
        @.reprojThresh  cv2.findHomography()传入参数，点对的阈值，原图像的点经过变换后点与目标图像上对应点的误差，1~10；
        @.show_lines  特征点之间连线，默认不画（False)；
        @.Hessian_Thresh  Hessian矩阵的阈值，阈值越大能检测的特征就越少   
    """
    def __init__(self,meth='surf', ratio=0.50, reprojThresh=5.0, show_lines=False, Hessian_Thresh=700, opti_method='g'):
        self.method, self.ratio, self.reprojThresh, self.show_lines, self.Hessian_Thresh, self.opti_method =meth, ratio, reprojThresh, show_lines, Hessian_Thresh, opti_method

    def registration(self,imgs):
        """
        输入一对图像，完成融合拼接，返回融合后图像
        """
        (img1,img2) = imgs
        
        # 灰度图转换
        img1gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        # 提取特征点
        if self.method == 'surf':
            surf_1 = cv2.xfeatures2d.SURF_create(self.Hessian_Thresh)  # self.Hessian_Thresh 是Hessian矩阵的阈值，阈值越大能检测的特征就越少
        else:
            surf_1 = cv2.xfeatures2d.SIFT_create(self.Hessian_Thresh)  # sift算法
        keypoints_surf_img2,desc2 = surf_1.detectAndCompute(img2gray,None) # None为mask参数
        keypoints_surf_img1,desc1 = surf_1.detectAndCompute(img1gray,None)
        
        """
        Keypoints类包含关键点位置、方向等属性信息：
            @.pt(2f):位置坐标；
            @.size(float):特征点邻域直径；
            @.angle(float):特征点方向(0~360°)，负值表示不使用；
            @.octave(int):特征点所在图像金字塔组；
            @.class_id(int)：用于聚类的id
        """
        good_match_points, Mat_trans, mask = self.get_matrix_goodmatch(keypoints_surf_img2,keypoints_surf_img1,desc2,desc1)
        
        # good_match_points绘线
        self.drawMatchesKnn_cv2(img1,img1gray,keypoints_surf_img1,img2,img2gray,keypoints_surf_img2,good_match_points)
        
        # 透视变换
        img_joint = cv2.warpPerspective(img2,Mat_trans,(img1.shape[1]+img2.shape[1],max(img1.shape[0],img2.shape[0])))
        temp = img_joint 
       
        # 边界优化
        if self.opti_method == 'w':
            img_joint[:img1.shape[0],:img1.shape[1]] = img1        
            # 四个角
            self.left_top,self.left_bottom,self.right_top,self.right_bottom = self.get_corners(img2,Mat_trans)
            img_result = self.OptimizeSeam(img1,temp,img_joint)
        elif self.opti_method == 'g':
            guasslup_list = [img1, temp]
            img_result = self.guass_lup(guasslup_list)
        return img_result

    def get_matrix_goodmatch(self,keypoints_src,keypoints_dst,desc_src,desc_dst):
        """
        依次传入src_points,dst_points以及description_src,description_dst
        TIPs:
        此处可使用Flann算法或'BruteForce'算法。
        > 1. 若使用FLANN匹配需要传入两个字典参数：
                @1.一个参数是IndexParams，对于SIFT和SURF，可以传入参数:
                index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                对于ORB，可以传入参数index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)；
                @2.第二个参数是SearchParams，可以传入参数:
                search_params=dict(checks=100)，
                它来指定递归遍历的次数，值越高结果越准确，但是消耗的时间也越多；
        > 2. Dmath类包含匹配对应的特征描述子索引、欧式距离等属性
                @.queryIdx(int):该匹配对应的查询图像的特征描述子索引；
                @.trainIdx(int):该匹配对应的训练(模板)图像的特征描述子索引；
                @.imgIdx(int):训练图像的索引(若有多个)； 
                @.distance(float):两个特征向量之间的欧氏距离，越小表明匹配度越高；
        """
        # >1.Flann算法
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc_src, desc_dst, 2)
        # >2.BruteForce算法
        # matcher = cv2.DescriptorMatcher_create('BruteForce')
        # matches = matcher.knnMatch(desc_src, desc_dst, 2)  # 2为KNN算法中的k值
        # matches为一对对Dmath类对象组成的列表
        good_match_points = []
        for m,n in matches:
            if m.distance < self.ratio*n.distance:
                good_match_points.append(m)
        imagePoints1=[]
        imagePoints2=[]
        for i in range(len(good_match_points)):
            imagePoints1.append(keypoints_src[good_match_points[i].queryIdx].pt)
            imagePoints2.append(keypoints_dst[good_match_points[i].trainIdx].pt)
        imagePoints1 = np.float32(imagePoints1).reshape(-1,2)
        imagePoints2 = np.float32(imagePoints2).reshape(-1,2)        
        Mat_trans,mask = cv2.findHomography(imagePoints1, imagePoints2, cv2.RANSAC,self.reprojThresh)
        if len(good_match_points) >= 4:
            return good_match_points, Mat_trans, mask
        else:
            return None
            
    def drawMatchesKnn_cv2(self,img1,img1gray,keypoints_surf_img1,img2,img2gray,keypoints_surf_img2,good_match_points):
        """在两幅原图中绘制匹配特征点的连线,默认show_lines=False不显示"""
        if self.show_lines == True:
            # 将两幅图左右放置在一张图中       
            img_draw_lines = np.zeros((max(img1gray.shape[0], img2gray.shape[0]), img1gray.shape[1] + img2gray.shape[1], 3), np.uint8)
            img_draw_lines[:img1gray.shape[0], :img1gray.shape[1]] = img1
            img_draw_lines[:img2gray.shape[0], img1gray.shape[1]:img1gray.shape[1] + img2gray.shape[1]] = img2
            # 获取匹配点坐标索引
            pointIdx_of_img1 = [point_Dmath.trainIdx for point_Dmath in good_match_points]
            pointIdx_of_img2 = [point_Dmath.queryIdx for point_Dmath in good_match_points]
            # 获取匹配点坐标
            img1_points_coordi = np.int32([keypoints_surf_img1[idx].pt for idx in pointIdx_of_img1])
            img2_points_coordi = np.int32([keypoints_surf_img2[idx].pt for idx in pointIdx_of_img2]) + (img1gray.shape[1], 0)
            # 绘线
            for (x1, y1), (x2, y2) in zip(img1_points_coordi, img2_points_coordi):
                cv2.line(img_draw_lines, (x1, y1), (x2, y2), (0,0,255), 2, lineType = 8)
            cv2.namedWindow("match",cv2.WINDOW_NORMAL)
            cv2.imshow('match',img_draw_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            pass

    def get_corners(self,src_img,Mat_trans):
        """返回四个角"""
        # 原始坐标
        self.left_top_ori = np.array([[0],[0],[1]],np.float32)
        self.left_bottom_ori = np.array([[0],[src_img.shape[0]],[1]],np.float32)
        self.right_top_ori = np.array([[src_img.shape[1]],[0],[1]],np.float32)
        self.right_bottom_ori = np.array([[src_img.shape[1]],[src_img.shape[0]],[1]],np.float32)
        # 转换后坐标
        vec_lt = np.dot(Mat_trans,self.left_top_ori)
        self.left_top = (vec_lt[0]/vec_lt[2],vec_lt[1]/vec_lt[2])
        vec_lb = np.dot(Mat_trans,self.left_bottom_ori)
        self.left_bottom = (vec_lb[0]/vec_lb[2],vec_lb[1]/vec_lb[2]) 
        vec_rt = np.dot(Mat_trans,self.right_top_ori)
        self.right_top = (vec_rt[0]/vec_rt[2],vec_rt[1]/vec_rt[2])
        vec_rb = np.dot(Mat_trans,self.right_bottom_ori)
        self.right_bottom = (vec_rb[0]/vec_rb[2],vec_rb[1]/vec_rb[2])
        # 各项最值
        self.x_min = min(self.left_top[0],self.left_bottom[0],self.right_top[0],self.right_bottom[0])
        self.y_min = min(self.left_top[1],self.left_bottom[1],self.right_top[1],self.right_bottom[1])
        self.x_max = max(self.left_top[0],self.left_bottom[0],self.right_top[0],self.right_bottom[0])
        self.y_max = max(self.left_top[1],self.left_bottom[1],self.right_top[1],self.right_bottom[1])
        self.len_x = self.x_max-self.x_min
        self.len_y = self.y_max-self.y_min

        return self.left_top,self.left_bottom,self.right_top,self.right_bottom      
    
    def OptimizeSeam(self,img1,temp,img_joint):
        """ 对边界按权值进行优化
        """
        if self.y_min < 0:
            self.y_min = 0
        if self.x_min < 0:
            self.x_min = 0
        if self.x_max > min(img_joint.shape[1],img1.shape[1]):
            self.x_max = min(img_joint.shape[1],img1.shape[1])
        if self.y_max > min(img_joint.shape[0],img1.shape[0]):
            self.y_max = min(img_joint.shape[0],img1.shape[0])
        process_width = self.x_max - self.x_min
        for j in range(int(self.x_min),int(self.x_max)): 
            for i in range(int(self.y_min),int(self.y_max)):
                # if (img_joint[i][j][0] == 0) and (img_joint[i][j][1] == 0) and (img_joint[i][j][2] == 0):
                #     alpha = 1.0
                if not (img_joint[i][j]-[0,0,0]).any():
                    alpha = 1.0
                else:
                    alpha = (img1.shape[1]-j)/process_width
                img_joint[i][j] = temp[i][j]*(1-alpha)+img1[i][j]*alpha  
        return img_joint

    def guass_lup(self,imglist):
        """
        高斯金字塔融合
        """
        LeftImg, RightImg = imglist[0], imglist[1]
        # 对图像进行金字塔运算前预处理
        LeftImg = self.transimg(LeftImg, 'l')
        RightImg = self.transimg(RightImg, 'r')
        # 分别获取左右图像高斯金字塔
        gua_LeftImg_pyramid = self.GaussiPyramidProcess(LeftImg, 6)
        gua_RightImg_pyramid = self.GaussiPyramidProcess(RightImg, 6)

        # 拉普拉斯金字塔，总共5级处理，高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
        lup_LeftImg_pyramid = self.LupPyramidProcess(gua_LeftImg_pyramid)
        lup_RightImg_pyramid = self.LupPyramidProcess(gua_RightImg_pyramid)

        # 存放拼接后的Lup图,将LeftImg和RightImg的 （高斯N层-N-1层升采样）的差值  进行拼接，存入列表
        # 差值实际就是gauss升采样的处理结果
        LS = [] 
        for lleft,lright in zip(lup_LeftImg_pyramid,lup_RightImg_pyramid):
            rows,cols,dpt = lleft.shape
            # hstack 和 vstack分别：水平拼接和竖直拼接
            # ls = np.vstack((lleft[:rows//2,:],lright[rows//2:,:]))
            ls = np.hstack((lleft[:,:cols],lright[:,cols:]))    
            LS.append(ls)

        ls_ = LS[0] # LS[0]是最小的一张图
        for i in np.arange(1,6):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        return ls_

    def GaussiPyramidProcess(self, img, num):
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

    def LupPyramidProcess(self, gua_img_pyramid, num=5):
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

    def transimg(self, src_img, flag):   
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

