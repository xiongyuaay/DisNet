import cv2
import numpy as np
import os
 
 
def getColorImg(alpha,beta,img_path,img_write_path):
    img = cv2.imread(img_path)
    colored_img = np.uint8(np.clip((alpha * img + beta), 0, 255))
    cv2.imwrite(img_write_path,colored_img)
 
def color(alpha,beta,img_dir,img_write_dir):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
 
    img_names=os.listdir(img_dir)
    print(img_names)
    for img_name in img_names:
        print(img_name, 'color')
        if(img_name == 'EC'):
            continue
        img_path=os.path.join(img_dir,img_name)
        img_write_path=os.path.join(img_write_dir,'color'+str(int(alpha*10))+img_name[:-4]+'.png')
 
        getColorImg(alpha,beta,img_path,img_write_path)
 
def claheMethod(img_dir,img_write_dir):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
 
    img_names=os.listdir(img_dir)
    for img_name in img_names:
        print(img_name, 'clahe')
        if(img_name == 'EC'):
            continue
        img_path=os.path.join(img_dir,img_name)
        img_write_path=os.path.join(img_write_dir,'clahe'+img_name[:-4]+'.png')
 
        img = cv2.imread(img_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#彩色图要拆分三个通道分别做均衡化，否则像我这里一样转为灰度图
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 自适应均衡化，参数可选
        cl1 = clahe.apply(hsv)
 
        #测试加了滤波没能让边缘清晰
        #cl1MedianBlur = cv2.medianBlur(cl1, 1)
        # cl1GaussianBlur = cv2.GaussianBlur(cl1, (1, 1), 0)
        cv2.imwrite(img_write_path, cl1)
 
 
def adjust_gamma(img_dir,img_write_dir,gamma = 1.0):
    if not os.path.exists(img_write_dir):
        os.makedirs(img_write_dir)
 
    img_names=os.listdir(img_dir)
    for img_name in img_names:
        print(img_name, 'gamma')
        if(img_name == 'EC'):
            continue
        img_path=os.path.join(img_dir,img_name)
        img_write_path=os.path.join(img_write_dir,'adjust_gamma'+img_name[:-4]+'.png')
 
        image = cv2.imread(img_path)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        cv2.imwrite(img_write_path, cv2.LUT(image, table))
 
 
 
def AdjustColor(img_dir, img_write_dir):
    alphas = [0.3, 0.5, 1.2]                             
    beta = 10
    for alpha in alphas:                             
        color(alpha, beta, img_dir, img_write_dir)   
                                                     
    # #第二步自适应直方图均衡化，减少色彩不同和不均衡影响                     
    claheMethod(img_dir, img_write_dir)              
                                                     
    #第三步伽马矫正，减少光照影响                                  
    gamma = 2.2                                      
    adjust_gamma(img_dir, img_write_dir,gamma=gamma) 
     
     
     
     
     
     
     
     
