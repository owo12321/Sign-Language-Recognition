import bin.pyopenpose as op   # 注意要3.7才能正确import，版本与文件名对应

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from PIL import Image

# 视频路径，若要打开摄像头，设为0
filePath = 'examples/media/test4.mp4'

# 要追踪的关节点，点的定义在kepoints_hand.png和kepoints_pose_25.png里
body_point_list = [1, 2, 3, 4, 5, 6, 7] # 7个
left_hand_point_lsit = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20] # 16个
right_hand_point_list = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20] # 16个
# left_hand_point_lsit = [0] # 1
# right_hand_point_list = [0] # 1


# 身体和手部的颜色定义
body_color = [int(36.4*(i+1)) for i in range(7)]
hand_color = [int(15.9*(i+1)) for i in range(16)]
# print(body_color)
# print(hand_color)

# 找到模型文件
params = dict()
params["model_folder"] = "models/"
params["net_resolution"] = "256x256"    # 大小要设置成1：1的，且是2的倍数，1660ti超过256就爆显存了
params["hand"] = True

# opWrapper用来启动openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


# 打开视频
cap = cv2.VideoCapture(filePath)

# 轨迹图
track_img = np.zeros(shape=(1080, 1920, 3), dtype=np.int8)

fps = 0
while True:
    t0 = time.time()
    # 截取一帧图片
    flag, frame = cap.read()
    if flag == False:
        break
    # 将图片到加载openpose的运行时里做检测
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 打印出点的位置
    # datum.poseKeypoints的类型是ndarray
    # print("Body keypoints: \n" + str(datum.poseKeypoints))

    # 获取关节点
    body_point = datum.poseKeypoints
    left_hand_point = datum.handKeypoints[0]
    right_hand_point = datum.handKeypoints[1]


    # 绘制点轨迹图
    color_idx = 0
    for idx in body_point_list:
        cv2.circle(track_img, (int(body_point[0][idx][0]),int(body_point[0][idx][1])), 1, (0, 0, body_color[color_idx]), 5)
        color_idx += 1
        
    color_idx = 0
    for idx in left_hand_point_lsit:
        cv2.circle(track_img, (int(left_hand_point[0][idx][0]),int(left_hand_point[0][idx][1])), 1, (0, hand_color[color_idx], 0), 5)
        color_idx += 1
    
    color_idx = 0
    for idx in right_hand_point_list:
        cv2.circle(track_img, (int(right_hand_point[0][idx][0]),int(right_hand_point[0][idx][1])), 1, (hand_color[color_idx], 0, 0), 5)
        color_idx += 1


    # 显示图片
    cv2.imshow('', datum.cvOutputData)
    if cv2.waitKey(10) & 0xFF == 27:
        break

    # 计算并打印fps
    t1 = time.time()
    # fps = 1/(t1 - t0)
    # print('fps = {:.3f}'.format(fps))

# 保存图片
track_img = np.uint8(track_img)
cv2.imwrite('track_of_point.jpg', track_img)



################################################################################
################################################################################

# 取消注释下面这几行，就可以把图片送到classification中去做分类

from classification.classification_pose import Classification

# 将BGR转成RGB
track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)

# 分类器
classification = Classification()

# 传到classification中去分类
track_img = Image.fromarray(track_img)
class_name = classification.detect_image(track_img)
print(class_name)