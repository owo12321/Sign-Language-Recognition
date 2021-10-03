# 此文件用来进行图片骨架检测的测试

import bin.pyopenpose as op   # 注意要3.7才能正确import
import matplotlib.pyplot as plt
import cv2
import numpy as np

filePath = 'examples/media/COCO_val2014_000000000395.jpg'

# 找到模型文件
params = dict()
params["model_folder"] = "models/"
params["net_resolution"] = "256x256"    # 大小要设置成1：1的，且是2的倍数，1660ti超过256就爆显存了
params["hand"] = True

# opWrapper用来启动openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 加载图片到openpose的运行时里
imageToProcess = cv2.imread(filePath)
datum = op.Datum()
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# 打印出点的位置
# datum.poseKeypoints的类型是ndarray
print("Body keypoints: \n" + str(datum.poseKeypoints))
print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

# 显示图片
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
