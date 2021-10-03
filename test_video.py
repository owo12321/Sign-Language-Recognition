# 此文件用来进行视频骨架检测的测试


import bin.pyopenpose as op   # 注意要3.7才能正确import，版本与文件名对应
import matplotlib.pyplot as plt
import cv2
import time

# 视频路径，若要打开摄像头，设为0
filePath = 'examples/media/test.mp4'

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

fps = 0
while True:
    t0 = time.time()
    # 截取一帧图片
    flag, frame = cap.read()
    if flag == False:
        break
    # 将图片到加载openpose的运行时里
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # 打印出点的位置
    # datum.poseKeypoints的类型是ndarray
    # print("Body keypoints: \n" + str(datum.poseKeypoints))


    # 显示图片
    cv2.imshow('', datum.cvOutputData)
    if cv2.waitKey(10) & 0xFF == 27:
        break

    # 计算并打印fps
    t1 = time.time()
    fps = 1/(t1 - t0)
    print('fps = {:.3f}'.format(fps))
