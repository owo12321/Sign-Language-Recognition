# 手语识别
---
## 0、使用到的模型
(1). openpose，作者：CMU-Perceptual-Computing-Lab
```
https://github.com/CMU-Perceptual-Computing-Lab/openpose
```
(2). 图像分类classification，作者：Bubbliiiing
```
https://github.com/bubbliiiing/classification-pytorch

B站对应视频：https://www.bilibili.com/video/BV143411B7wg
```
(3). 手语教学视频，作者：二碳碳
```
https://www.bilibili.com/video/BV1XE41137LV
```
（感谢大佬们的开源项目和教程，都已star加三连）

<br>
<br>

---

## 1、大致思路
方法一： 将视频输入到openpose中，检测出关节点的变化轨迹，将轨迹绘制在一张图片上，把这张图片传到图像分类网络中检测属于哪个动作  
```
视频  ----->  |  openpose  |-----> 关节点运动轨迹图-------> |  图像分类模型  | ----------> 单词分类  
```
方法二： 将视频输入到openpose中，检测出每一帧中关节点的位置，将多帧进行堆叠，形成一个三维张量，其中两个维度是图片的宽和高，一个维度是时间，然后对这个三维张量使用三维卷积进行训练和预测  
```
视频  ----->  |  openpose  |-----> 多张关节点位置图 ---------> |  堆叠  | --------> 三维张量 -------> |  三维卷积网络  | ----------> 单词分类  
```

<br>
<br>

---

## 2、环境配置
python：3.7（其他版本会导致openpose无法运行，建议使用anaconda的python环境）  
cuda：10  
cudnn：7或8应该都行  
（配置cuda和cudnn会比较麻烦，如果实在不想配，你可以去openpose的github网站下载使用cpu的版本，这里这个版本应该不支持cpu）

<br>

具体的配置环境方式：  
### (0).python和cuda和cudnn自己装
<br>

### (1).下载文件
下载代码文件后，再从网盘下载模型和数据文件（没有这些跑不起来），网盘链接：
```
链接：https://pan.baidu.com/s/1Q2aVVhMhSfWL4qKS9QslkQ 
提取码：abcd 
```
将从github下载的文件夹和网盘下载的文件夹合并，然后就可以下一步了。
（当然你大可直接找我要u盘拿完整的文件）

<br>

### (2).安装requirements.txt中的库  
cmd进入环境后，cd到项目文件夹下，执行指令：  
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
<br>

### (3).安装torch和torchvision  先下载好torch（1.2.0）和torchvision（0.4.0）的whl文件，下载地址：
```
链接：https://pan.baidu.com/s/1QIuJfEE5qQFpXY8ZlHeLNQ 
提取码：abcd 

（当然你依旧可直接找我拿u盘）
```
&emsp;&emsp;&emsp;&emsp;下载好torch和torchvision的whl文件后，cmd进入环境，cd到下载文件夹下，执行指令：
```
pip install [torch或torchvision的whl文件的文件名]

（先装torch再装torchvision，不然有可能会报错）
```

<br>
<br>

---

## 3、测试运行openpose
项目文件夹下有三个文件：
```
test.py
test_video.py
test_video_track_point.py
```
分别对应openpose的功能：检测图片、检测视频、检测视频并绘制关节点轨迹  
具体的使用方法可以看文件中的注释部分

<br>

在test_video_track_point.py中，取消掉最后几行的注释，就可以将绘制的轨迹图送到classification中去做分类检测  
（不过现阶段分类器尚未做好）

<br>
<br>

---

## 4、classification的训练和使用
可以看下classification文件夹中的README.md文件，大佬已经在里边讲得很详细了

