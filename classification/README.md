## Classification：分类模型在Pytorch当中的实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [文件下载 Download](#文件下载)
3. [预测步骤 How2predict](#预测步骤)
4. [训练步骤 How2train](#训练步骤)
5. [评估步骤 How2eval](#评估步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
pytorch == 1.2.0

## 文件下载
训练所需的预训练权重都可以在百度云下载。     
链接: https://pan.baidu.com/s/1KwH-O7EafSpfPXnCyX_DVw    
提取码: 2g5y  

训练所用的示例猫狗数据集也可以在百度云下载。   
链接: https://pan.baidu.com/s/1Jz1nafsrRcw2Hqso4c8i0g    
提取码: hmmx   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，model_data已经存在一个训练好的猫狗模型mobilenet025_catvsdog.h5，运行predict.py，输入  
```python
img/cat.jpg
```
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在classification.py文件里面，在如下部分修改model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
```python
_defaults = {
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    "input_shape"   : [224,224,3],
    "backbone"      : 'mobilenet',
    "cuda"          : True
}
```
3. 运行predict.py，输入  
```python
img/cat.jpg
```  

## 训练步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片。  
2. 在训练之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_train.txt，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后修改model_data文件夹下的cls_classes.txt，使其也对应自己需要分的类。  
5. 在train.py里面调整自己要选择的网络和权重后，就可以开始训练了！  

## 评估步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片，在评估的时候，我们使用的是test文件夹里面的图片。  
2. 在评估之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成评估所需的cls_test.txt，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后在classification.py文件里面修改如下部分model_path、classes_path、backbone和alpha使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类，backbone对应使用的主干特征提取网络，alpha是当使用mobilenet的alpha值**。  
```python
_defaults = {
    "model_path"    : 'model_data/mobilenet_catvsdog.pth',
    "classes_path"  : 'model_data/cls_classes.txt',
    "input_shape"   : [224,224,3],
    "backbone"      : 'mobilenet',
    "cuda"          : True
}
```
5. 运行eval_top1.py和eval_top5.py来进行模型准确率评估。

## Reference
https://github.com/keras-team/keras-applications   

