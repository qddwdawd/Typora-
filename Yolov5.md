#  Yolov5

~~~python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
~~~

** default为初始值（默认值）**

#### weights:

当训练图像的所有类个数不相同时,我们可以更改类权重, 即而达到更改图像权重的目的.然后根据图像权重新采集数据，这在图像类别不均衡的数据下尤其重要。
使用yolov5训练自己的数据集时，各类别的标签数量难免存在不平衡的问题，在训练过程中为了就减小类别不平衡问题的影响，yolov5中引入了类别权重和图像权重的设置。
dafault = ‘yolov5m.pt'是训练好的网络模型，可以更改，模型从下方下载：

![image-20220711205339181](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220711205339181.png)

，可以提前下载复制到项目中

![image-20220711205515733](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220711205515733.png)

![image-20220711205602004](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220711205602004.png)

~~~python
  parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
~~~

- source给网络模型指定一个输入，当指定为文件夹时，他会把文件夹下所有的文件进行检测并保存起来。

也可以值检测一张图片，直接输入地址即可：

![image-20220711210233768](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220711210233768.png)

也可以检测视频：

![image-20220711210321447](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220711210321447.png)

最好是英文，中文解析可能出现问题

~~~python
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#输入和输出过程中大小是一样的，而在训练的过程中将图片进行缩放，再写入结果的时候将成品按比例放大。
~~~

~~~python
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    #置信度大于0.25才会显示出来这是什么
~~~

![image-20220712092942908](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712092942908.png)

当置信度设为0的时候，如上图所示，但凡是有一点相似的都会检检测

~~~python
 parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
~~~

![image-20220712093346591](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712093346591.png)

- 选取最优的检测框，如上图所示一个人会出现三个检测框架。
- iou![image-20220712093538872](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712093538872.png)

IoU计算公式是两个框架交集部分/并集部分

![image-20220712093645462](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712093645462.png)



当计算结果大于设定值（default=0.45），则选取最优秀的框架，当Iou小于阈值时，不会进行挑选，最用就是避免重复。

当设为0的时候，框和框不会有交际部分，当设为1的时候，将会由许多重复的框架。

~~~python
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#指定你的运行设备。
~~~

## 当参数没有dafault时：

~~~python
 parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
~~~

**可以看到以上参数并没有dafault，他的意思是运行时将不会自动运行这些代码，需要手动操作，操作过程如下**

~~~python
  parser.add_argument('--view-img', action='store_true', help='display results')#的意思是先不显示结果
~~~

![image-20220712094936393](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712094936393.png)

点击编辑配置，

![image-20220712095017525](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712095017525.png)

在形参上输入这一行代码就可以实时显示。

![image-20220712095053235](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712095053235.png)

~~~python
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')#将结果保存成txt形式
~~~

![image-20220712095610130](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712095610130.png)

标注一些结果，还可以保存一些置信度。

~~~python
 parser.add_argument('--nosave', action='store_true', help='do not save images/videos')#看help解释：不要保存图片或者Videos，所以这个方法基本不会使用。
~~~

~~~python
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') #在检测过程中，对于每个目标都有一个分类，比如说人是class 0 ，在txt文本中有显示：
~~~

![image-20220712100227817](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712100227817.png)

人就是class 0 ，如果你只想在结果中检测或者显示人的检测，只需要输入参数：

![image-20220712100335448](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712100335448.png)

~~~python
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#增强的nms，将他设置之后nms会更加强大。
~~~

~~~python
 parser.add_argument('--augment', action='store_true', help='augmented inference')#也是增强检测，提升结果的方式
~~~

![image-20220712100828085](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712100828085.png)

![image-20220712100951935](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712100951935.png)

将会增强检测，好处是将正确的检测结果进行增强，但是缺点就是如上图所示，可能会增强一些错误的结果。

~~~python
 parser.add_argument('--update', action='store_true', help='update all models')#将网络模型中的一些不必要的部分，比如优化器之类的都会去掉，基本用不到。
~~~

![image-20220712101428362](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712101428362.png)

![image-20220712101629732](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712101629732.png)

- 如上图所示，这个参数是将一些不必要的优化器等等训练结果进行设置为None

~~~python
 parser.add_argument('--project', default='runs/detect', help='save results to project/name') #将结果保存到什么位置。dafult可以更改保存路径
~~~

~~~python
 parser.add_argument('--name', default='exp', help='save results to project/name')#保存结果的名字。
~~~

~~~python
 parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')  #大家会看到我每一次运行他都会重新创建一次文件夹，而当我应用他的时候，结果会保存在同一个文件夹里，且由name的默认值default来决定。
~~~

![image-20220712102746737](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712102746737.png)

最后Debug一下，先断点进行查看参数：

![image-20220712103126562](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712103126562.png)

## 3.如何训练YoLov5神经网络

本地上训练Yolov5

利用云端GPU训练Yolov5 

![image-20220712170615893](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712170615893.png)

一定要注意打开哪个项目。

在训练过程中，需要下载coco训练集，注意放在与文件夹yolov5同目录（运行下载然后解压就行）

![image-20220712170725769](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712170725769.png)

由于数据集太多，进行删减进行训练，得到的结果在exp中

![image-20220712170922699](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712170922699.png)

best是最好的网络模型，last是最后的训练模型

还有其他的内容：

![image-20220712171019942](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171019942.png)

![image-20220712171106808](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171106808.png)

labels.jpg是标注的一些分布

![image-20220712171143967](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171143967.png)

labels_correlogram.jpg是标注的一些相关矩阵

![image-20220712171358120](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171358120.png)results.txt是相关的一些结果记录

![image-20220712171443715](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171443715.png)

opt是设置的参数

![image-20220712171527675](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712171527675.png)

train_batch1.jpg是一些训练图片



### 参数讲解

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')#指定一个训练好的模型的路径，模型在网上会自动下载下来
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')#
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')#指定默认训练集在coco128.yaml这种文件中，会告诉你训练集的下载地址以及一些类名，可以任意选择
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')#超参数，hyp
    parser.add_argument('--epochs', type=int, default=300)#训练多少轮
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')#
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')#你的训练集的图片大小
    parser.add_argument('--rect', action='store_true', help='rectangular training')#矩阵的训练方式
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')#在最近的训练模型基础之上进行训练
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')#保存最后一次训练的模型的一些权重数据就是pt文件
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    #是否只在最后一个epoch进行测试
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')#是否采用锚点，模型可以分为有锚点和没锚点
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    #默认没有开启
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')#通过这个下载谷歌云盘上的东西
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')#是否将图片进行缓存加快训练速度
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')#从上一轮测试过程中对于测试图片部分不好，在下一轮进行权重的增加
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#选择cpu还是Gpu
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')#将图片尺寸进行变换
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')#训练的数据集是单类表还是多类
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')#优化器，不使用会使随机梯度下降
    parser.add_argument('--synac-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')#如果有多个GPU可以启用
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')#不能随便改，在run的train下边
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')#
    parser.add_argument('--entity', default=None, help='W&B entity')#库对应的东西，不用管
    parser.add_argument('--name', default='exp', help='save to project/name')#保存文件名
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')#如果设置就只会保存到exp文件
    parser.add_argument('--quad', action='store_true', help='quad dataloader')#数据加载，它可以在一个对于更小的相片尺寸来进行训练
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')#对随机数进行调整,改变处理方式
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')#标签平缓，防止在分类算法中的过拟合
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')#上传数据集
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')#对模型打一些日志，没有装wand就设置为-1
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')#没有什么的作用
    opt = parser.parse_args()
```





~~~python
 parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')#模型的一些配置，在yolov5l.yaml这种文件当中
~~~

![image-20220712172713964](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712172713964.png)

~~~python
  parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')#超参数，scratch是开始检测的文件，而finetune是微调的文件
~~~

![image-20220712182811271](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712182811271.png)

~~~python
 parser.add_argument('--rect', action='store_true', help='rectangular training')#矩阵的训练方式
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/WX%5BK0%5D85~XE_S%7DJ%60NOENLGF.png)

只需要对补分进行填充，减少了一些不必要的信息，加速推理过程。



~~~python
 parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')#在最近的训练模型基础之上进行训练
~~~

![image-20220712184422187](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220712184422187.png)

## 4.如何制作和训练自己的数据集

- 标注

- 自己获得数据集（手动）-人工标注

- 自己获得数据集—半人工标注

- 仿真数据集（GAN,数字图像处理方式）

需要指定数据集根目录，

![image-20220713175533075](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220713175533075.png)

![image-20220713184044834](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220713184044834.png)

![image-20220713184308704](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220713184308704.png)



![image-20220713184423356](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220713184423356.png)

指定训练路径，更改yaml文件：

![image-20220715160130203](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160130203.png)

![image-20220715160143181](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160143181.png)

进行训练(也可以创建（实验集）得到：

![image-20220715160301665](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160301665.png)

将此文件放在detect中：

![image-20220715160334786](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160334786.png)

前两行更改即可。



## YoloV5系列解析

![image-20220714141358566](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220714141358566.png)

![image-20220714141509227](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220714141509227.png)

![image-20220714141652175](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220714141652175.png)



## YOLOV5 deepsort

deep_sort是追踪端

models是目标检测端

deepsort的目的是让目标检测更加平滑

![image-20220715151309483](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715151309483.png)

deep部分是特征提取的网络，核心是train和model

model中的：

![image-20220715151524656](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715151524656.png)

 是将每一个检测框输入到net中，经过y = net（x）提取一个特征，y就得到了特征，然后enaluate.py是对网络精度的验证，feature_extractor.py就是对model进行了包装，用了__call__魔术方法

![image-20220715152004452](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715152004452.png)

### deep的工作流程：

训练自己的数据集，首先需要准备数据，并且与Yolov5的训练数据集不同，这个数据集是将检测目标抠出来，放到一起，进行训练之后会通过deep中的其他文件得到训练后的网络模型，并且会以ckpt.t7的形式保存在checkpoint中。

### sort的工作流程：

![image-20220715160522427](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160522427.png)

![image-20220715160531091](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715160531091.png)

