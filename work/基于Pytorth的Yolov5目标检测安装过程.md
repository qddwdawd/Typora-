  # 基于Pytorth的Yolov5目标检测安装过程

（先检测你的GPU是否能够正常使用）

[yolov5目标检测安装](![img](file:///C:\Users\DELL\AppData\Roaming\Tencent\QQTempSys\%W@GJ$ACOF(TYDYECOKVDYB.png)https://blog.csdn.net/qq_44697805/article/details/107702939?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165209755016782248536713%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=165209755016782248536713&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-107702939-null-null.142^v9^control,157^v4^control&utm_term=%E5%8F%B2%E4%B8%8A%E6%9C%80%E8%AF%A6%E7%BB%86yolov5%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE%E6%90%AD%E5%BB%BA%2B%E9%85%8D%E7%BD%AE%E6%89%80%E9%9C%80%E6%96%87%E4%BB%B6%C2%A0%E5%8E%9F%E5%88%9B&spm=1018.2226.3001.4187)

#### 1.首先是对Pycharm的配置

Pycharm帮助用户在使用Python语言开发时提高其效率的工具，详细安装请去CSDN或者百度进行查询，这里不再啰嗦。

#### 2.配置Visual Studio C++环境

这个需要进行详细的安装过程，详见B站。

#### 3.配置CUDA和cudnn

这个需要注意的是，必须从官网下载，并且可能加载有些慢，需要耐心等待，同时两者的版本**必须**是相同的，否则会出现问题，当下载cudnn时，会让你注册一个英伟达的账号密码，这个注册就好了。

##### 3.1CUDA

当安装CUDA时，你可以另外从E盘新建两个文件夹，这样的好处是可以节省C盘的空间，当然你的文件名称是要和默认的文件名相同的（C盘），这个可以从B站看下载视频。

##### 3.2cudnn

当安装好CUDA时，他是一个文件类型，这时，你需要将文件下的几个小文件复制到CUDA相应文件中，这个跟着视频来。

#### 4.更改环境变量

这个我觉得应该是比较重要，因为我第一次安装时稀里糊涂，没有注意这个问题，第二次也没太注意，后来改了环境变量还是会报错，但是后来经过修改一些内容就好了，可能是某些问题吧，这个还是需要进行修改的，详细看教程。（可能和直接删除本机中原有的CUDA版本有联系，你也可以直接删除本机中的CUDA，然后进行安装新版的CUDA和cudnn这样可能就不会出现这个问题）

#### 5.进行Pytorch的安装

这个可以直接从网上搜到，注意你的CUDA版本和Windows下安装，将指令输入到cmd中，进行安装。

#### 6.创建名为Pytorch的环境

在cmd中输入：

(base) C:\Users\Administrator>conda create -n pytorch

在创建环境之后，在cmd中输入conda activate pytorch进入这个环境，你需要进行的是重新安装一些包，比如matplotlib，script，numpy，pandas等等这些，因为你创建的是另一个虚拟环境，并不是你的bace。

#### 7.在Pycharm中编辑解释器

你需要在此项目文件中重新添加解释器，选择conda环境，选择现有环境。然后选择在anaconda目录下的envs选择Pytorch中的python.exe，最后选择可用于所有文件，然后保存。

#### 8.下载Yolov5目标检测

这个在github（先注册账号）上下载Yolov5的文件（注意选择5.0版本），这个是现成的文件，可以用来进行目标检测。需要注意的是下载的Yolov5检测方法能够直接对80中物件进行检测，并且给出检测结果，因为作者已经从CoCo数据集中导入了数据并进行了训练，运行时它可以直接从网上下载模型，当你所要进行检测的东西并不在80个数据集中，你需要自己进行照片的收集，进行打标操作，进行自己的模型训练，然后再进行目标检测。

#### 9.Yolov5中出现的问题

在安装完之后可能会运行不了，因为代码本身有一些漏洞，所一要进行修改：

1. 可能会出现：

~~~python
Cant get attribute SPPF on module models#错误
~~~

- 解决方法：

~~~python
import warnings

class SPPF(nn.Module):
  # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
  def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
    super().__init__()
    c_ = c1 // 2  # hidden channels
    self.cv1 = Conv(c1, c_, 1, 1)
    self.cv2 = Conv(c_ * 4, c2, 1, 1)
    self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

  def forward(self, x):
    x = self.cv1(x)
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
      y1 = self.m(x)
      y2 = self.m(y1)
      return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
~~~

2. 可能会出现现可以运行但是结果并没有标注：

- 可能是因为主代码再第53行：

~~~python
vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
~~~

在会后一行末尾加上：

~~~python
 cudnn.benchmark = True 
~~~

