# Qt上位机

主窗口类型

有3种窗口

QMainWindow

QWidget

QDialog

QMainWindow :可以包含菜单栏，工具栏，状态栏和标题栏，是最常见的窗口形式。

QDialog：是对话窗口的基类。没有菜单栏，工具栏，状态栏。

QWidget：不确定窗口的用途，就使用QWidget。

~~~python
import sys  #用里边一个获取参数的api
from PyQt5.QtWidgets import QMainWindow,QApplication #用来创建窗口，以及创建任何应用程序都比与要用QApplication
from PyQt5.QtGui import QIcon

class FirstMainWin(QMainWindow):#从QMainWindow主窗口继承
    def __init__(self,parent = None):
        super(FirstMainWin,self).__init__(parent)
        #设置主窗口的标题
        self.setWindowTitle('第一个主窗口应用')
        #设置主窗口的尺寸
        self.resize(400,300)
        self.status = self.statusBar()
        self.status.showMessage('只存在5秒的消息',5000)

if __name__== '__main__':
    app = QApplication(sys.argv)
    main = FirstMainWin()
    main.show()
    sys.exit(app.exec_())
~~~

### 如何关闭窗口

~~~python
import sys
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QApplication, QWidget,QPushButton


class QuitApplication(QMainWindow):
    def __init__(self):
        super(QuitApplication, self).__init__()
        self.resize(300, 120)
        self.setWindowTitle('退出应用程序')
        
        # 添加Button
        self.button1 = QPushButton('退出应用程序')
        self.button1.clicked.connect(self.onClick_Button)
        
        layout = QHBoxLayout()#水平布局
        layout.addWidget(self.button1)
        
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)
        
        # 按钮单击事件的方法

    def onClick_Button(self):
        sender = self.sender()  # sender获得发送消息的对象
        print(sender.text() + '按钮被按下')  # 输出Button的文本
        app = QApplication.instance()  # 获取app单例对象，也就是当前对象
        app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QuitApplication()
    main.show()
    sys.exit(app.exec_())

    
~~~

一个简单的控件连接案例

~~~python
import sys
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QApplication, QWidget,QPushButton
from Setting import Ui_Setting
from PyQt5 import QtCore, QtGui


class QuitApplication(QMainWindow):
    def __init__(self):
        super(QuitApplication, self).__init__()
        self.resize(300, 120)
        self.setWindowTitle('退出应用程序')
        # 添加Button
        self.button1 = QPushButton()
        self.button1.clicked.connect(self.SettingPBt_ClickEvent)
        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)
        # 按钮单击事件的方法
        self.button2 = QPushButton('shezhi')  # self.LoginWgt， # 添加Button
        self.button2.clicked.connect(self.SettingPBt_ClickEvent)
        layout = QHBoxLayout()
        layout.addWidget(self.button2)
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.SettingMW = QMainWindow()
        self.ui_Setting = Ui_Setting()
        self.ui_Setting.setupUi(self.SettingMW)
        self.button1.setMinimumSize(QtCore.QSize(40, 40))  # 设置按钮大小
        self.button1.setMaximumSize(QtCore.QSize(40, 40))

        self.button1.setStyleSheet("QPushButton{"
                                      "}\n"
                                      "/*按钮普通态*/\n"
                                      "QPushButton\n"
                                      "{\n"
                                      "    /*字体为微软雅黑*/\n"
                                      "    font-family:Microsoft Yahei;\n"
                                      "    /*字体大小为20点*/\n"
                                      "    font-size:12pt;\n"
                                      "    /*字体颜色为白色*/    \n"
                                      "    color:white;\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(67,67,67);\n"
                                      "    /*边框圆角半径为8像素*/ \n"
                                      "    border-radius:7px;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:hover\n"
                                      "{\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(90, 90, 90);\n"
                                      "}\n"
                                      "\n"
                                      "/*按钮按下态*/\n"
                                      "QPushButton:pressed\n"
                                      "{\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(14 , 135 , 228);\n"
                                      "    /*左内边距为3像素，让按下时字向右移动3像素*/  \n"
                                      "    padding-left:3px;\n"
                                      "    /*上内边距为3像素，让按下时字向下移动3像素*/  \n"
                                      "    padding-top:3px;\n"
                                      "}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("png/3E9]I~U815DD3{7OQXM]8JF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button1.setIcon(icon)
        self.button1.setIconSize(QtCore.QSize(40, 40))

    def SettingPBt_ClickEvent(self):
        if self.SettingMW.isVisible():#如果设置窗口可见，第一次应该返回false
            return self.SettingMW.close() and self.SettingMW.show()
        else:
            return self.SettingMW.show()
    def onClick_Button(self):
        sender = self.sender()  # sender获得发送消息的对象
        print(sender.text() + '按钮被按下')  # 输出Button的文本
        app = QApplication.instance()  # 获取app单例对象，也就是当前对象
        app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QuitApplication()
    main.show()
    sys.exit(app.exec_())

~~~

## Qt实践

~~~python
from Mainwindow import Ui_MainWindow  #主窗口
from Login import Ui_LoginWindowWgt
from CameraPopupWindow import Ui_CameraPopupWindow #
from Setting import Ui_Setting  #设置窗口
from AutoPhoto import Ui_AutoPhoto #连续自动拍照
from AutoPhotoSetting import Ui_AutoPhotoSetting#连续自动拍照的设置时间
from HIKInfoWgt import Ui_HIKInfoWgt
from SensorInfoWgt import Ui_SensorInfoWgt  #传感器设置
import configure as cfg#configure是一个脚本，一般由Autoconf工具生成，它会检验当前的系统环境，看是否满足安装软件所必需的条件：比如当前系统是否支持待安装软件，是否已经安装软件依赖等。configure脚本最后会生成一个Makefile文件。
import HIKconfigure as HIKcfg#有关配置的操作，他是配置文件
import Sensorconfigure as Sensorcfg#有关配置的操作
import serial#串口通信
import struct#结构体
import binascii#binascii模块包含很多在二进制和ASCII编码的二进制表示转换的方法
import math
import pyqtgraph as pg#实施绘制数据
from PyQt5 import QtCore, QtGui, QtWidgets,QtMultimedia,QtMultimediaWidgets#视频播放器
import sys#该模块提供对解释器使用或维护的一些变量的访问和获取
import os#对目录和文件的一般常用操作
import shutil #提供了复制、移动、删除、压缩、解压等操作，这些 os 模块中一般是没有提供的
import time
import random
import cv2 as cv
import cv2
import qimage2ndarray#QImage 转 numpy.ndarray，QImage类是设备无关的图像
from PIL import Image, ImageDraw, ImageFont  #图片操作所
import imageio #极简化的图像数据读写库
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import torch
import warnings
import argparse#，可以用来方便地读取命令行参数
from utils.datasets import LoadStreams, LoadImages
from utils.draw import draw_boxes
from utils.general import check_img_size
from utils.torch_utils import time_synchronized
from shrimp_detect_yolov5 import Person_detect
from deep_sort import build_tracker
from utils.parser import get_config
from utils.log import get_logger
from collections import Counter#Counter是一个简单的计数器，例如，统计字符出现的个数。
from collections import deque#类似于list的容器，可以快速的在队列头部和尾部添加、删除元素
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']  
plt.rcParams['axes.unicode_minus']=False
~~~

```
# 在主程序中，需要根据UI对应类以及QtWidgets派生一个新类，在该新类中实现所有槽函数的代码。
# 关于派生的新类请注意：
# 1、一定要有两个基类，一个是UI界面窗口的窗口类，一个是UI类本身；
# 2、一定要实现新类的构造方法，并在构造方法中调用父类的构造方法；
# 3、新类的构造方法中要调用self.setupUi(self) ，setupUi为PyUIC生成的UI类图形界面初始化的重要函数
```

~~~python
class MainProgram(QtWidgets.QMainWindow,Ui_MainWindow):#派生一个新类
    def __del__(self):
        try:
            self.camera.release()
        except:
            return None
    def __init__(self,parent=None):#新类构造函数
        super(MainProgram,self).__init__(parent)#调用父类构造函数
        self.setupUi(self) #进行图形界面初始化，必须有
        self.tabWidget.setCurrentIndex(0)#QTabWidget有个setCurrentIndex槽，可用于修改当前活动标签页,默认展示第一页
~~~

![image-20220724201735987](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20220724201735987.png)

构造完之后还会有好多的子窗口，比如说设置，海康摄像头，传感器等等这些，都需要进行以上类似的操作，但由于是弹窗之类的子窗口，所以直接进行：

~~~python
        self.CameraPopupWindowMW = QtWidgets.QMainWindow()
        self.ui_CameraPopupWindow = Ui_CameraPopupWindow()
        self.ui_CameraPopupWindow.setupUi(self.CameraPopupWindowMW)
        
        self.SettingMW = QtWidgets.QMainWindow()
        self.ui_Setting = Ui_Setting()
        self.ui_Setting.setupUi(self.SettingMW)
               self.SettingMW.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint)#setWindowFlags设置窗口属性，为Windows系统上的窗口装饰一个窄的对话框边框，通常这个提示用于固定大小的对话框
        self.HIKInfoWgt = QtWidgets.QWidget()
        self.ui_HIKInfoWgt = Ui_HIKInfoWgt()
        self.ui_HIKInfoWgt.setupUi(self.HIKInfoWgt)
        self.SensorInfoWgt = QtWidgets.QWidget()
        self.ui_SensorInfoWgt = Ui_SensorInfoWgt()
        self.ui_SensorInfoWgt.setupUi(self.SensorInfoWgt)

        self.UserLoginWindow = QtWidgets.QWidget()
        self.ui_UserLoginWindow =Ui_LoginWindowWgt()
        self.ui_UserLoginWindow.setupUi(self.UserLoginWindow)

        self.AutoPhotoMW = QtWidgets.QMainWindow()
        self.ui_AutoPhoto = Ui_AutoPhoto()
        self.ui_AutoPhoto.setupUi(self.AutoPhotoMW)
        self.AutoPhotoMW.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint) 
        #self.AutoPhotoMW.show()
        
        self.AutoPhotoSettingMW = QtWidgets.QMainWindow()
        self.ui_AutoPhotoSetting = Ui_AutoPhotoSetting()
        self.ui_AutoPhotoSetting.setupUi(self.AutoPhotoSettingMW)
        self.AutoPhotoSettingMW.setWindowFlags(QtCore.Qt.MSWindowsFixedSizeDialogHint) 
        self.VideoPlayer_Wgt('','',False,False)
~~~

注意：当用到子窗口的控件时，直接采用 self.ui_CameraPopupWindow.的形式，而当弹出窗口时，用self.CameraPopupWindowMW.show()。

~~~python
 def SettingPBt_ClickEvent(self):
        if self.SettingMW.isVisible():#如果设置窗口可见，第一次应该返回的是false
            return self.SettingMW.close() and self.SettingMW.show()
        else:
            return self.SettingMW.show()
~~~

### 实例化完成之后就i可以配置本地摄像机

~~~python
        self.config = cfg.config_ini() #初始化配置本地相机
        self.HIKconfig = HIKcfg.config_ini()#初始化配置海康摄像头，包括写入IP地址，用户名及密码
        self.Sensorconfig = Sensorcfg.config_ini()#初始化配置传感器数据
        self.ReadConfig()#创建本地摄像头配置
        self.IdentifyPath=self.ui_Setting.IdentifiedFilePathLE.text()
        self.HIKReadConfig()#创建海康摄像头配置
        self.SensorReadConfig()#创建传感器配置
        self.GetDesktopResolutionRatio()##获取显示器分辨率大小
~~~

这里有configure.py文件，它是对用户信息的一个管理，同时也可以用来进行摄像头初始化以及保存信息（HIKconfig ）传感器（Sensorconfig）。

打开之后我们会看到：

![image-20220724204022070](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204022070.png)

它return的是creat_config()，这个文件是用户信息保存的文件：

![image-20220724204121861](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204121861.png)

当他返回creat_config：

![image-20220724204155382](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204155382.png)

是一个创建配置的函数，我们可以自定义config信息，在后续保存操作就可以让我们的信息保存到这里。

~~~python
self.ReadConfig()
~~~

![image-20220724204317992](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204317992.png)

首先定义self.来等于config中的参数，进行参数映射，而后

![image-20220724204428318](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204428318.png)

```python
self.ui_Setting.CameraSettingLE.setText#这是设置当在QLineEdit中写入信息
```

最后是对海康摄像头的一个操作：

![image-20220724204728701](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724204728701.png)

设立Flag能够更加灵活的进行true以及False的选项，让我们能够对海康摄像头进行清晰的处理。

```
self.EmbeddedCameraFlag = False
self.CameraDisplayWindowOpening = False
self.CaptureFlag = False
self.DegreeFlag = False
self.HIKCheckBoxFlag=False
self.AutoPhotoFlag = False
self.HIKCameraInitFlag=False
self.loginFlag = False
self.imagenumber = 0
self.Degree = 0
```

self.SensorCOM=self.ui_SensorInfoWgt.SensorCOMLE.text()  #传感器
self.SensorDelay=float(self.ui_SensorInfoWgt.SensorDelayLE.text())#传感器
self.SensorBps=float(self.ui_SensorInfoWgt.SensorBpsLE.text())#传感器设置
self.cfg=get_config()#获取配置信息，读取配置文件中的配置参数，
self.logger=get_logger("root")#记录日志信息？

~~~python
self.PrepCameraSettingIndex()#这个好像是准备工作也要用
    def PrepCameraSettingIndex(self):
        self.PrepCamera()
        
        self.VideoRecordFlag = False

        self.Image_num = 0
        self.R = 1
        self.G = 1
        self.B = 1

        self.SetExposure()
~~~

```python
if self.ui_Setting.HIKCameraCheckBox.isChecked():
    self.url="rtsp://{}:{}@{}/Streaming/Channels/1".format(self.HIKUsername,self.HIKPasswd,self.HIKCameraIP)
    self.CameraSetting=self.url
    self.ui_Setting.CameraSettingLE.setText(self.ui_HIKInfoWgt.HIKCameraIPLE.text())
self.ui_Setting.SensorSettingLE.setText("COM:{};Timeout:{};Bps:{}".format(self.ui_SensorInfoWgt.SensorCOMLE.text(),self.ui_SensorInfoWgt.SensorDelayLE.text(),self.ui_SensorInfoWgt.SensorBpsLE.text()))
```

接下来是对定时器的解释，

~~~python
 self.TimerState={}
        self.Timer={}
        
        self.CameraTimer = QtCore.QTimer()#定时器
        self.CameraTimer.timeout.connect(self.CameraAdjustProcess)##图像框展示的核心部分，对索取图片进行展示，在startcamera中有较好的表现
        self.Timer['CameraTimer']=self.CameraTimer
        self.TimerState['CameraTimer']=self.CameraTimer.isActive()#如果定时器正在运行，返回真，否则返回假
        
        self.AutoPhotoTimer = QtCore.QTimer()
        self.AutoPhotoTimer.timeout.connect(self.AutoPhoto)
        self.Timer['AutoPhotoTimer']=self.AutoPhotoTimer
        self.TimerState['AutoPhotoTimer']=self.AutoPhotoTimer.isActive()
        
        self.WindowTimer = QtCore.QTimer()
        self.WindowTimer.timeout.connect(self.ControlWindows)        
        self.WindowTimer.start(0)#启动定时器
        self.Timer['WindowTimer']=self.WindowTimer
        self.TimerState['WindowTimer']=self.WindowTimer.isActive()

        self.LoginWidgetTimer = QtCore.QTimer()
        self.LoginWidgetTimer.start(0) #函数初始化的时候就要启动计def OtherParts(self):
        self.Timer['LoginWidgetTimer']=self.LoginWidgetTimer
        self.TimerState['LoginWidgetTimer']=self.LoginWidgetTimer.isActive()

        self.LoginWidgetTimer.timeout.connect(self.LoginWidget_Location)
                                                  
        self.VideoWgtTimer = QtCore.QTimer()
        self.VideoWgtTimer.timeout.connect(self.ControlVideoWgt)
        self.VideoWgtTimer.start(0)
        self.Timer['VideoWgtTimer']=self.VideoWgtTimer
        self.TimerState['VideoWgtTimer']=self.VideoWgtTimer.isActive()
        
        self.PhotoAutoRecognitionRBt_Controller_Timer = QtCore.QTimer()
        self.PhotoAutoRecognitionRBt_Controller_Timer.timeout.connect(self.PhotoAutoRecognitionRBt_Control)
        self.PhotoAutoRecognitionRBt_Controller_Timer.start(0)
        self.Timer['PhotoAutoRecognitionRBt_Controller_Timer']=self.PhotoAutoRecognitionRBt_Controller_Timer
        self.TimerState['PhotoAutoRecognitionRBt_Controller_Timer']=self.PhotoAutoRecognitionRBt_Controller_Timer.isActive()
                
        self.IdentifyTimer=QtCore.QTimer()
        self.IdentifyTimer.timeout.connect(self.IdentifyCameraAdjustProcess)
        self.Timer['IdentifyTimer']=self.IdentifyTimer
        self.TimerState['IdentifyTimer']=self.IdentifyTimer.isActive()
~~~

**定时器有自启以及超时启动，他们能够很好的进行窗口的操作问题，尤其是在选取文件时能够将关闭一些不必要的进程然后再开启**.

简单介绍两个：(图像框展示的核心部分，对索取图片进行展示，在startcamera中有较好的表现)：

~~~python
self.CameraTimer = QtCore.QTimer()#定时器
        self.CameraTimer.timeout.connect(self.CameraAdjustProcess)##图像框展示的核心部分，对索取图片进行展示，在startcamera中有较好的表现
        self.Timer['CameraTimer']=self.CameraTimer
        self.TimerState['CameraTimer']=self.CameraTimer.isActive()#如果定时器正在运行，返回真，否则返回假
~~~

~~~python
 def CameraAdjustProcess(self):
        if self.CameraPopupWindowMW.isVisible() == False:#这个是相机弹窗不可见
            self.CameraPopupWindowPBt.setText('打开相机弹窗')
            if self.CamaraPBt.text()=='开启摄像头':
                self.StopCamera()
        elif self.CameraPopupWindowMW.isVisible() == True:
            self.CameraPopupWindowPBt.setText('关闭相机弹窗')
        try:
            success,img = self.camera.read()
            #img = img[:,::-1,:]
            #print(success,img)
            if success:
                self.Image = self.ColorAdjust(img)#因为进行颜色调节，因为我们的BGR是滑动的
                #self.Image = img
                self.DispImg()#进行图像的处理(颜色通道转化)和展示
                self.Image_num+=1
    
                if self.Image_num%10==9:
                    frame_rate=10/(time.perf_counter()-self.timelb)
                    self.FmRateLCD.display(frame_rate)
                    self.timelb = time.perf_counter()
                    #size=img.shape
                    self.IMGWidthLCD.display(self.camera.get(3))##display是取值，self.camera.get()是cv。videocapture.get()du读取帧的宽度
                    self.IMGHeightLCD.display(self.camera.get(4))#
        except:
            pass
~~~

这个过程是故意设置在startcamral当中的，意义为当超时过后会执行图像的读取，用cv2读取的图像通道为BGR所以需要对图像进行处理：这里边也有一些勾选的判断

~~~python
    def DispImg(self):
        #IMG_Width = round(self.geometry().width()/2.5+125)
        #IMG_Height = round(self.geometry().height()/1.4)
        
        if self.GrayIMGCkB.isChecked():
            self.Image = cv.cvtColor(self.Image, cv.COLOR_BGR2GRAY)
        else:
            self.Image = cv.cvtColor(self.Image, cv.COLOR_BGR2RGB)
        if self.DegreeFlag == True:
            self.Image = DIY_ImgAddText(self.Image,'当前温度：{}℃'.format(self.Degree),round(self.Image.shape[1]*0.75),round(self.Image.shape[0]*0.1),(255, 0, 0), 40)
        if self.VideoRecordFlag:#在开始录像中是True
            img = cv.cvtColor(self.Image, cv.COLOR_RGB2BGR)#cv.cvtColor：一个颜色空间转换到另一个颜色空间的转换
            self.video_writer.write(img)
        #print()
        qimg = qimage2ndarray.array2qimage(self.Image)
        pix_img = QtGui.QPixmap(qimg)
        if self.EmbeddedCameraFlag == True:
            self.EmbeddedPictureLb.setPixmap(pix_img)
            self.EmbeddedPictureLb.show()
        if self.CameraDisplayWindowOpening == True:#
            self.ui_CameraPopupWindow.CameraPopupWindowLb.setPixmap(pix_img)
            self.ui_CameraPopupWindow.CameraPopupWindowLb.show()
~~~

**  self.ui_CameraPopupWindow.CameraPopupWindowLb.setPixmap(pix_img)
            self.ui_CameraPopupWindow.CameraPopupWindowLb.show()**是将最后的图片展示在Qlabel上。

#### 由于TabWidget的限制，我们自己又添加了两个设置：

~~~python
    def OtherParts(self):#创建一个窗体为tabwidget的子窗体，进行添加系统和用户
        self.LoginWgt = QtWidgets.QWidget(self.tabWidget)#这里的widget我们可以理解为子窗口，他永远指向父类，所以是一级一级向上指定，在这里可以是又创建了一个窗口，并且将系统和用户登录嵌入到tabWidget当中，与三个分页在同一行
        self.LoginWgt.setMaximumSize(QtCore.QSize(100, 300))
        self.LoginWgt.setObjectName("LoginWgt")
        self.LoginWgt.setMinimumSize(QtCore.QSize(350, 40))
        self.LoginWgt.setMaximumSize(QtCore.QSize(350, 40))
        self.LoginWgt.move(780,0)# MainWindow.geometry().width()-355，move的原点是父窗口的左上角，在一级一级的关系中，与他最接近的是LoginWgt，在这一窗口的左上角进行移动
        self.horizontalLayout_login = QtWidgets.QHBoxLayout(self.LoginWgt)#水平布局
        self.horizontalLayout_login.setContentsMargins(0, 0, 0, 0)#可以调整控件在布局中的位置，4个参数顺序是左上右下
        self.horizontalLayout_login.setSpacing(0)#表示各个控件之间的上下间距
        self.horizontalLayout_login.setObjectName("horizontalLayout_login")
        
        self.SettingPBt = QtWidgets.QPushButton(self.LoginWgt)#self.LoginWgt， # 添加Button
        self.SettingPBt.setMinimumSize(QtCore.QSize(40, 40))#设置按钮大小
        self.SettingPBt.setMaximumSize(QtCore.QSize(40, 40))

        self.SettingPBt.setStyleSheet("QPushButton{"
            "}\n"
"/*按钮普通态*/\n"
"QPushButton\n"
"{\n"
"    /*字体为微软雅黑*/\n"
"    font-family:Microsoft Yahei;\n"
"    /*字体大小为20点*/\n"
"    font-size:12pt;\n"
"    /*字体颜色为白色*/    \n"
"    color:white;\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(67,67,67);\n"
"    /*边框圆角半径为8像素*/ \n"
"    border-radius:7px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(90, 90, 90);\n"
"}\n"
"\n"
"/*按钮按下态*/\n"
"QPushButton:pressed\n"
"{\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(14 , 135 , 228);\n"
"    /*左内边距为3像素，让按下时字向右移动3像素*/  \n"
"    padding-left:3px;\n"
"    /*上内边距为3像素，让按下时字向下移动3像素*/  \n"
"    padding-top:3px;\n"
"}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Mainwindow_Images/设置.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SettingPBt.setIcon(icon)
        self.SettingPBt.setIconSize(QtCore.QSize(40, 40))        
        self.horizontalLayout_login.addWidget(self.SettingPBt)#添加按钮
        
        self.UserIMGLb = QtWidgets.QLabel(self.LoginWgt)
        self.UserIMGLb.setMinimumSize(QtCore.QSize(40, 40))
        self.UserIMGLb.setMaximumSize(QtCore.QSize(40, 40))
        self.UserIMGLb.setStyleSheet("QLabel{\n"
"    border-image: url(Mainwindow_Images/User.png);\n"
"}")
        self.UserIMGLb.setObjectName("UserIMGLb")
        self.horizontalLayout_login.addWidget(self.UserIMGLb)

        self.UserLb = QtWidgets.QLabel(self.LoginWgt)
        self.UserLb.setStyleSheet("font: 45 8pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);")
        self.UserLb.setMinimumSize(QtCore.QSize(150, 30))
        self.UserLb.setMaximumSize(QtCore.QSize(150, 30))
        self.UserLb.setObjectName("UserLb")
        self.UserLb.setText("用户您好，请在这里登陆！")
        self.horizontalLayout_login.addWidget(self.UserLb)

        self.UserLoginPBt = QtWidgets.QPushButton(self.LoginWgt)
        self.UserLoginPBt.setMinimumSize(QtCore.QSize(80, 30))
        self.UserLoginPBt.setMaximumSize(QtCore.QSize(80, 30))
        
        icon_login = QtGui.QIcon()
        icon_login.addPixmap(QtGui.QPixmap("Mainwindow_Images/登陆.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.UserLoginPBt.setIcon(icon_login)
        self.UserLoginPBt.setIconSize(QtCore.QSize(20, 20))
        self.UserLoginPBt.setStyleSheet("QPushButton{"
            "}\n"
"/*按钮普通态*/\n"
"QPushButton\n"
"{\n"
"    /*字体为微软雅黑*/\n"
"    font-family:Microsoft Yahei;\n"
"    /*字体大小为20点*/\n"
"    font-size:12pt;\n"
"    /*字体颜色为白色*/    \n"
"    color:white;\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(14 , 150 , 254);\n"
"    /*边框圆角半径为8像素*/ \n"
"    border-radius:7px;\n"
"}\n"
"\n"
"QPushButton:hover\n"
"{\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(44 , 137 , 255);\n"
"}\n"
"\n"
"/*按钮按下态*/\n"
"QPushButton:pressed\n"
"{\n"
"    /*背景颜色*/  \n"
"    background-color:rgb(14 , 135 , 228);\n"
"    /*左内边距为3像素，让按下时字向右移动3像素*/  \n"
"    padding-left:3px;\n"
"    /*上内边距为3像素，让按下时字向下移动3像素*/  \n"
"    padding-top:3px;\n"
"}")
        self.UserLoginPBt.setText('登陆')
        self.UserLoginPBt.setObjectName("UserLoginPBt")
        self.horizontalLayout_login.addWidget(self.UserLoginPBt)
        return None
~~~

### 控件部分

我们知道每一个button对应一个槽函数，如何将其链接，接下来会介绍：

~~~python
self.Clicked_Moved_Action_MainWindow()
~~~

~~~python
    def Clicked_Moved_Action_MainWindow(self):
        #On the Tab
        #self.UserLoginPBt.clicked.connect(self.LoginWindow)
        self.SettingPBt.clicked.connect(self.SettingPBt_ClickEvent)#设置按键以及它的槽函数
        self.GrowthRateCurvePBt.clicked.connect(self.growthrate)#查看对虾生长曲线,并且将其放在label当中
        self.StartIdentifyPBt.clicked.connect(self.IdentifyCameraEmbeddedWindow)
        self.SensorPBt.clicked.connect(self.ClickEvent_SensorPBt)#开启传感器
        #setting.ui
        #关于设置中的操作，应当先进性选择摄像头是否使用海康，还有检测什么东西，而后进行预览操作，选择路径，也可以直接复制粘贴输入路径，最后点击左侧检查设置是否成功
        self.ui_Setting.CameraSettingLinkPBt.clicked.connect(self.CameraSettingLinkPBt_ClickEvent)#摄像头设置，会判断是否有test，并弹窗
        self.ui_Setting.SensorSettingLinkPBt.clicked.connect(self.SensorSettingLinkPBt_ClickEvent)#传感器设置，会判断是否有参数，并弹窗
        self.ui_Setting.ImageStorageLinkPBt.clicked.connect(self.ImageStorageLinkPBt_ClickEvent)#图片存储路径，会打开指定文件夹，如果路径错误会报错，没有错会自动打开路径
        self.ui_Setting.VideoStorageLinkPBt.clicked.connect(self.VideoStorageLinkPBt_ClickEvent)#视频存储路径，会打开指定文件夹，如果路径错误会报错，没有错误打开路径
        self.ui_Setting.ModelLinkPBt.clicked.connect(self.ModelLinkPBt_ClickEvent)#模型存储路径
        self.ui_Setting.ModelMapFileLinkPBt.clicked.connect(self.ModelMapFileLinkPBt_ClickEvent)#模型映射文件存储路径
        self.ui_Setting.IdentifiedFileLinkPBt.clicked.connect(self.IdentifiedFileLinkPBt_ClickEvent)#识别内容路经检测，如果正确会弹出来
        self.ui_Setting.ResultFileLinkPBt.clicked.connect(self.ResultFileLinkPBt_ClickEvent)#识别结果储存路径检测，如果正确会弹出来
        
        self.ui_Setting.HIKCameraSettingPBt.clicked.connect(self.HIKInfoSet)#就是预览，弹出海康摄像头面板
        self.ui_Setting.SensorSettingPBt.clicked.connect(self.SensorInfoSet)#预览按钮，弹出传感器设置
        self.ui_Setting.ImageStoragePathChangePBt.clicked.connect(self.SetImageStoragePath)#可以进行图片保存路径选择（只能是文件夹）
        self.ui_Setting.VideoStoragePathChangePBt.clicked.connect(self.SetVideoStoragePath)#可以进行视频保存路径的选择（只能是文件夹）
        self.ui_Setting.ModelPathChangeBt.clicked.connect(self.SetModelPath)#选取模型位置
        self.ui_Setting.ModelMapFileChangePBt.clicked.connect(self.SetModelMapFilePath)#选取模型映射文件位置
        self.ui_Setting.IdentifiedFileChangePBt.clicked.connect(self.SetIdentifiedFilePath)#设置检测文件信息
        self.ui_Setting.ResultFileChangePBt.clicked.connect(self.SetResultFilePath)#结果保存的文件夹信息
        
        self.ui_Setting.PatternCBox.currentIndexChanged.connect(self.PatternCBox_SignalChangeEvent)#信号重载问题，当选择切换Patterncbox时，相应的检测文件目录也会清除
        
        self.ui_Setting.PhotoAutoRecognitionRBt.clicked.connect(self.PhotoAutoRecognitionRBt_ClickEvent)##连续自动拍摄
        
        self.ui_Setting.ResetSettingPBt.clicked.connect(self.ResetConfig)#重置设置
        self.ui_Setting.SaveSettingPBt.clicked.connect(self.SaveConfig)#保存设置
        
        #HIK.ui
        self.ui_HIKInfoWgt.HIKResetConfigPBt.clicked.connect(self.HIKResetConfig)#海康摄像头设置重置
        self.ui_HIKInfoWgt.HIKSaveConfigPBt.clicked.connect(self.HIKSaveConfig) #海康摄像头设置保存
        
        #Sensor.ui
        self.ui_SensorInfoWgt.SensorResetSettingPBt.clicked.connect(self.SensorResetConfig)#传感器设置重置
        self.ui_SensorInfoWgt.SensorSaveSettingPBt.clicked.connect(self.SensorSaveConfig)#传感器设置保存
        
        #All Tabs
        #Tab 1
        self.GrayIMGCkB.stateChanged.connect(self.SetGray)#状态改变信号，只要改变就会发出信号
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)#将滑动按钮与数值按钮放在一起
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)#将滑动按钮与数值按钮放在一起
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        #self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        #self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        #self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        #self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        #self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        #self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        #self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        #self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)

        self.CamaraPBt.clicked.connect(self.CameraDisplayEmbeddedWindow)#开启摄像头
        #self.CloseVideoPBt.clicked.connect(self.StopCamera)
        self.CameraPopupWindowPBt.clicked.connect(self.CameraDisplayPopupWindow)#相机弹窗的有关操作
        self.TakePhotoPBt.clicked.connect(self.Photo)#拍照相关操作
        self.VideoPBt.clicked.connect(self.Video)#录像有关操作
        #self.AutoDelayPhotographyPBt.clicked.connect(self.AutoDelayPhoto)
        #用户登录：
        self.UserLoginPBt.clicked.connect(self.ClickEvent_UserLoginWindow)
        #Tab2
        #self.StartIdentifyPBt.clicked.connect(self.IdentifyCameraEmbeddedWindow)
        
        #Tab3
        self.CloudServerPBt.clicked.connect(self.ClickEvent_CloudServerPBt)
        return None
    
~~~

![image-20220724211203436](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724211203436.png)

进行简单介绍，点击左侧的设置是对于文本框中的内容进行判断是否正确，比如第一个：

~~~python
    def CameraSettingLinkPBt_ClickEvent(self):
        if self.ui_Setting.HIKCameraCheckBox.isChecked():#如果打勾
            self.url="rtsp://{}:{}@{}/Streaming/Channels/1".format(self.HIKUsername,self.HIKPasswd,self.HIKCameraIP)#访问海康摄像头网址
            print(self.url)
            self.HIKCameraThread=HIKCameraThread(self.url)#传入参数进行cv2.调用摄像头
            self.CountdownThread=CountdownThread(5)#CountdownThread子线程
            self.HIKCameraThread.start()#打开海康摄像头
            self.CountdownThread.start()
            while (self.CountdownThread.TimeoutFlag==False):
                QtWidgets.QApplication.processEvents()#处理密集型耗时的事情
            self.CountdownThread.terminate()#terminate() 函数 是用来杀死子进程的
            if self.HIKCameraThread.HIKCameraRunFlag==True:#HIKCameraThread子线程，表示摄像头正常开启
                if self.HIKCameraThread.HIKCamera.read()[0]:#cv2.VideoCapture.read函数返回两个值，一个是索引，一个是图像，索引为布尔值，但是摄像头现在没有show
                    self.QtMessageBox('Information','摄像机初始化成功！','摄像机IP地址：{}'.format(self.ui_Setting.CameraSettingLE.text()))
                    self.HIKCameraThread.terminate()#terminate() 函数 是用来杀死子进程的，terminate（）只对run（）执行的代码能够暴力结束
                    self.CameraSetting=self.url
                else:
                    self.QtMessageBox('Critical','摄像机初始化失败！','请检查海康摄像机网站信息是否设置正确！')
            else:
                self.QtMessageBox('Critical','摄像机初始化失败！','请检查海康摄像机网站信息是否设置正确！')
                self.HIKCameraThread.terminate()
        else:#如果没有使用海康传摄像头
            try:
                self.CameraSetting = self.ui_Setting.CameraSettingLE.text()#默认是0
                #print(self.CameraSetting)
                if self.CameraSetting == '':
                    self.QtMessageBox('Critical','摄像机初始化失败！','请填写摄像机接口地址')
                    return None
                else:
                    try:
                        self.CameraSetting = eval(self.CameraSetting)
                    except:
                        pass
                    finally:
                        self.camera = cv.VideoCapture(self.CameraSetting)
                        if self.camera.read()[0]:
                            self.QtMessageBox('Information','摄像机初始化成功！','摄像机接口：{}'.format(self.CameraSetting))
                            self.camera.release()
                        else:
                            self.QtMessageBox('Critical','摄像机初始化失败！','请更换摄像机接口')
            except:
                self.QtMessageBox('Critical','摄像机初始化失败！','请更换摄像机接口')
            return None
    
~~~

- 有些比如说图片存储路径，视频存储路径以及模型文件等等是对于文件判断而后进行弹窗

~~~python
    def ImageStorageLinkPBt_ClickEvent(self):
        try:
            Path = self.ui_Setting.ImageStoragePathLE.text()
            if len(Path.split('/'))== 1:
                pass
            else:
                Path_list = Path.split('/')
                Path = ''
                for i in range(len(Path_list)):
                    Path += Path_list[i]+ '\\'
                Path = Path[0:-1]
            os.startfile(r'{}'.format(Path))
        except:
            self.QtMessageBox('Critical','请更换路径','系统找不到指定的路径{}！'.format(self.ui_Setting.ImageStoragePathLE.text()))
        return None
~~~

而点击右侧的预览则是对摄像头进行选择以及文件进行具体选择：

![image-20220724211731023](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724211731023.png)

![image-20220724211741111](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724211741111.png)

~~~python
 def HIKInfoSet(self):
        if self.HIKInfoWgt.isVisible():#一般是false
            return self.HIKInfoWgt.close() and self.HIKInfoWgt.show()
        else:
            return self.HIKInfoWgt.show()
~~~

~~~python
 def SensorInfoSet(self):
        if self.SensorInfoWgt.isVisible():
            return self.SensorInfoWgt.close() and self.SensorInfoWgt.show()
        else:
            return self.SensorInfoWgt.show()
~~~

文件类型则是对于文件的保存地址进行搜索

~~~python
    def SetImageStoragePath(self):
        self.TimerShift(stop=True)
        self.ImageStoragePath_name = QtWidgets.QFileDialog.getExistingDirectory(self, "选择图片存放路径", '.')#获取一个文件夹路径
        if self.ImageStoragePath_name:
            self.ui_Setting.ImageStoragePathLE.setText(self.ImageStoragePath_name)
        self.SettingPBt_ClickEvent()
        self.TimerShift(stop=False)
        return None
    
    def SetVideoStoragePath(self):
        self.TimerShift(stop=True)
        self.VideoStoragePath_name = QtWidgets.QFileDialog.getExistingDirectory(self, "选择视频存放路径", '.')
        if self.VideoStoragePath_name:
            self.ui_Setting.VideoStoragePathLE.setText(self.VideoStoragePath_name)
        self.SettingPBt_ClickEvent()
        self.TimerShift(stop=False)
        return None
~~~

- 接下来是重置和保存

~~~python
    def ResetConfig(self):
        try:
            #win32api.SetFileAttributes('UserConfig.conf', win32con.FILE_ATTRIBUTE_NORMAL)
            os.remove('UserConfig.conf')#删除指定路径下的文件
        except:
            pass
        cfg.creat_config()
        self.config = cfg.config_ini()
        self.ReadConfig()#而后在创建
        self.QtMessageBox('information','重置设置','已更新保存路径！')
        return None
    
    def SaveConfig(self):
        try:
            self.IMGStoragePath = r'{}'.format(self.ui_Setting.ImageStoragePathLE.text())
            self.VideoStoragePath = r'{}'.format(self.ui_Setting.VideoStoragePathLE.text())
            self.ModelPath = r'{}'.format(self.ui_Setting.ModelPathLE.text())
            self.ModelMapFilePath = r'{}'.format(self.ui_Setting.ModelMapFilePathLE.text())
            self.ResultFilePath = r'{}'.format(self.ui_Setting.ResultFilePathLE.text())
            self.CloudServiceURL = r'{}'.format(self.ui_Setting.CloudServiceURLLE.text())
             if self.ui_Setting.HIKCameraCheckBox.isChecked()==True:
                self.HIKCameraCheckBoxFlag = 1
            elif self.ui_Setting.HIKCameraCheckBox.isChecked()==False:
                self.HIKCameraCheckBoxFlag = 0
            if self.ui_Setting.PatternCBox.currentText() == '检测图像':
                self.IdentifiedFilePath = r'{}'.format('0'+'###'+ self.ui_Setting.IdentifiedFilePathLE.text())
            elif self.ui_Setting.PatternCBox.currentText() == '检测文件夹内的图像':
                self.IdentifiedFilePath = r'{}'.format('1'+'###'+ self.ui_Setting.IdentifiedFilePathLE.text())
            elif self.ui_Setting.PatternCBox.currentText() == '检测视频':
                self.IdentifiedFilePath = r'{}'.format('2'+'###'+ self.ui_Setting.IdentifiedFilePathLE.text())            
        except:
            pass
        finally:
            new_config = {'CameraSetting':self.ui_Setting.CameraSettingLE.text(),'ImagePath':self.IMGStoragePath,'VideoPath':self.VideoStoragePath,'ModelPath':self.ModelPath,'ModelMapFilePath':self.ModelMapFilePath,'IdentifiedFilePath':self.IdentifiedFilePath,'ResultFilePath':self.ResultFilePath,'HIKCameraCheckBoxFlag':self.HIKCameraCheckBoxFlag,'CloudServiceURL':self.CloudServiceURL}
            cfg.modify_config(new_config)#修改配置
            self.QtMessageBox('information','保存设置','已更新保存路径!')
        return None
~~~

接下来是一些滑动按钮与调整按钮的连接

~~~python
self.GrayIMGCkB.stateChanged.connect(self.SetGray)#状态改变信号，只要改变就会发出信号
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)#将滑动按钮与数值按钮放在一起
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)#将滑动按钮与数值按钮放在一起
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
~~~

### 开启摄像头与弹窗和拍照等：

![image-20220724212352293](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220724212352293.png)

~~~python
self.CamaraPBt.clicked.connect(self.CameraDisplayEmbeddedWindow)#开启摄像头
        #self.CloseVideoPBt.clicked.connect(self.StopCamera)
        self.CameraPopupWindowPBt.clicked.connect(self.CameraDisplayPopupWindow)#相机弹窗的有关操作
        self.TakePhotoPBt.clicked.connect(self.Photo)#拍照相关操作
        self.VideoPBt.clicked.connect(self.Video)#录像有关操作
~~~

~~~python
    def CameraDisplayEmbeddedWindow(self):
        if self.ui_Setting.HIKCameraCheckBox.isChecked() and self.HIKCameraInitFlag==False:
            self.HIKCameraThread=HIKCameraThread(self.url)
            self.CountdownThread=CountdownThread(5)#楞五秒开启
            self.HIKCameraThread.start()
            self.CountdownThread.start()
            while (self.CountdownThread.TimeoutFlag==False):
                QtWidgets.QApplication.processEvents()#处理复杂进程，如果五秒之后仍然没有开启就说明太慢了，进程被堵住了。
            self.CountdownThread.terminate()
            if self.HIKCameraThread.HIKCameraRunFlag==True:
                if self.HIKCameraThread.HIKCamera.read()[0]:
                    self.HIKCameraInitFlag=True
                    self.HIKCameraThread.terminate()#来杀死子进程的
                    if self.CamaraPBt.text()=='开启摄像头':
                        self.CamaraPBt.setText('关闭摄像头')
                        self.EmbeddedCameraFlag = True
                        if self.CameraPopupWindowPBt.text()== '打开相机弹窗':
                            self.StartCamera()
                        elif self.CamaraPBt.text()=='关闭摄像头':
                            self.CameraPBt.setText('开启摄像头')
                            self.EmbeddedCameraFlag = False
                            self.EmbeddedPictureLb.setPixmap(QtGui.QPixmap(""))#QtGui.QPixmap加载图片并显示，setPixmap是将图片嵌入到label(GrowthRateCurveLb)当中.这里是关闭所以什么都没显示
                            self.EmbeddedPictureLb.show()
                            if self.CameraDisplayWindowOpening == False and self.EmbeddedCameraFlag == False:
                                self.StopCamera()
            else:
                self.QtMessageBox('Critical','摄像机初始化失败！','请检查海康摄像机网站信息是否设置正确！')
                self.HIKCameraThread.terminate()
        else:
            if self.CamaraPBt.text()=='开启摄像头':
                self.CamaraPBt.setText('关闭摄像头')
                self.EmbeddedCameraFlag = True
                if self.CameraPopupWindowPBt.text()== '打开相机弹窗':
                    self.StartCamera()
            elif self.CamaraPBt.text()=='关闭摄像头':
                self.CamaraPBt.setText('开启摄像头')
                self.EmbeddedCameraFlag = False
                self.EmbeddedPictureLb.setPixmap(QtGui.QPixmap(""))
                self.EmbeddedPictureLb.show()
                if self.CameraDisplayWindowOpening == False and self.EmbeddedCameraFlag == False:
                    self.StopCamera()
        return None
~~~

其中：

~~~python
  self.HIKCameraThread=HIKCameraThread(self.url)
            self.CountdownThread=CountdownThread(5)#楞五秒开启
            self.HIKCameraThread.start()
            self.CountdownThread.start()
~~~

~~~python
class HIKCameraThread(QtCore.QThread):
    def __init__(self,url):
        super().__init__()
        self.url=url
        self.HIKCameraRunFlag = False
        
    def run(self):
        self.HIKCamera = cv.VideoCapture(self.url)#读取海康摄像头信息，但还不能够显示
        self.HIKCameraRunFlag = True
~~~

这里是多线程问题，在传参的时候init中必执行，而run需要用对应的子线程用start才能够执行。

接下来是相机弹窗：

~~~python
 def CameraDisplayPopupWindow(self):
        if self.CameraPopupWindowPBt.text()=='打开相机弹窗':
            self.CameraDisplayWindowOpening = True
            if self.CamaraPBt.text()=='打开摄像头':
                self.StartCamera()
            self.CameraPopupWindowMW.show()
        elif self.CameraPopupWindowPBt.text()=='关闭相机弹窗':
            self.CameraPopupWindowPBt.setText('打开相机弹窗')
            self.CameraDisplayWindowOpening = False
            self.CameraPopupWindowMW.close()
            if self.CameraDisplayWindowOpening == False and self.EmbeddedCameraFlag == False:
                self.StopCamera()
~~~

startcamera是对于当点击开启摄像头时，一些相应的按钮会变得不饿能够使用：

~~~python
    def StartCamera(self):
        if self.Ini == True:
            try:
                self.PrepCamera(self.ScreenWidth,self.ScreenHeight)#1920*1080
                self.Ini = False
            except:
                self.QtMessageBox('Critical','错误！','摄像头接口:{}初始化失败！\n请检查或更换接口'.format(self.CameraSetting))
                return None
        else:
            self.PrepCamera(self.ScreenWidth,self.ScreenHeight)#1920*1080

        self.VideoPBt.setEnabled(True)
        self.TakePhotoPBt.setEnabled(True)
        #self.DegreePBt.setEnabled(True)
        self.GrayIMGCkB.setEnabled(True)
        #self.AutoDelayPhotographyPBt.setEnabled(True)
        
        if self.GrayIMGCkB.isChecked() is False:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
            self.RGBjudgeIMGLb.setStyleSheet("QLabel{\n"
"    border-image: url(Mainwindow_Images/RGB.png);\n"
"}")
        #self.ExpTimeSld.setEnabled(True)
        #self.ExpTimeSpB.setEnabled(True)
        #self.GainSld.setEnabled(True)
        #self.GainSpB.setEnabled(True)
        #self.BrightSld.setEnabled(True)
        #self.BrightSpB.setEnabled(True)
        #self.ContrastSld.setEnabled(True)
        #self.ContrastSpB.setEnabled(True)

        self.CameraTimer.start(1)#启动或重新启动定时器，时间间隔单位为毫秒，这个是显示图片的关键
        self.timelb = time.perf_counter()#返回性能计数器的值（以分秒为单位），一般用于计算程序运行时间
~~~

stopcamera顾名思义：

~~~python
 def StopCamera(self):
        #self.EmbeddedCameraFlag = False
       
        #self.AutoDelayPhotographyPBt.setEnabled(False)
        self.GrayIMGCkB.setEnabled(False)
        self.VideoPBt.setEnabled(False)
        self.TakePhotoPBt.setEnabled(False)
        #self.DegreePBt.setEnabled(False)
        
        
        self.FmRateLCD.display(0)#display设置值
        self.IMGWidthLCD.display(0)
        self.IMGHeightLCD.display(0)
        #cv.destroyAllWindows()
        self.CameraTimer.stop()#关闭图片展示的关键，同时也是计时器
        self.camera.release()
        #self.EmbeddedCameraTimer = QtCore.QTimer()
        #self.EmbeddedCameraTimer.timeout.connect(self.EmbeddedCameraAdjustProcess)
        
        #self.AutoPhotoSettingMW.close()
        #self.AutoPhotoMW.close()
        return None
~~~

