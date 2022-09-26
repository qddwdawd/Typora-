

# Opencv计算机视觉实战

- 图像基本操作

图像是由一个一个点组成的，而每一个点我们叫做像素点，它的值为（0-255），这表示的是亮度，0最低为灰的，255最亮接近白色。

每一个像素点分别对应R中一个值，G中一个值，B中一个值。RGB叫做图像的颜色通道。一般的彩色图像都是RGB三颜色通道的。

![RBG](C:\Users\DELL\Desktop\work\picture\RBG.png)

201对应红色通道R中一个值，155对应绿色通道G中一个值，165对应蓝色通道B中一个值。

- 黑白图没有颜色通道
- 灰度图只有一个通道表示灰度图

***

~~~python
import cv2 #opencv读取的格式是BGR
import matpoltlib.pyplot as plt
import numpy as np
%matpoltlib inline
img=cv2.imread('')
~~~

***

~~~python
#用matplotlib读取csv类型数据并转化成图像
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
csv=pd.read_csv(r"C:\Users\zhang\Documents\Tencent Files\3281939931\FileRecv\data.csv",header=None)#header=None，添加行列索引，如果没有则默认为从文本开始。此时读取的数据含有索引列表为DataFrame的形式。
img=np.asarray(csv,dtype=np.float32)#将DataFrame的形式转化为数组的形式
plt.imshow(img,"gray")  #负责对图像进行处理，并显示其格式
plt.show()#plt.show()则是将plt.imshow()处理后的函数显示出来
~~~

- 以下是读取的csv数据：
- ![CSV数据](C:\Users\DELL\Desktop\work\picture\读取的csv为DataFrame数据有索引列.png)
- 以下为转化为二维数组的img数据：
- ![img](C:\Users\DELL\Desktop\work\picture\\img的array数据.png)
- 其中DataFrame的一行数转化为的是二维数组的一组。
- 先通过pd. read_csv进行读取csv数据，而后进行asarray进行变化为数组的形式,调整daype为np.float，一是调整精度，显示更为清晰，二是为了保证像素点在0-255范围之内。plt.imshow()中的数据类型可以为ndarray，及数组类型"gray"表示为灰度图像，负责对图像进行处理，并显示其格式，plt.show是将处理后的图像展示出来。

***

### numpy中矩阵的反转

- numpy.flip(m,axis = None)

把m在axis维度进行切片，并把这个维度的index进行颠倒。

源代码：

~~~python
#用matplotlib读取csv类型数据并转化成图像
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
csv=pd.read_csv(r"C:\Users\zhang\Documents\Tencent Files\3281939931\FileRecv\data.csv",header=None)#header=None，添加行列索引，如果没有则默认为从文本开始。
img=np.asarray(csv,dtype=np.float32)
plt.imshow(img,"gray")  #负责对图像进行处理，并显示其格式
plt.show()#plt.show()则是将plt.imshow()处理后的函数显示出来
~~~

![原图像](C:\Users\DELL\Desktop\work\picture\\原图像.png)

axis=0：上下翻转，意味着把行看成整体，行的顺序发生颠倒，每一行的元素不发生改变。(及镜像转换)(numpy 矩阵左右翻转/上下翻转)

~~~python
img1 = np.flip(img,axis=0) #将数组进行行的颠倒
plt.imshow(img1)
plt.show()
~~~

axis=1：左右翻转，意味着把列看成整体，列的顺序发生颠倒，每一列的元素不发生改变。

~~~python
img1 = np.flip(img,axis=0) #将数组进行列的颠倒
plt.imshow(img1)
plt.show()
~~~

***

#### 旋转180°：

~~~python
new_img = img.reshape(img.size) #将二维数组先转化为一维数组
new_img = new_img[::-1]#取一维数组的倒叙
new_img= new_img.reshape(img.shape)#将一维数组返回为原来的形状(二维数组
plt.imshow(new_img)
plt.show()
~~~

![旋转180°](C:\Users\DELL\Desktop\work\picture\\旋转180°.png)

#### 向右旋转90°：

~~~python
new_img = img.reshape(img.size)
new_img = new_img[::-1]
new_img = new_img.reshape(img.shape)
new_img = np.transpose(new_img)[::-1]
plt.imshow(new_img)
plt.show()
~~~

- 补充：二维数组的转置：
- ![二维数组转置](C:\Users\DELL\Desktop\work\picture\二维数组转置.png)

![向右偏转90°]()![向右偏转90°](C:\Users\DELL\Desktop\work\picture\向右偏转90°.png)

#### 向左偏转90°

~~~python
new_img = np.transpose(img)
new_img= new_img[::-1]
plt.imshow()
plt.show()
~~~

![向左偏转90°](C:\Users\DELL\Desktop\work\picture\向左偏转90°.png)

***

### 补充：关于fig. add_subplot的作用

~~~python
#代码如下：
new_img = img.reshape(img.size)
new_img = new_img[::-1]
new_img = new_img.reshape(img.shape)
new_img = np.transpose(new_img)[::-1]
fig = plt.figure(figsize=(20,20),dpi=150)
fig.add_subplot(121)
plt.imshow(new_img)
new_img = np.transpose(img)
new_img= new_img[::-1]
fig.add_subplot(122)
plt.imshow(new_img)
plt.show()
~~~

- 上述代码中的fig = plt.figure(figsize=(20,20), dpi =150)的意思创建一个画布，而后用fig. add_subplot(121)，(在画布上所以用你创建的fig. )分成1行2列的格局，后边那个参数1代表第几个。当写第几个图像时，就在图像的上边加上fig. add_subplot(121)，意思是你在写这个，不用加plt.show（），这个只需要在最后一行加就行。

***

- 插值：

~~~python
import cv2
csv=pd.read_csv(r"E:\Jupyter与pycharm\pythonProject1\jupyter数据\data.csv",header=None)
img=np.array(csv,dtype=np.float32)
fig = plt.figure(figsize = (20,8),dpi = 150)
fig.add_subplot(121)
plt.imshow(img)

fig.add_subplot(122)
img1=cv2.resize(img,(500,500))
plt.imshow(img1)
plt.show()
~~~

- ![cv2.resize的各种插值方法](C:\Users\DELL\Desktop\work\picture\cv2,resize的各种插值方法.png)
- 就直接用cv2进行图像插值了。

***

### 如何显示和保存

- 先用plt.imshow()括号中填入图片信息，可以是数组，也可以是图片，进行处理，而后用plt.show()才能显示图片。
- 保存图像用plt.savefig()。括号中为保存的地址如：（r"E:\Jupyter与pycharm\pythonProject1\jupyter数据\size.png“），而调用此函数之前必须要有图片，否则将只会保存一个画布，而没有任何东西，任何数据都不行，只能由plt.show()处理过数据之后才行，如在matplotlib中进行plt.plot(x,y)之后在画布上有东西之后才能保存。

***

### 数据读取——图像

~~~python
import cv2 #opencv读取的格式是BGR
import matlpotlib.pyplot as plt
import numpy as np
%matpoltlib inline


img=cv2.imread('')
#现在img为ndarray的形式。
~~~

***

opencv默认读取格式为BGR格式，与matplotlib的展示有些冲突。

~~~python
#图像的显示，也可以创建多个窗口
cv2.imshow('image',img)
#等待时间，毫秒级，0表示任意键终止,填写其他的值表示停留那个值的时间。
cv2.waitKey(0)
cv2.dasrtoaryAllWindows()
~~~

- 当然也可以用

- ~~~python
  import skimage.io as io
  ~~~

- 这个是按照RBG进行的读取数据。

所有的图像无论用cv2.imread，还是io.imread,都是传入图像之后都对将图像转化为三维数组的形式，及RGB，而这两个一个为BGR，一个为RGB，而matplotlib中的plot为RBG显示系统，所以：

cv2.imshow()用专有的来显示：

~~~python
cv2.imshow('name',img)
cv2.waitKey(0)
cv2.dasrtoaryAllWindows()
~~~

而io.imread则可以用plt.imshow(img),plt.show()来展示。

#### 想要灰度图像

~~~python
picture = cv2.imread(r"C:\Users\DELL\Desktop\work\picture\test.jpg")
gray = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
cv2.imshow('ersion',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

![image-20220823190750296](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220823190750296.png)

或者直接：

~~~python
picture = cv2.imread(r"C:\Users\DELL\Desktop\work\picture\test.jpg",cv2.COLOR_BGR2GRAY)
cv2.imshow('ersion',picture)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~



***

## 插值：

~~~python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
def Interpolation_Bilinear(filepath, desHeight, desWidth):
    # 双线性插值法
    img = Image.open(filepath)  # 读取图片
    img = np.array(img, np.uint8)  # 转化为numpy数组
    desImageNumpy = np.zeros(img.shape, np.uint8)  # 生成一个大小相同的全0的numpy数组
    height, width, mode = img.shape[0], img.shape[1], img.shape[2]  # 高、宽、channel数

    # 找出目标位置在源图中的位置
    scale_x = float(width)/desWidth  # x轴缩放比例
    scale_y = float(height)/desHeight  # y轴缩放比例
    des_image = np.zeros((desHeight, desWidth, mode), np.uint8)
    for n in range(mode):
        for des_y in range(desHeight):
            for des_x in range(desWidth):
                # 确定四个近邻点坐标
                src_x = (des_x + 0.5) * scale_x - 0.5  #
                src_y = (des_y + 0.5) * scale_y - 0.5

                src_x_1 = int(np.floor(src_x))  #
                src_y_1 = int(np.floor(src_y))
                src_x_2 = min(src_x_1+1, width-1)  # 防止坐标点寻找溢出
                src_y_2 = min(src_y_1+1, height-1)
                # 两次x轴线性插值
                value_1 = (src_x_2 - src_x)*img[src_y_1, src_x_1, n]+(src_x - src_x_1)*img[src_y_1, src_x_2, n]
                value_2 = (src_x_2 - src_x)*img[src_y_2, src_x_1, n]+(src_x - src_x_1)*img[src_y_2, src_x_2, n]
                # y轴线性插值
                des_image[des_y, des_x, n] = (src_y_2 - src_y)*value_1 + (src_y - src_y_1)*value_2
    print(des_image.shape)
    des_img = Image.fromarray(des_image)
    plt.imshow(des_img)
    plt.show()

if __name__ == '__main__':
    file_path = r"E:\Jupyter and pycharm\pythonProject1\jupyter2\img2.jpg"
    Interpolation_Bilinear(file_path, int(183*2), int(275*2))  # 双线性插值法
~~~

***

![cv2.imread](C:\Users\DELL\Desktop\work\picture\cv2.imread.png)

~~~python
#为了简便
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroaryAllWindows()
~~~

### 几个常见的方法：

- img.shape() 往往为三维数组，第三个参数为3表示为三层为彩色图，在Opencv中表示为BGR。

- 读取灰度图：

~~~python
img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)#第二参数表示的是读取的数据类型。
~~~

![灰度图](C:\Users\DELL\Desktop\work\picture\灰度图.png)

- 上图为灰度图，可以看到只有两个参数，最后一个参数为0表示只有一个颜色通道。
- 但是一般都是先转化为灰度图之后进行操作才可以。

***

## 数据读取——视频

- cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备。
- 如果是视频文件，直接制定好路径即可。

~~~python
vc = cv2.VideoCapture('text.mp4')#指定视频路径
#检查打开是否正确
if vc.isOpened():
    open,farme = vc.read( )#open会返回一个值，如果设一个循环，vc.read()指的是先读取第一帧，而后读取第二帧，而后继续。这个表达式会返回两个值，第一个为bool类型的值，如果打开open就会返回Ture否则就会返回False，第二个参数为这一帧的图像，且返回到False中。
else:
    open = False
~~~

~~~python
while open:
    ret,frame = vc.read() #此表达式中ret和open是一个参数，都是返回Ture或者False参数。
    if frame is None: #如果捕获到了帧数但帧数图像为空。
        break
    if ret ==True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        if cv2.waitKey(100) &0xFF ==27:
            break
vc.release()
cv2.destroyAllWindows()
~~~

- break在此程序中只能跳出 if ret==True这个循环，大循环while还是有的，而下一次由于frame还没有读取帧数为空，所以会执行第一个break。

***

##### ROI：截取部分图像数据

~~~python
img = cv2.imread("")
cat = img[0:50,0:200]
cv2.imshow('cat',cat)
cv2.waitKey(0)
cv2.destroaryAllWindows()
~~~

##### 颜色通道提取

~~~python
b,g,r = cv2.split(img)#split将图片进行通道提取。
~~~

~~~python
img = cv2.merge(b,g,r)
#将三通道结合
~~~

~~~python
#只保留R通道(其他两通道相似)：
cur_img=img.copy()
cur_img[:,:,0]=0
cur_img[:,:,1]=0
cv2.imshow('cat',cur_img)
cv2.waitKey(0)
cv2.destroaryAllWindows()
~~~

### 边界填充

~~~python
top_size,bottom_size,left_size,right_size=(50,50,50,50)#设置边界填充的距离。
replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BPRDER_REPLICATE)
~~~

![opencv边界填充](C:\Users\DELL\Desktop\work\picture\opencv边界填充.png)

![opencv图像填充](C:\Users\DELL\Desktop\work\picture\opencv图像填充.png)

## 数值计算

~~~python
img_cat= cv2.imread('')
img_cat2 = img_cat+10
(img_cat2+img_cat)#每一个像素点的值会分别相加，但是如果超过255，结果会取余数即%。
~~~

而cv2.add()函数：

~~~python
cv2.add(img_cat,img_cat2)#如果相加之后大于255，将会取最大值255
~~~

### 图像放缩，改值

~~~python
img_cat= cv2.resize(img_cat,(500,414))#改变大小，输出的shape时(414，500，3)。
~~~

~~~python
res = cv2.resize(img,(0,0),fx=4,fy=4)#实现图像放缩,想放大就输入大值，缩小输入小值。
~~~

### 图像融合

~~~python
res = cv2.addWeighted(ing_cat,0.4,img_dog,0.6,0)#名称表示是哪一项，小数表示图像权重，0表示提高亮度的程度。
~~~

### 图像阈值

![图像阈值处理](C:\Users\DELL\Desktop\work\picture\图像阈值处理.png)

![图像阈值处理](C:\Users\DELL\Desktop\work\picture\图像阈值处理2.png)

***

## 图像平滑

- 均值滤波
- 简单的一个平均卷积操作

~~~python
blur= cv2.blur(img,(3,3))

cv2.imshow('blur',blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 方框滤波
- 基本和均值一样，可以选择归一化 ,但容易越界。normalize控制是否除以9.

~~~python
box = cv2.boxFilter(img,-1,(3,3),normalize = True)

cv2.imshow('box',box)
cv2.waitKey(0)
cv2.destroaryAllWindows()
~~~

- 高斯滤波
- 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的，相当于自己做了一个权重矩阵。

~~~python
aussian = cv2.GaussianBlur(img,(5,5),1)

cv2.imshow('aussian',aussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 中值滤波
- 相当于用中值替代

~~~python
median = cv2.medianBlur(img,5)
cv2.imshow('median',median)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 展示所有的

~~~python
res = np.hstack(blur,aussian,median)
print(res)
cv2.imshow('median vs aveage',res)
cv2.watiKey(0)
cv2.destroyAllWindows()
~~~

### 形态学-腐蚀操作

~~~python
img = cv2.imread('dige.png')
kernel = np.ones((5,5),np.uint8) #卷积层
erosion =cv2.erode(img,kernel,iterations = 1)#iterations 腐蚀1次
cv2.imshow('ersion',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 腐蚀操作：首先取一个卷积核为（5，5），即单位操作空间。然后利用erode进行腐蚀操作，iterations为重复次数。
- 原理：在腐蚀操作中，对于一个像素为（5，5）的卷积核中，如果存在颜色不同的黑白点，即像素值为0和255共同存在，则这个卷积核将白色的进行腐蚀操作，即变成黑色。
- 结果:尖刺消失。

![image-20220823194425840](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220823194425840.png)

### 膨胀操作

~~~python
kernel = np.ones((5,5),np.uint8) #卷积层
erosion =cv2.dilate(erosion,kernel,iterations = 1)#iterations膨胀1次，erosion是腐蚀完之后的图像。
cv2.imshow('dilate',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

- 结果：线条变粗。

![image-20220823194341371](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220823194341371.png)

![image-20220823194623791](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220823194623791.png)

### 开运算与闭运算

~~~python
#开：先腐蚀，再膨胀
img = cv2.imread('')
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
cv2.imshow('opening',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

~~~python
#闭：先膨胀，再腐蚀
img = cv2.imread('')
kernel = np.ones((5,5),np.uint8)
closing= cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
cv2.imshow('closing',vlosing)
cv2.waitKey(0)
cv2.destroyAllWindows()
~~~

## 梯度运算

![image-20220824095308118](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824095308118.png)

![image-20220824095333769](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824095333769.png)



### 礼帽和黑帽

![image-20220824095653917](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824095653917.png)

- 礼帽结果：
- ![image-20220824095727149](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824095727149.png)
- 黑帽结果：
- ![image-20220824095834864](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824095834864.png)

## 图像梯度-Sobel算子

![image-20220824100159954](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824100159954.png)

![image-20220824100430566](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824100430566.png)

- Gx水平梯度，Gy数值梯度。来算取像素点之间的差异。

![image-20220824100731140](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824100731140.png)

~~~python
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize = 3)#img是传入的图片，cv2.cv_64F是一种更高级的操作，将其表现为负的结果。dx=1算的是水平的，dy=0表示不算竖直的梯度。
cv_show =(sobelx,'sobelx')
~~~

![image-20220824101143780](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101143780.png)



![image-20220824101333711](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101333711.png)

由于右边黑减白必是负值，opencv截断操作变为0，而后变为黑色。

![image-20220824101838092](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101838092.png)

- 进行一个绝对值转换，convertScaleAbs。

![image-20220824101654005](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101654005.png)

接下来算Y：

![image-20220824101845212](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101845212.png)

![image-20220824101829069](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101829069.png)

![image-20220824101915822](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824101915822.png)

- 0.5是权重，偏置项为一般为0。

![image-20220824102017507](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824102017507.png)

![image-20220824102249659](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824102249659.png)

![image-20220824102304218](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824102304218.png)

## 图像梯度-Scharr算子

![image-20220824103536274](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824103536274.png)

## 图像梯度-laplacian算子

![image-20220824103600131](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824103600131.png)



- 不同算子之间的比较：

![image-20220824104109603](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824104109603.png)

![image-20220824104117633](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824104117633.png)

# Canny边缘检测

![image-20220824104420139](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824104420139.png)

![image-20220824110115776](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110115776.png)

![image-20220824110151258](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110151258.png)

![image-20220824110230311](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110230311.png)

![image-20220824110259497](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110259497.png)

![image-20220824110726106](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110726106.png)



![image-20220824110815017](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824110815017.png)

![image-20220824111136810](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824111136810.png)

- (80,150)代表的是自己设定的梯度最大值最小值。

![image-20220824111353598](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824111353598.png)

![image-20220824111404477](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824111404477.png)

![image-20220824111419748](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824111419748.png)

