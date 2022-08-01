#  matplotlib

1. 能将数据进行可视化，更直观的呈现
2. 使数据更加客观，更具有说服力

matplotlib：仿照MATLAB实现，最流行的Python底层绘图库，主要做数据可视化图表。

aixs表示x或者y这种坐标轴

- 两种导包方式

~~~python
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
~~~

~~~python
from matplotlib import pyplot as plt #导入pyplot
x = range(2,26,2)
y = [15,13,14.5,17,20,25,26,26,27,22,18,15]
plt.plot(x,y)
plt.show()
~~~

### 设置图片大小

~~~python
import matplotlib.pypolt as plt
fig = plt.figure(figsize  = (20,8),dpi = 80)
#figure图形图标的意思，在这里指的就是我们画的图
#通过实例化一个figure并且传递参数，能够在后台自动使用该figure的实例
#在图像模糊的时候可以传入dpi参数，让图片更加清晰
x =range(2,26,2)
y = [15,13,14.5,17,20,25,26,26,27,22,18,15]
plt. plot(x,y)
plt.savefig('.t1\sig_size.png')  #在完成会图纸后保存
plt.show()
~~~

- 注意：plt.savefig的使用：
- 在文件夹中复制地址时，`文件夹中的地址是用 \ 来分隔不同文件夹的`，而`Python识别地址时只能识别用 / 分隔的地址`。在字符串前加上r或R，即：imread(r‘C:\Users\li735\PycharmProjects\untitled\abc.txt’) ，其中r或R在python中表示一个不转义的字符串。

### 绘制X Y轴

***

#### x轴的刻度改变

~~~python
plt.xticks(range(2,25)) #绘制x轴的刻度，利用range可迭代对象，从2到24
~~~

- 也可以自己传一个列表

~~~python
_xtick_labels = [i/2 for i in range(2,49)]
plt.xticks(_xticks_labels)
~~~

~~~
plt.xticks([i/2 for i in range(2,49)])
~~~

#### y轴的刻度改变

同x轴的刻度改变

~~~python
plt.yticks(range(min(y),max(y)+1))
~~~

***

- 练习

~~~python
from matplotlib import pyplot as plt
import random
plt.figure(figsize=(13,2),dpi = 90)
x = range(0,120)
y = [random.randint(20,35) for i in range(120)]
plt.xticks(range(0,120,2))
plt.yticks(range(0,120,2))
plt.plot(x,y)
plt.show()
~~~

![练习题](C:\Users\DELL\Desktop\work\picture\练习1.png)

- 补充：

**np.random.randint()**:函数的作用是，返回一个随机整型数，其范围为[low, high)。如果没有写参数high的值，则返回[0,low)的值。

***

### x轴刻度改变（进阶）

- plt.xticks够把x传进去时，能够绘制出相应所对应的数字，但只能是数字，如：plt.xticks(range(0,120,2))这种，但是我们想传入字符串，这就需要传入另一个参数。

- 想要显示字符串，需要在plt.xticks()中传入两个参数，第一个参数显示的是数字，第二个参数为字符串，这两个参数是一一对应的，也就是说它们的长度，范围必须对应相同，可以理解为_x为1时，即在x轴1的位置对应的是_xticks_lable的第一个字符串实例：

~~~python
from matplotlib import pyplot as plt
import random
plt.figure(figsize=(12,5),dpi = 100)
x = range(0,120)
y = [random.randint(20,35) for i in range(120)]
_x = list(x)[::10]
_xticks_lable = ["hellow{}".format(i) for i in _x]
plt.yticks(range(0,120,2))
plt.xticks(_x,_xticks_lable)
plt.plot(x,y)
plt.show()
~~~

- 改良版：

~~~python
from matplotlib import pyplot as plt
import random
plt.figure(figsize=(12,5),dpi = 100)
x = range(0,120)  #它不是列表的形式
y = [random.randint(20,35) for i in range(120)]
_x = list(x)
_xticks_lable = ["10点{}分".format(i) for i in range(60)]
_xticks_lable += ["11点{}分".format(i) for i in range(60)]
plt.yticks(range(0,120,2))
plt.xticks(_x[::3],_xticks_lable[::3],rotation = 270) #rotation逆时针旋转，45°的时候时斜着的
plt.plot(x,y)
plt.show()
~~~

![练习2](C:\Users\DELL\Desktop\work\picture\练习2.png)

### 设置中文字体

- 注意:在matplotlib中中文是不显示的，需要修改默认字体。

~~~python
#解决方法  为Windows和设置字体的方法
import matplotlib
matplblib.rc  #可以用ctrl+B查看源码
font = {'family' : 'MicroSoft YaHei',
              'weight' : 'bold',
              'size'   : 'larger'}
matplotilb.rc('font',**font)
~~~

或者：

~~~python
import matplotlib
matplblib.rc  #可以用ctrl+B查看源码
matplblib.rc("font",family = 'MicroSoft YaHei',weight="blod")
~~~

***

- 正确的方法：

~~~python
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
~~~

### 添加描述信息

~~~python
plt.xlabel间')
plt.ylabel温度 单位(℃)")
plt.title("10点到12点每分钟的气温变化")
~~~

##### 注意

1. plt.figure()是对图像框架进行设置
2. plt.x/ytick()是对x，y轴的刻度进行设置
3. plt.x/ytick()是对x，y轴添加描述信息（表头）
4. x,y是对所描绘的曲线进行的取值，且两值个数必须相同

案例：

~~~python

~~~

***

![案例1](C:\Users\DELL\Desktop\work\picture\案例1.png)

## 绘制网格

~~~python
plt.grid()
~~~

- 网格的x，y的个数与plt.x/yticks()设置的横纵坐标有关系

##### 设置网格透明度

~~~python
plt.grid(alpha = 0.4) #从0到1越来越透明
~~~

***

##### 绘制多条曲线

- 只需要在当前的基础之上，再次书写另一条直线的x，y别的不变，因为用同一个图像。如：

~~~python
x = range(11,31)
y = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
x_1= range(11,31)
y_1 =[1,2,3,4,3,4,4,4,4,3,2,5,6,7,6,5,4,3,4,5]
plt.plot(x,y)
plt.plot(x_1,y_1)
~~~

图像如下：

![两条线的图像](C:\Users\DELL\Desktop\work\picture\绘制两条折线.png)

#### 在图像上标注

~~~python
plt.plot(x,y_1, label ='自己')
plt.plot(x,y_2, label ='同桌')
#添加图例
plt.legend(loc = "upper left")
~~~

#### 自己设置线条颜色

 在plot中传输线条颜色

~~~python
plt.plot(x,y_1, label ='自己',c = 'orange') #c就是color的缩写
plt.plot(x,y_2, label ='同桌',color = 'c')#第二个c是蓝色
~~~

- 参考表

![参考表](C:\Users\DELL\Desktop\work\picture\绘制颜色matplotlib.png)

##### 自己设置线条类型

~~~python
plt.plot(x,y_1, label ='自己',c = 'orange',linestyle = ":") 
plt.plot(x,y_2, label ='同桌',color = 'c',linestyle = "-.")
~~~

##### 自己设置线条粗细

~~~python
plt.plot(x,y_1, label ='自己',c = 'orange',linewidth = 5) 
plt.plot(x,y_2, label ='同桌',color = 'c',linewidth = 6)
~~~

****

## 绘制散点图

- plt.scatter

~~~python
from matplotlib import pyplot as plt
import random
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
plt.figure(figsize=(12, 5), dpi=100)
y_3 = [11, 17, 12, 13, 14, 13, 15, 12, 4, 5, 6, 12, 23, 34, 12, 34, 54, 13, 13, 13]
y_10 = [4, 3, 3, 43, 23, 45, 43, 23, 44, 32, 13, 13, 14, 15, 23, 23, 23, 45, 56, 67]
x_3 = range(1, 21)
x_10 = range(51, 71)
_x = list(x_3) + list(x_10)
_xtick_labels = ["3月{}日".format(i) for i in x_3]
_xtick_labels += ["10月{}日".format(i-50) for i in x_10]
plt.xticks(_x[::2],_xtick_labels[::2],rotation = 90)
plt.yticks(list(range(1,70))[::8])
plt.scatter(x_3, y_3,label = '三月',c = "orange")
plt.scatter(x_10, y_10,label = '十月',c = "c")
plt.grid(alpha = 0.4)
plt.legend()
plt.xlabel = ('两个月')
plt.ylabel = ('温度')
plt.title = ('气温散点图')
plt.show()
~~~

![绘制散点图](C:\Users\DELL\Desktop\work\picture\绘制散点图.png)

fig = plt.figure(figsize(4,5))中的figsize是表示你画的图的大小，4表示宽度，5表示高度

fig ,ax = plt.subplots()

与fig = plt.figure()    ax = fig.add_subplot(1,1,1)

fig = plt.figure()可以调整每个子图的大小

ax = fig.add_subplot(1,1,1)表示的是有1行1列，其中1表示第一个图

***

#### 解方程实例

~~~python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
fig = plt.figure(figsize= (8,4),dpi = 150)
x = np.linspace(-4,4,1000)
y = 4*(x**3)+3*(x**2)+6*x+3
plt.xticks = (range(-4,4))
plt.xlabel('x自变量')
plt.ylabel('y因变量')
plt.title('求解方程')
plt.grid(alpha = 0.3)
plt.plot(x,y,color = 'c')
plt.show()
~~~

## 绘制条形图

~~~python
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
a = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','脯氨酸']
plt.figure(figsize = (10,8),dip = 80)
b = [56.01,54,33,16.49,45,3,56,23,45,65,1,13]
plt.yticks(range(0,70,10))
plt.xticks(range(len(a)),a,rotation = 90)
plt.xlabel('电影名称')
plt.ylabel('电影票房')
plt.bar(range(len(a)),b，width = 0.3)
plt.show()
~~~

- 注意，在xticks中有两个参数，且第一个参数必定是为数字。可以放数组，也可以放range（）产生的数组。且与后面的字符也好，数字也好相互对应。

## 画横着的条形图

~~~python
#绘制横着的条形图
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
a = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','脯氨酸']
plt.figure(figsize = (10,8),dpi = 80)
b = [56.01,54,33,16.49,45,3,56,23,45,65,1,13]
plt.xticks(range(0,70,10))
plt.yticks(range(len(a)),a)
plt.ylabel('电影名称')
plt.xlabel('电影票房')
plt.barh(range(len(a)),b，height = 0.3,color = 'orange')
plt.grid(alpha = 0.4)
plt.show()
~~~

- 注意：将x变化成y，而后将bar变成barh第三个参数为高度height

***

~~~python
from matplotlib import pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
a = ['酒精','苹果酸','灰','灰的碱性','镁','总酚']
b_14 = [12,34,43,23,34,21]
b_15 = [23,45,43,45,43,21]
b_16 = [45,67,54,43,12,24]

fig = plt.figure(figsize  = (20,15),dpi = 150)
x_14 = list(range(len(a)))
x_15 = [i+0.2 for i in x_14]
x_16 = [i+2*0.2 for i in x_14]
plt.xticks(range(len(a)),a)
plt.yticks(list(range(70))[::10])
plt.xlabel('电影名称')
plt.ylabel('票房总数')
plt.bar(x_14 ,b_14,width = 0.2,label= '9月14日')
plt.bar(x_15 ,b_15,width = 0.2,label = '9月15日')
plt.bar(x_16 ,b_16,width = 0.2,label = '9月16日')
plt.legend()
plt.grid(alpha = 0.4)
plt.show()
~~~

***

## 绘制直方图

没有经过统计的数据可以绘制直方图，hist方法会自动进行分组，可以用来处理庞大的数据。

~~~python
from matplotlib import pyplot as plt
a =[12,13,12,23,23,35,35,23,45,23,44,23,42,12,35,1,32,12,33,43,43,34,23,34,45,43,35,23,12,11,34,45,23,34,45,23,45,43,45,56,23,34,43,23,12,34,54,33,34,32,23,22,12,12,23,34,45,23,34,23]
plt.hist(a,20) #hist为直方图独有的方法，第一个参数是传入的数据，第二个参数是分为多少组，有多少组就有说少个竖条。
plt.show()
~~~

- 计算组数

~~~python
d = 5 #组距
num_bins = (max(a)-min(a)//d)
~~~

~~~python
from matplotlib import pyplot as plt
a =[12,13,12,23,23,35,35,23,45,23,44,23,42,12,35,1,32,12,33,43,43,34,23,34,45,43,35,23,12,11,34,45,23,34,45,23,45,43,45,56,23,34,43,23,12,34,54,33,34,32,23,22,12,12,23,34,45,23,34,23]
d = 5
num_bins = (max(a)-min(a)//d)
plt.hist(a,num_bins) #这是频数直方图，若想计算频率直方图，添加参数density = Ture
plt.xticks(range(min(a),max(a)+d,d))
plt.grid(alpha = 0.4)
plt.show()
~~~

## basemap

首先是安装，由于我已经安装所以不再赘述

~~~python
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
map = Basemap()
map.drawcoastlines()
plt.show()
~~~

![image-20220730114847136](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220730114847136.png)

第一行代码引入了Basemap库和matplotlib，这两个都是必须的。
这张地图是由Basemap类创建的，这个类包含很多属性。如果使用默认值，则使用普通圆柱投影模式显示地图。
如果设置了属性，我们就能根据需要创建地图。在这个例子中，用drawcoastlines()方法画出海岸线，海岸线的数据已经默认包含在了库文件中。
最后使用mathplotlip中的方法显示和保存图片，在这个例子中，plt.show()打开一个新的窗口来显示运行结果，plt.savefig('file_name')把运行结果保存为图片。

改变地图的投影方式非常简单，只用在Basemap（）中加入projection参数和lat_0, lon_0参数。(关于更多的地图投影只是，可以参考百度百科https://www.baidu.com/link?url=d7mToqKOKCk3Ba60s2HtT-4pEuC4jzHhFqhytovCw2IKA6cu6GiBHuR-V7negpngoN8dKyHBNA8_8y8-GRs1tJ7Q6o2bwDLtuB9277b6L6UrIKTRv2DPJtbw87iv6NeA&wd=&eqid=8077e6880001a123000000025a600f50)

~~~python
即使使用新的地图投影方式，生成的地图还是丑的一逼，用下面的代码可以给陆地和海洋填上不同的颜色：

From mpl_toolkits.basemp import Basemap
Import matplotlib.pyplot as plt

Map = Basemap(projection = 'ortho', lat_0 = 0, lon_0 = 0)
# 首先给地球涂上蓝色的一层
map.drawmapboundary(fill_color = 'aqua')
# 再给大陆涂上屎黄色,给江河湖泊涂上大海一样的颜色
map.fillcontinents(color = 'coral', lake_color = 'aqua')
map.drawcoastlines()
~~~

利用epsg 设置地图投影

ESPG是一种标准的命名投影方式的数字编码。Basemap允许使用这些标记来创建地图，但只局限于某些特定的情况下。要使用ESPG标记，需要在Basemap（）里面加上epsg参数。

Basemap中的<python_packages_path>/mpl_toolkits/basemap/data/epsg对这种EPSG提供支持，但是有时使用这种方法还是会报错（ValueError: 23031 is not a supported EPSG code），所以不建议使用。

Basemap对带有"utm"的projection支持不太好，但是对带有"tmere"都能很好的支持。



~~~python
# 首先给地球涂上蓝色的一层
map.drawmapboundary(fill_color='aqua')
# 给大陆涂上珊瑚色,给江河湖泊涂上大海一样的颜色
map.fillcontinents(color='coral', lake_color='aqua')
# 画经纬度线
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
# 送上标题
plt.title("Mercator Projection")
~~~

### 管理地图投影

##### 1. projection参数用来设置投影模式

~~~python
map = Basemap(projection = 'cyl')
//注意：使用cyl，merc，mill，cea和gall投影时，如果没有设置边界，那么边界默认设置为-180，-90,180,90（也就是整个球体），其他投影都需要额外设置。
~~~

##### 2. 利用epsg设置地图投影，绘制台湾省

~~~python
map = Basemap(llcrnrlon=119.3, llcrnrlat=20.7, urcrnrlon=124.6, urcrnrlat=26, resolution='h', epsg=3415)

~~~

### 绘制区域地图

##### 1. 利用经纬度设置边界，绘制海南省

~~~python
map = Basemap(llcrnrlon=108.3, llcrnrlat=18, urcrnrlon=111.5, urcrnrlat=20.3, resolution='i', projection='tmerc', lat_0=20, lon_0=111)
//注意：使用sinu，moll，hammer，npstere，spstere，nplaea，splaea，npaeqd， spaeqd， robin， eck4， kav7，和 mbtfpq投影中，不能使用这种方法。一是因为在有些投影中，整个地球都被绘制出来，二是因为在有些投影中，无法通过地理坐标计算extension的值
~~~

##### 2. 利用地图设置边界

~~~python
map = Basemap(resolution='l', satellite_height=3000000., projection='nsper', lat_0=30, lon_0=116, llcrnrx=500000.,llcrnry=500000., urcrnrx=2700000., urcrnry=2700000.)
//注意：只有在ortho，geos和nsper投影中使用这种方法
~~~

##### 3. 通过中心坐标和长度、宽度

~~~python
# 中心维度0，经度90，图片宽度10000000，高度10000000
map = Basemap(projection='aeqd', lat_0=0, lon_0=90, width=10000000, height=10000000.)
~~~

**4.部分参数**

drawcoastlines(): 绘制海岸线。
fillcontinents(): 通过填充海岸线多边形为地图着色。
drawcountries(): 绘制国家边界。
drawstates(): 在北美绘制状态边界。
drawrivers(): 绘制河流。
此外，可以将图像用作地图背景，而不是绘制海岸线和政治边界。Basemap提供了以下几个选项：

drawlsmask(): 绘制高分辨率的海陆图像，指定陆地和海洋的颜色，数据源于GSHHS海岸线。
bluemarble(): 绘制NASA蓝色大理石图像作为地图背景。
shadedrelief(): 绘制阴影浮雕图像作为地图背景。
etopo(): 绘制一张etopo浮雕图像作为地图背景。
warpimage(): 使用abitrary图像作为地图背景，必须是全球新的，从国际日东线向东和南极以北覆盖世界。

### 基本函数

##### 1. 在地图上画一个点

~~~python
map = Basemap(projection = 'ortho', lat_0 = 0, lon_0 = 120, width = 10000000,height = 10000000.)
# x是经度，y是纬度 
x, y = map(114, 22.4) map.plot(x, y, marker = 'D', color = 'm')
~~~

**2.在地图上画连续的点**

~~~python
m.drawcoastlines()
x, y = m(x1_longitude, y1_latitude)
m.scatter(x, y, marker = 'D', color = 'm',label = "任务完成")
x, y = m(x2_longitude, y2_latitude)
m.scatter(x, y, marker = 'D', color = 'c',label = "任务未完成")
plt.show() 
~~~

完整示意图代码：

~~~python
plt.figure(figsize=(10, 8))
m = map = Basemap(llcrnrlon=112.5, llcrnrlat=22, urcrnrlon=114.5, urcrnrlat=24, resolution='i', projection='tmerc', lat_0=y_centre, lon_0=x_centre)
m.shadedrelief(scale=0.5)
m.drawcoastlines()
x, y = m(x1_longitude, y1_latitude)
plt.scatter(x, y, marker = 'D', color = 'm',label = "任务完成")
x, y = m(x2_longitude, y2_latitude)
plt.scatter(x, y, marker = 'D', color = 'c',label = "任务未完成")
plt.legend()
m.drawcoastlines()#绘制海岸线
m.warpimage()#绘制国家边界
m.drawrivers(color='blue',linewidth=0.3)#绘制河流
m.shadedrelief()#绘制阴影浮雕图像作为地图背景
m.drawmeridians(np.arange(112.5,114.5, 0.5),labels=[0,0,0,1], fontsize=10,)#绘制经线
m.drawparallels(np.arange(22, 24, 0.5),labels=[1,1,1,0], fontsize=10)#绘制纬线
plt.show() 
~~~

![image-20220730151552183](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220730151552183.png)
