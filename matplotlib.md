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
#数据可视化：
from matplotlib import pyplot as plt
import matplotlib
import random
#设置图像
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False
fig = plt.figure(figsize = (12,10),dpi = 100)
x = range(11,31)
y = [1,0,1,1,2,4,3,2,3,4,4,5,6,5,4,3,3,1,1,1]
_x = list(x)
_xsticks_xlabel = ["{}岁".format(i)  for i in x]
plt.xticks(_x[::3],_xsticks_xlabel[::3])
plt.yticks(range(0,8))
plt.xlabel("年龄")
plt.ylabel("个数")
plt.title("年龄与个数的图例")
plt.plot(x,y)
plt.show()
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

