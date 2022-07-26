# Numpy

- 帮助我们处理数值型数据

特点：

1. 快速
2. 方便
3. 科学计算的基础库

- numpy重在数值计算，也是大部分python科学计算库的基础库，多用于大型，多维数组上执行数值运算。

## numpy创建数组(矩阵)

~~~python
import numpy as np
#使用numpy生成数组，得到ndarray的数据类型
t1 = np.array([1,2,3])    #array(numpy的一种数组类型)
print(t1)   #结果为[1 2 3]
print(tpye(t1))    #结果为array类型

t2 = np.array(range(10))
print(t2)    #结果为[0,1，2，3，4，5，6，7，8，9]

t3 = np.arange(4,10,2)  #相当于是list(range(10))
print(t3)#numpy中一种独有的生成数组的方法
#结果为[4，6，8]
~~~

- dtype:打印出数组中的数据的固有类型

~~~python
print(a.dtype)
#结果可能为int8  int32  int64 等 int为数据类型，后边的数字为所占的位数
~~~

##### numpy 的数据类型

~~~python
t4 = np.array(range(1,4),dtype = float)
t4 = np.array(range(1,4),dtype = "float")
t5 = np.array([1,1,0,1,0,0],dtpye = bool)
~~~

##### 调整数据类型

- astype

~~~python
t4 = np.array([1,1,0,1,0,0],dtype = bool)
#结果为[True True False True False False]
t6 = t5.astype('int8')   #改变t5的数据类型
print(t6)
print(t6.dtype)
~~~

##### numpy中的小数

- 保留小数round

~~~python
t7 = np.array[random.random()    for i in range(10)]
print(t7)
print(t7.dtype)
t8 = np.round(t7,2)  #保留两位小数
print(t8)
~~~

![numpy的运行结果](C:\Users\DELL\Desktop\work\picture\numpy的保留两位小数.png)

#### 数组的形状

np.shape

~~~python
t1 = array([1,2,3,4,5,6,7,8,9,10])
t1.shape  #结果为（10,）表示为有10个元素
t2 = np.array([[1,2,3],[4,5,6]])
print(t2)  # 结果为([[1,2,3],
                   [4,5,6]])
t2.shape #结果为(2,3) ,表示有两行，4列的**矩阵**
~~~

- 以上代码的t1为一维数组，t2为二维数组

![numpy的数组类型](C:\Users\DELL\Desktop\work\picture\numpy的数组类型.png)

- reshape修改数组形状

~~~python
t4 = np.arange(12)
t4 = array([0,1,2,3,4,5,6,7,8,9,0,11])
t4.reshape((3,4))
//结果t4 = array([[0,1,2,3],
       [4,5,6,7],
       [8,9,10,11]])
#能分成几组就写几组，不能够出现数据不够用
~~~

~~~python
t5 = np.arange(24).reshape((2,3,4))
//结果t5 = array([[[0,1,2,3],
                  [4,5,6,7],
                  [8,9,10,11]],
                
                [[12,13,14,15],
                 [ 16,17,18,19],
                 [20,21,22,23]]])
  #这是三维数组，其中的shap(2,3,4)中的2表示有几个区块，（3，4）表示在每个区块中的行列数。  
~~~

- t5.reshap((4,6))的运行结果

![reshape](C:\Users\DELL\Desktop\work\picture\numpy的reshape.png)

- 注意：t5还是原来的t5，t5.reshape产生一个新的数组，但是对t5本身不会产生任何改变。

- 变成一维数组 t5.reshape((24)，)

- 在不清楚一个二维数组的具体行列数，将其变成一维数组。

~~~python
t6 = t5.reshape((t5.shape[0]*t5.shape[1],))
#t5.shape[0]表示t5的行数，t5.shape[1]表示t5的列数。
**另一种简单的实现方法**
t5.flatten()
~~~

~~~python
t5+2，t5*2,t5/2,
#并不会报错，他的意思是在数组上的所有的值上边都加上2，生成为广播机制。
t5/0 ,返回nan，naf等翻译成汉语是，不是一个数字，为单独的一个值，inf为无限，无穷的意思，相当于一个数除以一个非常非常小的数接近于0，相当于求极限。
~~~

### 数组和数组进行计算

~~~python
t6 = np.arrange(100,124).reshape((4,6))
~~~

![t6](C:\Users\DELL\Desktop\work\picture\nunpy.t6.png)

t5

![t5](C:\Users\DELL\Desktop\work\picture\numpy.t5.png)

~~~python
#进行相加操作
t6+t5
#结果为相应数组进行相加
~~~

![数组相加](C:\Users\DELL\Desktop\work\picture\数组及相加.png)

但如果操作的数组是不同形状呢？

~~~python
t7 = np.arange(0,6)
#t7 = array([0,1,2,3,4,5])
t5-t7
~~~

结果为：

![t5 - t7](C:\Users\DELL\Desktop\work\picture\不同形状数组操作.png)

**由此可知，当两个不同形状的数组项操作时，对应同一维度的数组会进行相应操作，t7为一维数组t7.shape为（（6），），t5的shape为（（4，6）），在操作时，相减会发生在每一横行，因为其在同一维度。**

再比如说：

~~~python
t8 = np.arange(4).reshape((4,1))
#t8 = array([[0],
#             [1],
#             [2],
#            [3]])
~~~

t5-t8:

![t5-t8](C:\Users\DELL\Desktop\work\picture\t5-t8.png)

**同上，只是此数据为每一列进行相操作,但是，需要注意的是，他们能够计算是有前提的，必须是有相同的行数或者列数，多出来之后就不行了**

#### 数组转置

~~~python
t2 = np.arrange(24).reshape((4,6))
t2.transpose()
或者 t2.swapaxes(1,0)将0轴和1轴交换
~~~

运行结果:

![transpose](C:\Users\DELL\Desktop\work\picture\numpy.transpose.png)

***

### 轴

![numpy的轴](C:\Users\DELL\Desktop\work\picture\numpy的轴.png)

- 二维数组的轴

![二维数组的轴](C:\Users\DELL\Desktop\work\picture\二维数组的轴.png)

- 三维数组的轴

![三维数组的轴](C:\Users\DELL\Desktop\work\picture\三维数组的轴.png)

## numpy读取数据

***

~~~python
np.loadtxt(frame,dtype = np.float,delimiter = None,skiprows = 0,usecols = None,unpack = False)
~~~

- 注意:从文本文件读取内容,frame是文件路径,,dtype = np.float,是读文件成什么类型,delimiter是文件用什么分割开的,skiprows = 0,表示跳过第一行,usecols = None,使用哪几列,unpack转至.

![numpy读取数据](C:\Users\DELL\Desktop\work\picture\numpy读取数据1.png)

- 补充:unpack是转至效果,将数据进行行转列,列转行,有一定的旋转作用效果。

***

## numpy的索引和切片

- 取行

~~~python
print(t2[2])
~~~

- 取连续的多行

~~~python
print(t2[2:])
~~~

- 取不连续的多行

~~~python
print(t2[[2,8,10]])
~~~

总结：

在如下操作中，[,]逗号之前的为行，逗号之后为列。

~~~python
print(t2[1,:]) #1表示行取第二行，：表示列都要
print(t2[2:,:])#2：表示从第三行到最后一行，：表示列都要
print(t2[[2,10,3],:])#表示取第三，十一，四行，：表示列都要
print(t2[:,1])#表示行都要，取第一列
print(t2[:,3:])#表示行都要，取第四列后的每一列
print(t2[:,[0,2]])#表示行都要，取第一列和第三列
print(t2[2,3])#取第一行第二列
print(t2[2:5,2:4])#取第三到五行，第三到第四列的交叉点的位置
print(t2[[0,2,2],[0,1,3]])#取[0,0],[2,1],[2,3]三个点
~~~

- 总结：所取值与元组类似，用[ ]括起来，而不用的是，[,]逗号之前的为行，逗号之后为列。当取单独的行时，还得用[ ]括起来。有：时不用[ ],当用到，时且为取多行多列，则用到[ ]。

***

![赋值操作](C:\Users\DELL\Desktop\work\picture\numpy赋值相关操作.png)

##### numpy中的布尔索引

~~~python
t2[t2<10]=3 #是将t2中所有小于10的数变为3,这种情况下，t2的值发生替代和改变。
t2[t2>20] #--->array([21,22,23])
~~~

#####  numpy中的三元运算符

~~~python
t2.np.where(t2<=3,100,300) #将t2中<=3的替换成100，其他的替换成300
~~~

##### numpy中的clip(裁剪)

~~~python
t2.clip(10,18) #将t2中小于10的替换成10，大于18的替换成18。
~~~

***

## 数组的拼接

~~~python
np.vstack((t1,t2))#竖直拼接，添加行数
np.hstack((t1,t2))#水平拼接,添加列数
~~~

![水平竖直拼接](C:\Users\DELL\Desktop\work\picture\numpy的水平竖直拼接.png)

## 数组的行列交换

~~~python
t[[1,2],:] = t[[2,1],:]#行交换
t[:,[1,2]] = t[:,[2,1]]#列交换
~~~

- 创建一个全为0的数组

~~~python
np.zeros((t2.shape[0],1)) #为浮点数
~~~

- 创建一个全为1的数组

~~~python
np.ones((t2.shape[0],1))  #为浮点数
~~~

创建与拼接实例：

![实例](C:\Users\DELL\Desktop\work\picture\水平竖直拼接实例.png)

- 创建一个对角线为1的正方形数组（方阵） np.eye(3) #创建一个三行三列的数组，对角线为1。

#### 获取最大值最小值

np.argemax(t,axis = 0)

解析：

创建一个4*4的正方形矩阵t

~~~python
t = np.eye(4)
~~~

![4*4](C:\Users\DELL\Desktop\work\picture\正方形矩阵.png)

取最大值

~~~python
np.argmax(t,axis = 0) #表示取t这个矩阵的最大值，axis = 0表示在行上取最大值，则为每一列的最大值。且为([0，1，2，3])0，1，2，3表示为每一列最大值的具体位置。
~~~

取最小值

~~~python
np.argmin(t,axis = 1) #表示取t这个矩阵的最小值，axis = 1表示在列上取最小值，则为每一行的最大值。且为([0，1，2，3])0，1，2，3表示为每一行最小值的具体位置。
~~~

#### numpy生成随机数

~~~python
np.random
~~~

![生成随机数](C:\Users\DELL\Desktop\work\picture/numpy生成随机数.png)

~~~python
np.random.randint(10,20,(4,5))  #表示创建一个四行五列的数组，其中数值为10到19之间的随机数。
~~~

~~~python
np.random.seed(10) 
np.random.randint(10,20,(4,5))
#使用seed操作为随机种子，能够达到在后续每次随即得到数据时，会重复相同的形状和结果，重复seed()中的数据次数。
~~~

![copy和view](C:\Users\DELL\Desktop\work\picture\视图和复制.png)

***

## numpy中的nan和inf

![nan和inf](C:\Users\DELL\Desktop\work\picture\nan和inf.png)

- 注意：nan和inf都是浮点类型

特殊属性：

![nan的特殊属性](C:\Users\DELL\Desktop\work\picture\nan的特殊属性.png)

![nan的特殊用法](C:\Users\DELL\Desktop\work\picture\np.nan不等于np.nan.png)

- 只有nan的地方为True

~~~python
np.isnan(t2) #判断数组中有多少个值为nan，他就是np.nan!=np.nan
~~~

- nan和任何值计算都是nan

![nan的计算](C:\Users\DELL\Desktop\work\picture\nan的计算.png)

numpy中常用统计函数

![numpy常用统计函数](C:\Users\DELL\Desktop\work\picture\numpy常用统计函数.png)

~~~python
import numpy as np
t1 = np.arange(12).reshape((3,4)).astype('float')
t1[1,2:] = np.nan
def fill_ndarrat(t1):
   for i in range(t1.shape[1]);
       temp_col = t1[:,i]
       nan_num = np.count_nonzero(temp_col!=temp_col)
       if nan_num !=0; #说明当前这一列中有nan
       temp_not_nan_col = temp_col[temp_col==temp_col] #当前这一列不为nan的array
       temp_not_nan_col.mean()
       temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean() #选中当前nan的位置,把值赋值为不为nan的均值.
    return t1
if __name__ == '__main__':
    t1 = np.arange(12).reshape((3,4)).astype('float')
    t1[1,2:] = np.nan
    print(t1)
    t1 = fill_ndarray(t1)
    print(t1)
~~~

### 矩阵（数组）计算方法

- 注意：在计算内积，点积时，必须行列数完全相同。

**Numpy的数组内积函数 inner 以及矩阵内积函数 matmul**

1.数组内积 inner()函数：

数组的内积是两个数组的行与行乘积和运算，同上有数组A和B，则计算公式如下。

C[0,0]=A[0,0] *B[0,0] + A[0,1] *B[0,1]；

C[0,1]=A[0,0] *B[1,0] + A[0,1] *B[1,1]；

C[1,0]=A[1,0] *B[0,0] + A[1,1] *B[0,1]；

C[1,1]=A[1,0] *B[1,0] + A[1,1] *B[1,1]；

~~~python
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
print("a")
print(a)
print("b")
print(b)
d=np.inner(a,b)
print("d")
print(d)
二维数组a，b：
a
[[1 2]
 [3 4]]
b
[[5 6]
 [7 8]]
#输出结果：
[[17 23]
 [39 53]]
~~~

2. 矩阵内积matmul()函数：

假如有矩阵A和矩阵B他们都是两行两列，则内积计算结果也为两行两列的一个矩阵，假设C为内积矩阵，计算公式如下：

C[0,0]=A[0,0] *B[0,0] + A[0,1] *B[1,0]：A的第一行与B的第一列，对应元素的乘积之和;

C[0,1]=A[0,0] *B[0,1] + A[0,1] *B[1,1]：A的第一行与B的第二列，对应元素的乘积之和;

C[1,0]=A[1,0] *B[0,0] + A[1,1] *B[1,0]：A的第二行与B的第一列，对应元素的乘积之和;

C[1,1]=A[1,1] *B[0,1] + A[1,1] *B[1,1]：A的第二行与B的第二列，对应元素的乘积之和;

~~~python
a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])
c=np.matmul(a,b)
print("c")
print(c)
##输出结果：
c
[[19 22]
 [43 50]]
~~~

3. 数组点积dot()函数：

1）一维数组的点积：一维数组的点积就是计算两个数组对应位置乘积之和，假如有两个一维数组a和b，那么他们的点积计算结果为一个数值，计算公式为a[0] * b[0] +a[1] * b[1 +...+a[n]*b[n]。

~~~python
#一维数组的dot
a=np.array([1,2,3,4])
b=np.array([5,6,7,8])
print("a")
print(a)
print("b")
print(b)
print(np.dot(a,b))
#输出结果：
70
~~~

2)二维数组的点积：

二维数组的点积相对复杂一些，假如有数组A和数组B他们都是两行两列，则点积计算结果也为两行两列的一个数组，假设点积数组为C，计算公式如下：

C[0,0]=A[0,0] *B[0,0] + A[0,1] *B[1,0]：A的第一行与B的第一列，对应元素的乘积之和;

C[0,1]=A[0,0] *B[0,1] + A[0,1] *B[1,1]：A的第一行与B的第二列，对应元素的乘积之和;

C[1,0]=A[1,0] *B[0,0] + A[1,1] *B[1,0]：A的第二行与B的第一列，对应元素的乘积之和;

C[1,1]=A[1,1] *B[0,1] + A[1,1] *B[1,1]：A的第二行与B的第二列，对应元素的乘积之和;
**此计算结果与上述矩阵内积结果相同**

~~~python
#定义两个点积数组
a=np.array([[3,4],[5,6]])
b=np.array([[13,14],[15,16]])
print("a")
print(a)
print("b")
print(b)
#输出结果：
a
[[3 4]
 [5 6]]
b
[[13 14]
 [15 16]]
#点积dot 操作
c=np.dot(a,b)
print("c")
print(c)
#输出结果
c
[[ 99 106]
 [155 166]]
~~~

一维数组：

~~~python
[1,2,3]
~~~

二维数组(矩阵）：

~~~python
[[1,2,3],[4,5,6]]

[[1,2,3],
 [4,5,6]]
~~~

一维标签数组：（Series）

~~~python
0   0.999
1   1.233
2   3.566
3   66.5
4   34.112
~~~

二维表（pd.DataFrame())

只有DataFrame的形式才有.head()

数组的形式只有.shape
