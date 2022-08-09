# pandas

- 为了处理数值型数组据以外的非数值型数据，含有字符串，时间序列。

##### pandas常用的数据类型

1. Series 一维，带标签数组
2. DataFrame 二维，Series容器

~~~python
t = pd.Series([1,2,31,12,3,4])
t = pd.Series([1,23,2,2,1],index = list("abcde"))
~~~

结果为：

~~~python
0  1  和  a  1
1  2      b  23
2  31     c  2
3  12     d  2
4  3      e  1
5  4         #前面的0，1，2，3，4，5为标签
~~~

~~~python
temp_dict = {"name":"xiaohong","age":30,"tell":10086}  //字典
t3 = pd.Seise(temp_dict)
print(t3)
~~~

- 结果为：

![pandas结果](C:\Users\DELL\Desktop\work\picture\pandas的Series.png)

#### Serier切片和索引：

![切片和索引](C:\Users\DELL\Desktop\work\picture\pandas切片和索引.png)

- 取多行：t3[2:],t3[1,2]

- bool索引:t[t>10]

##### pandas的index

![index](C:\Users\DELL\Desktop\work\picture\pandas的index.png)

***

## pandas之读取外部数据

~~~python
import pandas as pd
df = pd.read_csv("./diabetes (2).csv",sheet_name ) #表示读取当前文件夹下的diabetes (2).csv文件，且最好是当前文件夹下的文件,右击复制绝对路径
~~~

运行结果：

![运行结果](C:\Users\DELL\Desktop\work\picture\pandas 的pd,read_csv.png)

***

## pandas之DataFrame

~~~python
import pandas as pd
pd.DataFrame(np.arange(12).reshape(3,4))
~~~

运行结果：

![pandas的DataFrame](C:\Users\DELL\Desktop\work\picture\pandas的Data.Frame.png)

- index表示行索引，index ，axis = 0,表示的是第一列，取每一行的第一个
- columns表示列索引，columns ,axis = 1,表示的是第一行，取每一列的第一个

~~~python
pd.DataFrame(np.arange(12).reshape(3,4),index = list("abc"),columns = list("wxyz"))
~~~

### pandas 的DataFrame的字典操作

~~~python
d1 = {"name":["xiaoming","xiaogang"],"age":[20,32],"tel":[10080,10086]}
pd.DataFrame(d1)
~~~

![运行结果](C:\Users\DELL\Desktop\work\picture\pandas 的DataFrame的字典操作.png)

~~~python
d2 =[ {"name":"xiaohong","age":32,"tel":10000},{"name":"xiaoming","age":20,"tel":10086},{"name":"xiaogang","age":40}]
t1 = pandas.DataFrame(d2)
print(t1)
~~~

- DataFrame的基础属性：

![DataFrame的基本属性](C:\Users\DELL\Desktop\work\picture\DataFrame的基本属性.png)

- 排序

~~~python
df.sort_values(by="Count_AnimalName")
~~~

##### pandas 取行或者列注意事项：

- 方括号写数组表示取行，对行进行操作
- 写字符串，表示的取列索引，对列进行操作 

![pandas取行取列操作](C:\Users\DELL\Desktop\work\picture\pandas取行取列操作.png)

实际操作：

![实际操作](C:\Users\DELL\Desktop\work\picture\pandas取行列实际操作.png)

取多行多列操作：

![多行多列](C:\Users\DELL\Desktop\work\picture\DataFrame多行多列操作.png)

- df.loc 通过**标签**索引行数据
- df.iloc通过**位置**获取行数据

![通过iloc位置索引](C:\Users\DELL\Desktop\work\picture\pandas的DataFrame中的iloc.png)

- 索引之后就可以进行赋值。

## pandas之布尔索引

~~~python
print(df[(800<df["Count_AnimalName"])&(df["Count_AnimalName"]<1000)])
~~~

***

##### pandas 的drop函数

IN [1]: data
Out[1]: 
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11

IN [2]: data.drop(index=0) #删除index=0的行
Out[2]:  
   A  B   C   D
1  4  5   6   7
2  8  9  10  11

IN [3]: data.drop(labels=0, axis=0) #删除 "行号为0" 的行
Out[3]:  
   A  B   C   D
1  4  5   6   7
2  8  9  10  11

# ipynb转html

jupyter nbconvert --to html --template full C:\Users\DELL\Desktop\Jupyter.python\西北大学.ipynb
