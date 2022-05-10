# 处理excel表格数据

~~~python
import pandas as pd
import dateparser  #将字符串解析成日期格式
~~~

- 解释参数：dateparser:它可以用来将字符串解析成日期格式，比如：dateparser.parse("2018年08月13日 10: 23: 20")结果为：2018-08-13 10: 23: 20 
- apply(func,*args,**kwargs):用途：当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并将元组或者字典中的参数按照顺序传递给参数。在实例中，因为data['购药时间']表示的是一列，他是一个列表形式，而apply的作用是将这个列表中的值一个接一个的依次传给后边的参数。
- lambda x:它可以看成是定义一个简单的函数，冒号后边是函数操作，但只能有一个即：lambda x: dateparser.parse(x,languages=['zh'])，它是将定义函数中的x当作自变量，传入后边进行计算。
- 这样就组成了：data['购药时间'].apply(lambda x: dateparser.parse(x,languages=['zh']))，这个表达式的意思是：利用apply函数将data[”购药时间“]中的参数依次传给lambda x: ，再利用dateparser.parse，将x进行int类型转化，实现时间的int化。

首先读取数据：

~~~python
data = d.read_excel("./xlsx"),sheet_name='sheet1')
#丢弃数据未na的空白数据
data=data.dropna( ) #效果不是很好
~~~

~~~ python
# 对部分分数数据进行格式化，譬如将购药时间字符串“2018-01-01 星期五”转换成日期格式的“购买时间——date"。格式化之后是产生一列新的数据。
data["社保卡号_int"] = data["社保卡号"].astype("int")#将科学计数法转化成int形式。
data["购药时间_date"]=data['购药时间'].apply(lambda x: dateparser.parse(x))
data["购药时间_month"] = data["购药时间_date"].apply(lambda x:x.month)
~~~

~~~python
#按照”购药时间——data“进行排序，预览一下数据
data.sort_values(by = "购药时间_date")
~~~

按照商品名称进行数据透视，对['销售数量'，'应收金额'，"实收金额"]进行求和

~~~python
sales_volume = pd.pivot_table(data,index = ['商品名称']，values=['销售数量','应收金额','实收金额'],aggfunc = 'sum',fill_values)
~~~

将数据就按照”销售数量“进行排序

~~~python
sales_volume_sorted = sales_valume.sort_values(by = "销售数量",ascending = False)
~~~

### 统计不同分数段学生人数

~~~python
import pandas as pd
data['分段']=pd.cut(data['高等数学A-1'],
                 [0,60,80,90,101],
                 labels = ['不及格','[60~80)','[80~90)','[90~100]'],
                right= False)  #在数据后面加一行data[’分段‘]
data["分段"].value_counts()  #统计这一列的不同分数段的人
~~~

value_count的用法：

~~~python
#他是统计你所用的列的不同数据的个数
~~~

- 例如：(批量统计)

- ![图标的value](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\图表values的用法.png)

- ~~~python
  data['Unit Name'].value_counts()
  #输出
  Percent of GDP                  3561
  Domestic currency               3561
  Percent of total expenditure     470
  Name: Unit Name, dtype: int64
  ###这个value_counts()的方法可以统计多行
  ~~~

##### 部分数据相加

~~~python
data_1=[]
for i in data['工程图学A']:
    for j = data["C语言程序设计"]:
        for k =data["通信专业导论"]:
            a = i+j+k
            data_1.append(a)
            
~~~



### 关于Nan的处理方法

#空值处理方式

~~~python
data.fillna(data.mean(), inplace = True)#以均值填充
data.fillna(0, inplace = True)#以0填充
# 加上 inplace=True 表示修改原对象
data.fillna('新值', inplace=True)#将nan换成你想要的数
~~~

nan统计方法：

~~~python
print(np.count_nonzero(data)) #获得nan的数量
print(data.isnull().any())  #判断哪些”列”存在缺失值
data[data.isnull().T.any().T]) #找出含有nan的所有行
~~~

***

##### pandas中的删除操作

~~~python
# 通过列名删除指定列
data.drop(['序号', '替代', '签名'], axis=1, inplace=True)
~~~

~~~python
#通过行名删除指定的行
data.drop(index = 1 )
~~~

***

##### 读取行列名称

~~~python
data._stat_axis.values.tolist() # 行名称
data.columns.values.tolist()    # 列名称
~~~



### excel表格批量替换

- 普通替换，即利用索引，将表中所有的数据进行批量替换

~~~python
for i in data["工程图学A"]: #中输入想要进行替换的值
    if i == 77.0:
        i = 70
data
~~~

- 替换行列名称

~~~python
#更改列的名称
data = data.rename(columns={"工程图学A":"工程图学"})
data
~~~

~~~python
#删除重复项
df["city"].drop_duplicates()
df["city"].drop_duplicates(keep="last")
~~~

~~~python
#数字修改和替换
data["city_1"]=data["city"].replace("sh","shanghai") #在后面添加一行
data = data["city"].replace("sh","shanghai") 
~~~

详细见[img]https://www.cnblogs.com/feily/p/14397470.html

