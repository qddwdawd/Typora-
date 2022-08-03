python统计分析

### 第一节：描述性统计分析

![image-20220801174527617](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801174527617.png)

![image-20220801174638326](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801174638326.png)

![image-20220801174701350](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801174701350.png)

![image-20220801174803422](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801174803422.png)

~~~python
import pandas as pd
import numpy as np
from scipy import stats
import os
insuance = pd.read_excel(r"C:\Users\DELL\Desktop\数学建模\2017国赛\B\附件二：会员信息数据.xlsx")
insuance
~~~

![image-20220801182110649](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801182110649.png)

~~~python
insuance['age'].median()#中位数
~~~

~~~python
insuance['age'].quantile([0,0.05,0.25,0.5,0.75])#百分数
~~~

![image-20220801182244552](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801182244552.png)

~~~python
insuance['age'].mode()#众数
~~~

![image-20220801182445122](image-20220801182445122.png)

~~~python
#分类变量
~~~

![image-20220801182554231](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801182554231.png)

~~~python
#离散性数据
insurance['age'].max() - insurance['age'].min()#极差
~~~

![image-20220801182926299](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801182926299.png)

~~~python
#四分位差
insuance['age'].quantile(0.75) - insuance['age'].quantile(0.25)
~~~

![image-20220801183036575](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801183036575.png)

![image-20220801183101729](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801183101729.png)

![image-20220801185444856](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801185444856.png)

![image-20220801185556928](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801185556928.png)

![image-20220801185703807](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801185703807.png)

![image-20220801185959478](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801185959478.png)

![image-20220801190134630](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801190134630.png)

~~~python
insurance['charges'].mean()
~~~

![image-20220801190403804](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801190403804.png)

~~~python
se = stats.sem(insurance['charges'])#计算样本均值标准误差
~~~

![image-20220801190530266](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801190530266.png)

![image-20220801190720514](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801190720514.png)

- 计算标准误差，std算的是总体标准偏差。

![image-20220801190929218](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801190929218.png)

- 0.95的置信区间。

![image-20220801191232059](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801191232059.png)

- 经过手动计算，算出来的置信区间左侧未12621.5300和上面算法相差无几。

![image-20220801191321297](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801191321297.png)



### 第二节：假设检验

![image-20220801191538987](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801191538987.png)







### 第三节：卡方分析和方差分析

![image-20220801194213943](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801194213943.png)

![image-20220801194415462](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801194415462.png)

![image-20220801201915611](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801201915611.png)

![image-20220801201925250](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801201925250.png)

# Python中的各种检验

### levene检验、T检验、卡方检验、F检验

#### 1.levene检验

![image-20220801210912667](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801210912667.png)

~~~python
from scipy import stats
stats.levene(x, y) #检测两项之间的方差齐性。#x,y必须是一维。
#当结果大于0.05时，认为方差是相等的，当结果小于0.05时认为不相等。
~~~

![image-20220801211258936](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801211258936.png)

#### 2.F检验

- F检验又叫方差齐性检验。在两样本t检验中要用到F检验
- 从两研究总体中随机抽取样本，要对这两个样本进行比较的时候，首先要判断两总体方差是否相同，即方差齐性。若两总体方差相等，则直接用t检验，若不等，可采用t"检验或变量变换或秩和检验等方法。

~~~python
model1 = SelectKBest(f_classif, k=2)#选择k个最佳特征  
model1.fit_transform(x, y)
model1.pvalues_
#p>0.05:方差具有齐次性，p<0.05:方差无差异
~~~

![image-20220801212240864](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801212240864.png)

![image-20220801212322622](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801212322622.png)

![image-20220801213546721](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801213546721.png)

#### 3.卡方检验

~~~python
scipy.stats.chi2_contingency([x1,x2])[1]#取0为卡方值，取1为p_value
#p_value越大说明相关性越小，值越小说明他们的相关性越大。通常是检验自变量与标签之间的关系。
#在机器学习中
model1 = SelectKBest(chi2, k=2)#选择k个最佳特征  
model1.fit_transform(x, y)
model1.pvalues_
~~~

![image-20220801211817126](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801211817126.png)



![image-20220801212151023](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801212151023.png)

#### 4.T检验

~~~python
stats.ttest_ind(normal[:, i+1], cancer[:, i+1], equal_var=False) 
~~~

![image-20220801213707953](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801213707953.png)

![image-20220801213828712](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801213828712.png)

![image-20220801214616188](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801214616188.png)

![image-20220801215441737](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801215441737.png)

![image-20220801215557256](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801215557256.png)

~~~python
#T检验还需要满足正态性检验，所以可以进行Q-Q图，直方图的检验。
~~~

#### 5.数据的归一化

- 数据归一化可以让数据符合正太分布，利于进行正态性检验。

#### 6.Box_Cox

这一变换可以使得线性回归模型在满足线性、正态性、 独立性‘方差齐性的同时又不丢失信息，在变换之后可以一定程度上减小不可观测的误差 和预测变量的 相关性，有利于线性模型的拟合以及分析出特征的相关性。#只能是正值且不能为0，所以先进行标准化，然后取abs绝对值。

~~~python
scipy.special.boxcox1p(x, lmbda)

from scipy.stats import boxcox 
scipy.stats.boxcox(x, lmbda=None, alpha=None)  #一列一列进行转化，通过迭代。
~~~

![image-20220801215216180](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220801215216180.png)

