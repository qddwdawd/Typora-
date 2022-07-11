机器学习（sklean）

### 决策树

决策树是将一张表总结成一个非常简单的树来让我们进行非常简单的操作。

- 本质：对特征进行一系列的提问。

![决策树](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\决策树.png)

![几种不同的树](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\几种不同的树.png)

1. 实例化，建立评估模型对象
2. 通过模型接口训练模型
3. 通过模型接口提取需要的信息

~~~python
from sklearn import tree     #导入需要的模块

clf = tree.DecisionTreeclassifier()  #实例化
clf=clf.fit(x_train,y_train)   #用训练集数据训练模型
result = clf.score(x_test,y_text)  #导入测试集，从接口中调用需要的信息

~~~

***



### Decision Tree Classifier（分类树）

分类树参数较多，用基尼系数或者信息熵来衡量。

### 2.1重要参数

2.1.1 criterion

不纯度越低，决策树对训练集的拟合越好，不纯度基于节点来计算，树中的每一个节点会有一个不纯度。不纯度基于节点来计算，树中的每个节点都有一个不纯度，并且子节点的不纯度一定是低于父节点的。在同一棵决策树上，叶子结点的不纯度一定是最低的。

Criterion这个参数正是用来决定不纯度的计算方法，sklearn提供了两种选择：

1）输入“entropy”，使用信息熵（entropy）

2）输入“gini'",使用基尼系数（Gini impurity）

![criterion](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\criterion.png)

- 比起基尼系数，信息熵对不纯度更加敏感，对不纯度的惩罚最强。但是在实际使用中，信息熵和基尼系数的效果基本相同。信息上的计算比基尼系数缓慢一些，因为基尼系数的计算不涉及对数。另外，因为信息熵比基尼系数更加敏感，所以信息熵作为指标时，决策树生长会更加”精细“，因此对于高维数据或者噪声很多的数据，信息熵往往会过拟合，基尼系数在这种情况下效果往往会比较好。当然，这不是绝对的。

- 噪音很大，数据维度很大的时候使用基尼系数
- 维度低，数据比较清晰的时候，信息熵和基尼系数没区别
- 当决策树的拟合程度不够的时候，使用信息熵两个都是是，不好就换另一个

![决策树的基本流程](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\决策树基本流程.png)

知道没有更多的特征可用，或者整体的不纯度指标已经最优化。

~~~python
#导入需要的包
from sklearn import tree   #导入树模型
from sklearn.datasets import load_wine #导入数据集
from sklearn.model_selection import train_text_split #导入分训练集和测试集的模块
wine = load_wine() #导入数据，但是数据中由数据标签，数据的名字等各种各样的东西
wine.data #调出需要的数据 ，真正写代码时，这是可以省略的这只是调出来看看
wine.traget()#调出需要的标签，真正写代码时，这是可以省略的这只是调出来看看
#如果是一张表格，用pandas结合
import pandas as pd
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)]，axis = 1) #利用pandas的DataFrame将wine.data数据集和wine,traget标签组合成一个列相加的二维数组
wine.feature_names #特征的名字，真正写代码时，这是可以省略的这只是调出来看看
wine.target_names  #标签的名字，真正写代码时，这是可以省略的这只是调出来看看
#分训练集和测试集
Xtrain,Xtest,Ytrain,Ytest=train_test_split(wine.data,wine.target,test_size=0.3) #利用导入的模块，分为训练集和测试集，它需要三个参数，为数据，标签和比例，百分之三十为测试集，百分之七十为训练集。
Xtarin.shap#可以看一下形状，经过操作后为二维数组模型，真正使用代码时不用敲这个代码
#进入建模部分
clf = tree.DecisionTreeClassifier(criterion = "entropy")#实例化，实例化的类取名叫clf,criterion默认为gini，基尼系数
clf = clf.fit(Xtrain,Ytrain)#训练模型
score = clf.score(Xtest, Ytest)#使用接口，导出我们所需要的分数，其实是对模型精确度的一个衡量，返回一个值
print(score)
~~~

### 接下来要画一棵树

~~~python
import graphviz #安装模块
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
dot_data = tree.export_graphviz(clf #训练过的模型
                               ,feature_names = feature_name
                               ,class_names =["琴酒","雪莉"，"贝尔摩德"]
                                ,filled = True
                                ,rounded= True)
graph = graphviz.Source(dot_data)
~~~

![运行结果](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\画一棵树.png)

由运行结果我们可以看出，我们所用的feature_names只用了4个特征，并没有完全使用。这证明决策树在建树的时候，有所取舍，并没有完全使用，所以它是由重要性的排名，我们可以调用clf.feature_importances_来看排名，数值越大越是重要，对这棵树来说有更加的意义。

![clf.feature_impoortance_](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\clf.feature_importance_.png)

- 使用[*zip(feature_name,clf.feature_importance_)]将特征名，和特征重要性装订起来,方便查看。

![*zip(feature.name,clf.feature_importance_)](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\[zip(feature.name,clf.feature_importance).png)

***

- 无论决策树的模型如何进化，在分支上的本质都还是追求某个不纯度相关的优化指标，而正如我们提到的，不纯度是基于节点来计算的，也就是说，决策树在建树时，是靠优化节点来追求一棵优化的 树。
- 用集成的方法来选取一颗最好的树,在分支的过程中，随机选取几个不同的特征，但是这样会导致参数的不稳定，利用random_state让参数永远不会变化。

~~~python
clf = tree.DecisionTreeClassifier(criterion = "entropy",random_state = 30)
clf = clf.fit(Xtrain,Ytrain)
score = clf.score(Xtest,Ytest)

score
~~~

#### random_state&splitter(控制决策树中的随机性)

~~~python
clf = tree.DecisionTreeClassifier(criterion="entropy"
                                  ,random_state = 30 #固定结果
                                  ,splitter = "random"#参数随机，加重随机性)
~~~

### 2.1.3剪枝参数

在不加限制的情况下，一棵决策树会生长到衡量不纯度的指标最优，或者没有更多的特征可以用为止。但是这样的决策树往往会过拟合，这就是说，**他在训练集上表现得很好，在测试集上却很糟糕。**我们收集的样本数据不可能和整体的状况完全一致，因此当一棵决策树队训练数据有了过于优秀的解释性，他找出的规则必然包含了训练集样本中的噪声，并且他对未知数据的拟合程度不足。

- 剪枝策略对决策树的影响巨大，正确的剪枝策略是欧化决策树算法的核心。

~~~python
max_depth
~~~

限制树的最大深度，超过设定深度的树枝全部剪掉。

这是用得最广泛的剪枝参数建议从3开始尝试，看看拟合效果。

~~~python
min_samples_leaf&min_samples_split
~~~

min_samples_leaf限定，一个节点在分支后的每个子节点都必须包含至少min_samples_leaf个训练样本，分支会朝着满足每个子结点都包含min_samples_leaf个样本的方向去发生。

- min_samples_leaf一般搭配max_depth来使用，在回归树中有神奇的效果，可以让模型变得更加平滑。这个参数的数量设置的太小会引起过拟合，设置的太大就会阻止模型学习数据。建议从 =5开始。(他是在样本确定之前进行的阈值处理。)

~~~python
min_samples_split
~~~

min_samples_split限定，一个节点必须要包含至少min_samples_split个训练样本，这个节点才会允许被分支，否则分支就不会发生。(他是在样本确定之后才进行的samples的判断是否能够分支)

~~~python
clf = tree.DecisionTreeClassifier(criterion="entropy"
                                 ,random_state=30
                                 ,splitter="random"
                                 ,max_depth=3
                                 ,min_samples_leaf = 10
                                 ,min_samples_split = 10
                                 )
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)
score
import graphviz
dot_data = tree.export_graphviz(clf
                               ,feature_names= feature_name
                               ,class_names=["琴酒","雪莉","贝尔摩德"]
                               ,filled=True
                               ,rounded=True
                               )  
graph = graphviz.Source(dot_data)
graph
~~~

~~~python
max_fratures&min_impority_decrease
~~~

max_feature限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。和max_depth异曲同工，max_features是同来限制高纬度数据的过拟合的剪枝参数，但其方法比较暴力，是直接限制可以使用的特征而强行使决策树停下的参数，在不知道决策树中各个特征的重要性的情况下 ，强行设定这个参数可能会导致学习不足。

~~~python
min_impurity_decrease
~~~

min_ impuirty _decrease限制信息增益的大小，信息增益小于设定数值的分支不会发生。信息增益是父节点的信息熵减子节点的信息熵，信息增益越大表示对整个信息越有效。

- 确定最优的剪枝参数

~~~python
import matplotlib.pyplot as plt
text = []
for i in range(10)
    clf = tree.DecisionTreeClassifier(max_depth = i+1
                                 ,criterion = 'entropy'
                                 ,random_state = 30)
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    text.append(score)
    
plt.plot(range(1,11),text,color = 'red',lablr = 'max_depth')
plt.legend()
plt.show()
~~~

![max_depth剪枝方法](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\决策树的max_depth剪枝方法.png)

- 剪枝参数不一定能够提升模型在测试集上的表现，调参没有绝对的答案，一切得看数据本身。

#### 重要的属性和接口

~~~python
clf.apply(Xtext)#返回每个测试样本所在的叶子结点的索引(及所在的位置)
clf.predict(Xtext)#返回每个测试样本的分类/回归结果(及返回标签)
~~~

***

## 随机森林

### 1.1集成算法的概述

集成算法的目标：集成算法会考虑多个评估器的建模结果，汇总之后得到一个综合的结果，以此来获取比单个模型更好的回归或分类表现。

多个模型集成成为的模型叫做集成评估器。

三类集成算法：

装袋法：构建多个互相独立的评估器，然后对其预测进行平均或多数表决原则来决定集成评估器的结果，装袋法的代表模型就是随机森林。

提升法：基评估器是相关的，是按顺序——构建的。其核心思想是结合 弱评估器的力量一次次对难以评估的样本进行预测，从而构成一个强评估器。提升法的代表模型有Adabosst和梯度提升树。

- 随机森林即是由多棵树组成的森林，即是由多个基评估器组成的集成评估器（集成算法），为Bagging集成算法。

### 1.2sklearn中的集成算法

- sklearn中的集成算法模块ensemble

![随机森林](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\集成算法(随机森林).png)

- 集成算法中，有一半以上都是树的集成模型，可以想见决策树在集成中必定是有很好的效果。在这堂课中，我们会以随机森林为例，慢慢为大家揭开集成算法的神秘面纱。

***

- sklearn的基本建模流程

1. 实例化，建立评估模型对象

2. 通过训练接口训练模型（clf.fit())

3. 通过模型接口提取需要的信息
4. 在下面这个流程中，随机森林的对应代码和决策树基本一致

~~~python
form sklearn.tree import RandomForestClassifier
rfc = RandomForestClassifier()      #实例化
rfc = rfc.fit(X_train,Y_train)      #用训练集数据训练模型
result = rfc.score(X_test,Y_test)   #导入测试集，从接口中调用需要信息
~~~

#### 2.1重要参数

- criterion 不纯度的衡量指标，有基尼系数和信息熵两种选择。
- max_depth 树的最大深度，超过最大深度的树枝都会被剪掉。
- min_samples_leaf 一个节点在分支后的每个子节点都必须包含至少min_samples_leaf 个训练样本。（改变训练样本，让他符合条件）。
- min_samples_split一个节点必须包含至少min_samples_split个训练样本，这个样本才允许被分支。（不改变训练样本个数，只是判断是否符合条件）。
- max_features 限制分支时考虑的特征个数，超过限制个数的特征会被舍弃，默认值为总特征个数开平方取整。
- min_impurity_decrease 限制信息增益的大小，信息增益小于设定数值的分支不会发生。

##### 2.1.2 n_estimators

这是森林中数目的数量，即基评估器的数量。这个参数对随机森林模型的精确性影响是单调的。n_estimators达到一定的程度之后，随即森林的精确性往往不再上升或者开始波动。并且，n_estimators越大，越需要的计算量和内存也越大，训练的时间也会越来越长。对于这个参数，我们是渴望在训练难度和模型效果之间取得平衡。n_estimators的默认值在现有版本的sklearn中是10，但在以后会改为100。

~~~python
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
import pandas as pd
wine = load_wine()
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)])
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,win.target,test_size = 0.3)
clf = DecisionTreeClassifier(random_state = 0)
rfc = RandomForestClassifier(random_state = 0)
clf = clf.fit(Xtrain,Ytrain)
rfc = rfc.fit(Xtrain,Ytrain)
score_c = clf.score(Xtest,Ytest)
score_c = rfc.score(Xtest,Ytest)
print("single Tree:{}".format(score_c))
print("Random Forest:{}".format(score_c))

~~~

~~~python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
wine = load_wine()
rfc = RandomForestClassifier(n_estimators = 25)
rfc_s = cross_val_score(rfc,wine.data,wine.target,cv = 10)#交叉验证
clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf,wine.data,wine.target,cv = 10)
plt.plot(range(1,11),rfc_s,label = "RandomForest")
plt.plot(range(1,11),clf_s,label="Decision Tree")
plt.legend()
plt.show()
~~~



![DecisionTreeClassfier与RandomTreeClassfier](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\DecisionTreeclassfier与RandomForest.png)

- 从上图可以看出，在拟合效果上，随机森林比决策树运行要好得多。

十组交叉验证：

~~~python
rfc_1=[]
clf_1=[]
for i in range(10):
    rfc = RandomForestClassifier(n_estimators = 25)
    rfc_s = cross_val_score(rfc,wine.data,wine.target,cv =10).mean()
    rfc_1.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s=cross_val_score(clf,wine.data,wine.target,cv= 10).mean()
    clf_1.append(clf_s)
plt.plot(range(1,11),rfc_1,label = "Random Forest")
plt.plot(range(1,11),clf_1,label = "Decision Tree")
plt.legend()
plt.show()
~~~

![十组数据对比](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\十组数据对比.png)

- 交叉验证：cross_val_score

将数据集分为10折，做一次交叉验证，实际上它是计算了十次，将每一折都当做一次测试集，其余九折当做训练集，这样循环十次。通过传入的模型，训练十次，最后将十次结果求平均值。

### 2.2重要属性和接口

- 随机森林当中，为了确保森林中的树每一棵都不尽相同，因此在选取训练集时，采用的是随机抽样（有放回）的方法，这样就会出现有些样本始终不能被选中的情况。这些数据叫做"袋外数据"。
- 随机森林的接口与决策树完全一致，因此依然由四个常用接口：apply，fit，predict，score。随机森林中的predict——proba接口，这个接口返回每个测试样本对应的被分到每一类标签的概率，标签有几个分类就返回几个概率。如果是二分类问题，则predict_proba返回的数值大于0.5的，被分为1，小于0.5的被分为0.传统的随机森林是利用袋装法中的规则，品军或者少数服从多数来决定集成结果，而sklearn中的随机森林，是平均每个样本对应的predict_preba返回的概率，得到一个平均概率，从而决定测试样本的分类。

~~~python
rfc = RandomForestClassifier(n_estimators = 25)
rfc = rfc.fit(Xtrain,Ytrain)
rfc.score(Xtest,Ytest)



rfc = feature_importances_
rfc.apply(Xtest)#apply返回每个测试样本所在的叶子节点的索引
rfc.predict(Xtest)#predict返回每个测试样本的分类/回归结果
rfc.predict_proba(Xtest)
~~~

![recdict_proba](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\这是predict_proda的运行结果.png)

### 2.3调参

***

在机器学习中，我们用来衡量模型在未知数据上的准确率的指标，叫做泛化误差。

![随机森林泛化误差]()![随即森林泛化误差](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\随即森林泛化误差.png)

![泛化误差之随机森林](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\随机森林之泛化误差.png)

![调参](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\调参.png)

### 5.实例：随机森林在乳腺癌数据上的调参

~~~python
from sklearn.datasets import load_breast_cancer#导入乳腺癌数据
from sklearn.ensemble import RandomForestClassifier#随机森林
from sklearn.model_selection import GridSearchCV#模型选择中的网格搜索
from sklearn.model_selection import cross_val_score#模型选择中的交叉验证
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data = load_breast_cancer()
rfc = RandomForestClassifier(n_estimators = 100,random_state = 90)
score_pre = cross_val_score(rfc,data.data,data.target,cv = 10).mean()
print(score_pre)
~~~

4.随机森林调整的第一步：无论如何先来调n_estimators

- 使用学习曲线来挑n_estimators，因为我想要观察n_setimators在何时开始变得平滑。

~~~python
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                  n_jobs = -1,
                                  random_state = 90)
    score = cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
plt.figure(figsize = [20,5])
plt.plot(range(1,201,10),scorel)
plt.show()
#list.index([object])
#返回这个object在列表list中的索引
~~~

![观察随机森林的学习](C:\Users\DELL\Desktop\work\picture\\观察随机森林的学习.png)

***

~~~python
#调整max_depth的参数
param_grid = {'max_depth':np.array(1,20,1)}
rfc = RandomorestClassifier(n_estimators = 39,
                            ,random_state = 90
                           )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
GS.veat_params_
GS.best_score_
~~~

### 1.1数据预处理与特征工程

1. 获取数据
2. 数据预处理：在数据中检测，纠正或删除损坏，不准确或者不适用于模型的记录过程。
3. 特征工程：将原始数据转换为更能代表预测模型的潜在问题的特征的过程，可以通过挑选最相关的特征，提取特征以及创造特征来实现，其中创造特征又经常以降维算法的方式实现。

### 1.2sklearn中的数据预处理

***

### 2.1数据无量纲化

在机器学习算法实践中，我们往往有着将不同规格的数据转换到同一规格，或不同分布的数据转换到某个特定分布的需求，这种需求统称为将数据“无量纲化”。譬如梯度和矩阵为核心的算法中，譬如逻辑回归，支持向量机，神经网络，无量纲化可以加快求解速度；而在距离类模型，譬如K近邻，K-Means聚类中，无量纲化可以帮我们提升模型精度，避免某一个取值范围特别大的特征对距离计算造成影响。（一个特例是决策树和树的集成算法们，对决策树我们不需要无量纲化，决策树可以把任意数据都处理得很好。）

数据的无量纲化可以是线性的，也可以是非线性的。线性的无量纲化包括**中心化**（Zero-centered或者Meansubtraction）处理和**缩放处理**（Scale）。中心化的本质是让所有记录减去一个固定值，即让数据样本数据平移到某个位置。缩放的本质是通过除以一个固定值，将数据固定在某个范围之中，取对数也算是一种缩放处理。

当数据（x）按照最小值中心化后，再按极差（最大值-最小值）缩放，数据移动了最小值个单位，并且会被收敛到[0,1]之间，这个过程叫做数据归一化。（Normalization,又称Min_Max Scaling)归一化之后的数据满足正态分布。在sklearn中，我们使用**preprocessing.MinMaxScaler**来实现这个功能。

![归一化](C:\Users\DELL\Desktop\work\picture\\归一化：符合正态分布.png)

~~~python
from sklearn.preprocessing import MinMaxScaler
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
import pandas as pd
pd.DataFrame(data)  #转化为数组形式。
#归一化：
scaler = MinMaxScaler()#实例化
scaler = scaler.fit(data)#fit，在这里是生成min(x)和max(x)
result = scaler.transform(data)#通过接口导出结果
result

result = scaler.fit_transform(data)#训练和导出结果一部达成

scaler.inverse_transform(result)#将归一化后的结果逆转，返回归一化之前的原始数据
#使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中
data = [[-1,2],[-0.5,6],[0,10],[1,18]]
scaler = MinMaxScaler(feature_range = [5,10])#依然实例化
result = scaler.fit_transform(data)#fit_rransform一部导出结果
result
#当x中的特征熟练非常多的时候，fit会报错并表示数据量太大了我计算不了，此时用partial_fit作为训练接口
#scaler = scaler.partial_fit(data)
~~~

![归一化](C:\Users\DELL\Desktop\work\picture\\归一化运行结果.png)

- preprocessing.SrandarScaler

当数据按均值中心化后，再按照标准差进行缩放，数据会服从均值为0，方差为1的正态分布。而这个过程叫做**数据标准化**。又称(score normalization)

![数据标准化](C:\Users\DELL\Desktop\work\picture\数据标准化.png)

~~~python
from sklearn.preprocessing import StandardScaler
data = [[-1,2],[-0,5,6],[0,10],[1,18]]
scaler = StandardScaler()#实例化
scaler.fit(data)#fit,本质是生成均值和方差
scaler.mean_#查看均值的属性mean_
scaler.var_#查看方差的属性var_
x_std = scaler.transform(data)#通过接口导出结果
x_std.mean()#导出的结果是一个数组，用mean()查看均值,结果为0
x_std.std()#用std()查看方差，结果为1
scaler.fit_transform(data)#使用fit_transform(data)#一步达成结果
scaler.inverse_transform(x_std)#使用inverse_transform逆转标准化
~~~

![无量纲化](C:\Users\DELL\Desktop\work\picture\无量纲化.png)

![img](C:/Users/DELL/AppData/Roaming/Tencent/Users/731453367/QQ/WinTemp/RichOle/`DC$N{30I3$THT2RX_@_W7I.png)

## 回归树案例：用回归拟合正弦曲线

~~~python
import numpy as np
from sklearn.tree  import DecisionTreeRegressor
import matplotlib.pyplot as plt
#创建一条含有噪声的正弦曲线
rng = np.random.RandomState(1)#生成随机种子
X = np.sort(5*rng.rand(80,1),axis=0)#生成0-5的x
y = np.sin(X).ravel()#ravel降维函数
y[::5]+=3*(0.5 - rng.rand(16))#进行噪声处理

~~~

接下来进行实例化

~~~python
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)#将上述所设立的x，y进行拟合，得到实例化模型的具体实例化
regr_2.fit(X, y)
~~~

~~~python
#导入测试集
X_test = np.arange(0.0,5.0,0.01)[:,np.newaxis]#后边的切片是增维函数
y_1 = regr_1.predict(X_test)#导入测试集之后得到每个测试样本点的回归或者分类的结果（得到的y）
y_2 = regr_2.predict(X_test)
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/PC~J71OZ%7B%5BU8NB%7DR_ITDL%5D6.png)

~~~python
plt.scatter(X, y, s=20, edgecolor="black",c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
~~~

![](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/20220707105003.png)

~~~python
#由图可知，当最大深度为2时较为合理，当为5时明显的是过拟合
~~~

### 实例：泰坦尼克号幸存者的预测

~~~python
import pandas as pd
from sklraen ,tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
form sklearn.model_selection import Grid SearchCV
~~~

- 具体见jupyter

## 缺失值的处理：

~~~python
data.loc[:,"Age"].values.reshape(-1,1).shape #reshape(-1,1)#将数据进行二维化
结果为：(891,1)
~~~

利用机器学习进行缺失值处理

~~~python
from sklearn,impute import SimpleImputer
imp_mean = SimpleImpiputer()#实例化，默认均值填补
imp_median = SimpleImputer(strategy = "median")#用中位数填补
imp_mode = SimpleImputer(strategy = "most_frequent")#用众数填补,数值型和字符型特征都可以
imp_0 = SimpleImputer(strategy = "constant",fill_value = 0)#用0填补，数值型和字符型都可
~~~

~~~python
imp_mean = imp_mean.fit_transform(Age)#fit_transform一步完成调取结果
imp_median = imp_median.fit_transform(Age)
imp_0 = imp_0.fit_transform(Age)
data.loc[:,"Age"] = imp_median
data.info()
~~~

缺失值直接填补：

~~~python
data_.loc[:,"Age"] = ata_.loc[:,"Age"].fillna(data_.loc[:,"Age"].median())
#.fillna在DataFrame里面直接进行填补
data_.dropna(axis= 0,inplace = True)#inplace是将处理之后的数据是否替代原数据
~~~

## 2.3处理分类型特征：编码与哑变量

- 将文字型数据转化成数字型数据:

- preprocessing.LabelEncoder:标签专用，能够将分类转换为分类数值

~~~python
from sklearn.preprocessing import LabelEncoder

y = data.iloc[:,-1]   #要输入的是标签，不是特征矩阵,所以允许一维
le = LabelEncoder()  #实例化
le = le.fit(y)  #导入数据
label = le.transform(y)#transfrom接口调取结果
le.classes_  #属性.classes查看标签中究竟有多少类别
label  #查看获取的结果label

le.fit_transfrom(y)#也可以直接fit——transfrom一步到位
le.inverse_transform(label)#使用inverse_transfrom可以逆转
~~~

- 也可使OrdinalEncoder

~~~ python
from sklearn.preprocessing import OrdinalEncoder
#接口categores_对应LabelEncoder的接口classes_,一模一样的功能
data_ = data.copy()
data_.head()
OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_
data.iloc[:,1:-1] = RrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
data_.head()

~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/ZI2MR6PO]35~QP$%@VT1T37.png)

![img](C:/Users/DELL/AppData/Roaming/Tencent/Users/731453367/QQ/WinTemp/RichOle/PF](1ZL2J}_16)4IYY6~N3Q.png)



![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/ZR9V@65WGH5%7DJHOHKOG%7D%7D%5B7.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/M`LWGNF_SJA]EC{%Z7{JBKL.png)

**哑变量**

![image-20220710090344065](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710090344065.png)

数据变换之前：

![image-20220710090611179](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710090611179.png)

~~~python
from sklrearn.preprocessing import OneHotEncoder
X = data,iloc[:,1:-1]
enc = OneHotEncoder(categories = 'auto').fit(x)
result = enc.transform(X).toarray()#把结果转化成二维数组
result
~~~

![image-20220710091058380](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710091058380.png)

解释：因为性别有两个哑变量Embarked有三个哑变量

还原之后：

![image-20220710091704251](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710091704251.png)



~~~python
重要属性：enc.get_feature_names()#他会告诉你这一列是哪一类
~~~

![image-20220710091853994](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710091853994.png)

可以将得到的结果拼接在数据后边：

~~~python
#axis=1，表示跨行进行合并，也就是将量表左右相连，axis=1表示列，列拼接为增加列数，所以是左后相连，axis=0时为行，行拼接为行数相加则量表上下相连
newdata = pd.concat([data,DataFrame(result)],axis = 1)
newdata.head()
~~~

![image-20220710093314113](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710093314113.png)

~~~python
newdata.drop(["Sex","Embarked"],axis =1,inplace = Ture)#去掉这两列并且替代原数据
new.data.columns = ["Age","Survivd","Female","Male","Enbarked_C","Enbarked_Q","Embarked_S"]
newdata.head()
~~~

![image-20220710093641237](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710093641237.png)

![image-20220710093845831](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710093845831.png)

![image-20220710094043478](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710094043478.png)

## 2.4处理连续型特征：二值化与分段

![image-20220710094446922](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710094446922.png)

~~~python
#将年龄二值化
data_2 = data.copy()
from sklearn.preprocessing import Binarizer
X = data_2.iloc[:,0].values.reshape(-1,1)#类为特征专用，所以不能使用一维数组，reshape（-1，1）转化为二维数组，且只有一列
~~~

~~~python
transformer = Binaeizer(threshold = 38).fit_transform(X)
transformer
~~~

![image-20220710095757006](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710095757006.png)

![image-20220710100019530](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710100019530.png)

### 分箱

~~~python
from sklearn.preprocessing import KBinsDiscretizer
X = data,iloc[:,0].values.reshape(-1,1)#类为特征专用不能够导入一维
~~~



~~~python
est = KBinsDiscretizer(n_bins = 3,encode = 'ordinal',stratrgy='uniform')
est.fit_transform(X)
~~~

![image-20220710101235240](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710101235240.png)

~~~python
#查看转换后分的箱：变成了一列中的三箱
set(est.fit_transfrom(X).ravel())#ravel降维操作
~~~

![image-20220710101549634](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710101549634.png)

~~~python
est = KBinsDiscretizer(n_bins = 3,encode='onehot',strategy = 'uninform')
#查看转换后的分的箱：变成了哑变量
est.fit_transform(X).toarray()#toarray将结果的稀疏矩阵转化成二维数组
~~~

![image-20220710101625779](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710101625779.png)

## 3. 特征选择feature——selection

跟数据提供者开会

特征选项而第一步，根据我们的目标，用业务常识来选择特征。

![image-20220710145745725](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710145745725.png)

![image-20220710145944133](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710145944133.png)

![image-20220710152013131](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710152013131.png)

~~~python
import pandas as pd
data =pd.read_csv(r"E:\Jupyter and pycharm\pythonProject1\jupyter2\digit recognizor.csv")
X = data.iloc[:,1:]
y = data.iloc[:,0]
'''
维度是指特征的数量

'''
~~~

~~~python
X.shape
data.head()
~~~

![image-20220710155737239](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710155737239.png)

![image-20220710155756712](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710155756712.png)

~~~python
from sklearn.fearuew_selection import VaianceThreshold
selector = VarianceThreshold() #实例化，不填参数默认方差为0
X_var0 = selector.fit_transfrom(X)#获取删除不合格特征之后的新特征矩阵
~~~

~~~python
#也可以直接写成X = VairanceThreshold().fit_transform(X)
X_var0.shape
~~~

![image-20220710153137814](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710153137814.png)

~~~python
import numpy as np
X.var()#查看每一列方差
np.median(X.var()) #查看方差中位数
~~~

~~~python
X_fsvar = VarianceThreshold(np.median(X.var())).fit_transform(X)
X_fsvar.shape
~~~

![image-20220710153923354](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710153923354.png)



![image-20220710154018740](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710154018740.png)

~~~python
#若特征是伯努利随机变量，假设p=0.8，即而分类特征中某种分类占到80%以上的时候删除特征
X_bvar = VarianceThreshold(.8*(1-.8)).fit_transform(X)
X_bvar.shape
~~~

![image-20220710154530511](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710154530511.png)

![image-20220710154745357](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710154745357.png)

~~~python
#KNN vs 随机森林在不同方差过滤效果下的对比
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
import numpy as np
X = data.iloc[:,1:]
y = data.iloc[:,0]
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
~~~

~~~python
#======【TIME WARNING：35mins +】======#
cross_val_score(KNN(),X,y,cv=5).mean()
#python中的魔法命令，可以直接使用%%timeit来计算运行这个cell中的代码所需的时间
#为了计算所需的时间，需要将这个cell中的代码运行很多次（通常是7次）后求平均值，因此运行%%timeit的时间会
远远超过cell中的代码单独运行的时间
#======【TIME WARNING：4 hours】======#
%%timeit
cross_val_score(KNN(),X,y,cv=5).mean()
~~~

![image-20220710161459451](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710161459451.png)

3.方差过滤之后

~~~python
#======【TIME WARNING：20 mins+】======#
cross_val_score(KNN(),X_fsvar,y,cv=5).mean()
#======【TIME WARNING：2 hours】======#
%%timeit
cross_val_score(KNN(),X,y,cv=5).mean()
~~~

![image-20220710161919402](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710161919402.png)

![image-20220710162013765](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710162013765.png)

**随机森林方差过滤前**

![image-20220710162125489](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710162125489.png)

**随机森林方差过滤后**

![image-20220710162223591](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710162223591.png)

![image-20220710163348854](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710163348854.png)

![image-20220710163547213](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710163547213.png)

3.1.2.1卡方过滤(非负值)

卡方过滤是专门针对离散型标签（即分类问题）的相关性过滤。卡方检验类**feature_selection.chi2**计算每个非负

特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。再结合**feature_selection.SelectKBest**

这个可以输入”评分标准“来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签，与我们分类目

的无关的特征。

另外，如果卡方检验检测到某个特征中所有的值都相同，会提示我们使用方差先进行方差过滤。并且，刚才我们已

经验证过，当我们使用方差过滤筛选掉一半的特征后，模型的表现时提升的。因此在这里，我们使用threshold=中位数时完成的方差过滤的数据来做卡方检验（如果方差过滤后模型的表现反而降低了，那我们就不会使用方差过滤后的数据，而是使用原数据）：

~~~python
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest#挑选特征
from sklearn.feature_selection import chi2  #卡方


#假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
X_fschi.shape
~~~

![image-20220710172107022](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710172107022.png)

~~~python
cross_val_score(RFC(n_estimators=10,random_state=0),X_fschi,y,cv=5).mean()
结果为：0.9333
~~~

![image-20220710172259244](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710172259244.png)

![image-20220710172357125](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710172357125.png)

~~~python
#======【TIME WARNING: 5 mins】======#
%matplotlib inline
import matplotlib.pyplot as plt
score = []
for i in range(390,200,-10):
    X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
    once = cross_val_score(RFC(n_estimators=10,random_state=0),X_fschi,y,cv=5).mean()
    score.append(once)
plt.plot(range(350,200,-10),score)
plt.show()
~~~

![image-20220710172947218](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710172947218.png)

![image-20220710173315672](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710173315672.png)

### 也可以叫卡方检验筛掉一些特征。

~~~python
chivalue, pvalues_chi = chi2(X_fsvar,y)
chivalue#卡方值
pvalues_chi#P值
~~~

![image-20220710173703608](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710173703608.png)

![image-20220710173728775](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710173728775.png)

- 由p值可知，所有的之都小于0.05，所以所有特征都相关

~~~python
#k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()#计算大于设定值的数量
~~~

## F检验过滤

F检验，又称ANOVA，方差齐性检验，是用来捕捉每个特征与标签之间的线性关系的过滤方法。它即可以做回归也

可以做分类，因此包含**feature_selection.f_classif**（F检验分类）和**feature_selection.f_regression**（F检验回

归）两个类。其中F检验分类用于标签是离散型变量的数据，而F检验回归用于标签是连续型变量的数据。和卡方检验一样，这两个类需要和类**SelectKBest**连用，并且我们也可以直接通过输出的统计量来判断我们到底要

设置一个什么样的K。需要注意的是，F检验在数据服从正态分布时效果会非常稳定，**因此如果使用F检验过滤，我们会先将数据转换成服从正态分布的方式。**F检验的本质是寻找两组数据之间的线性关系，其原假设是”数据不存在显著的线性关系“。它返回F值和p值两个统计量。

~~~python
from sklearn.feature_selection import f_classif
F, pvalues_f = f_classif(X_fsvar,y) #返回两个值
F  #和卡方值类似，不知多少号
pvalues_f#类似于p值
~~~

![image-20220710180445453](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710180445453.png)

![image-20220710180530102](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710180530102.png)

~~~python
k = F.shape[0] - (pvalues_f > 0.05).sum()
得到 k = 392
#X_fsF = SelectKBest(f_classif, k=填写具体的k).fit_transform(X_fsvar, y)
#cross_val_score(RFC(n_estimators=10,random_state=0),X_fsF,y,cv=5).mean()#交叉验证得到结果
~~~

3.1.2.4互信息法

![image-20220710181050235](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710181050235.png)

~~~python
from sklearn.feature_selection import mutual_info_classif as MIC
result = MIC(X_fsvar,y) 
~~~

![image-20220710203920630](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710203920630.png)

~~~python
k = result.shape[0] - sum(result <= 0)
#X_fsmic = SelectKBest(MIC, k=填写具体的k).fit_transform(X_fsvar, y)
#cross_val_score(RFC(n_estimators=10,random_state=0),X_fsmic,y,cv=5).mean()
~~~

**3.1.3** **过滤法总结**

到这里我们学习了常用的基于过滤法的特征选择，包括方差过滤，基于卡方，F检验和互信息的相关性过滤，讲解

了各个过滤的原理和面临的问题，以及怎样调这些过滤类的超参数。通常来说，我会建议，先使用方差过滤，然后

使用互信息法来捕捉相关性，不过了解各种各样的过滤方式也是必要的。所有信息被总结在下表，大家自取：

![image-20220710204119659](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220710204119659.png)·
