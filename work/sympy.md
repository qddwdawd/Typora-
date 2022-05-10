# sympy

导包

~~~python
import sympy as sp
~~~

求不定积分：

![不定积分](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\不定积分.png)

1. 表示自变量及积分对象

~~~python
x = sp.symbols('x')   #设置自变量x
formula = sp.Intgegral(sp.cos(x)*sp.exp(x),x)#sp.exp表示e后边的(x)表示求e的x次方
~~~

2. 求积分：

~~~python
formula.doit()
~~~

