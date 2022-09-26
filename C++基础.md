# C++基础

![image-20220729164608411](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220729164608411.png)



~~~c++
#include<iostream>
using namespace std;

int main()
{
	cout << "hello world" << endl;
	system("pause");
	return 0;
}
~~~

![image-20220729201930136](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220729201930136.png)

![image-20220729201942094](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220729201942094.png)

~~~python
#include<iostream>
using namespace std;
// 1.单行注释
//2.多行注释

/*  
都可以进行多行注释
main 是一个程序的入口
每个程序都必须有这么一个函数
有且仅有一个
*/
int main()
{   
	//11行代码的含义就是在屏幕中输出Hello world
	cout << "hello world" << endl;
	system("pause");
	return 0;
}
~~~

##### 1.3 变量

![image-20220729203833706](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220729203833706.png)

~~~C++
#include<iostream>
using namespace std;

int main()
{   
	//变量创建的语法：数据类型 变量名 = 变量初始值
	int a = 10;

	cout << "a = "<< a << endl;
	system("pause");
	return 0;
~~~

# 类

![image-20220913120156674](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220913120156674.png)



