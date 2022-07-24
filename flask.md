# PyQt5



#### 开发第一个基于PyQt5的桌面应用

#### 必须使用两个类：QApplication,和QWidget。都在PyQt5.QtWidgets

~~~python
import sys
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':
    # 创建QApplication类的实例
    app = QApplication(sys.argv)  # 用来获得Linux参数
    # 创建一个窗口
    w = QWidget()
    # 设置窗口的尺寸
    w.resize(400, 200)
    # 移动窗口
    w.move(300, 300)

    # 设置窗口的标题
    w.setWindowTitle('第一个基于PyQt5的桌面应用')
    # 显示窗口
    w.show()

    # 进入程序的主循环，并通过exit函数确保主循环安全结束
    sys.exit(app.exec_())
~~~



## Qt_Designer

![image-20220715195940017](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715195940017.png)

垂直布局：

Vertical Layout

![image-20220715201108548](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220715201108548.png)







