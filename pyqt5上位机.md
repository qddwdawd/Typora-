# Qt上位机

主窗口类型

有3种窗口

QMainWindow

QWidget

QDialog

QMainWindow :可以包含菜单栏，工具栏，状态栏和标题栏，是最常见的窗口形式。

QDialog：是对话窗口的基类。没有惨淡蓝，工具栏，状态栏。

QWidget：不确定窗口的用途，就使用QWidget。

~~~python
import sys  #用里边一个获取参数的api
from PyQt5.QtWidgets import QMainWindow,QApplication #用来创建窗口，以及创建任何应用程序都比与要用QApplication
from PyQt5.QtGui import QIcon

class FirstMainWin(QMainWindow):#从QMainWindow主窗口继承
    def __init__(self,parent = None):
        super(FirstMainWin,self).__init__(parent)
        #设置主窗口的标题
        self.setWindowTitle('第一个主窗口应用')
        #设置主窗口的尺寸
        self.resize(400,300)
        self.status = self.statusBar()
        self.status.showMessage('只存在5秒的消息',5000)

if __name__== '__main__':
    app = QApplication(sys.argv)
    main = FirstMainWin()
    main.show()
    sys.exit(app.exec_())
~~~

### 如何关闭窗口

~~~python
import sys
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QApplication, QWidget,QPushButton


class QuitApplication(QMainWindow):
    def __init__(self):
        super(QuitApplication, self).__init__()
        self.resize(300, 120)
        self.setWindowTitle('退出应用程序')
        
        # 添加Button
        self.button1 = QPushButton('退出应用程序')
        self.button1.clicked.connect(self.onClick_Button)
        
        layout = QHBoxLayout()#水平布局
        layout.addWidget(self.button1)
        
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)
        
        # 按钮单击事件的方法

    def onClick_Button(self):
        sender = self.sender()  # sender获得发送消息的对象
        print(sender.text() + '按钮被按下')  # 输出Button的文本
        app = QApplication.instance()  # 获取app单例对象，也就是当前对象
        app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QuitApplication()
    main.show()
    sys.exit(app.exec_())

    
~~~

一个简单的控件连接案例

~~~python
import sys
from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QApplication, QWidget,QPushButton
from Setting import Ui_Setting
from PyQt5 import QtCore, QtGui


class QuitApplication(QMainWindow):
    def __init__(self):
        super(QuitApplication, self).__init__()
        self.resize(300, 120)
        self.setWindowTitle('退出应用程序')
        # 添加Button
        self.button1 = QPushButton()
        self.button1.clicked.connect(self.SettingPBt_ClickEvent)
        layout = QHBoxLayout()
        layout.addWidget(self.button1)
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)
        # 按钮单击事件的方法
        self.button2 = QPushButton('shezhi')  # self.LoginWgt， # 添加Button
        self.button2.clicked.connect(self.SettingPBt_ClickEvent)
        layout = QHBoxLayout()
        layout.addWidget(self.button2)
        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.SettingMW = QMainWindow()
        self.ui_Setting = Ui_Setting()
        self.ui_Setting.setupUi(self.SettingMW)
        self.button1.setMinimumSize(QtCore.QSize(40, 40))  # 设置按钮大小
        self.button1.setMaximumSize(QtCore.QSize(40, 40))

        self.button1.setStyleSheet("QPushButton{"
                                      "}\n"
                                      "/*按钮普通态*/\n"
                                      "QPushButton\n"
                                      "{\n"
                                      "    /*字体为微软雅黑*/\n"
                                      "    font-family:Microsoft Yahei;\n"
                                      "    /*字体大小为20点*/\n"
                                      "    font-size:12pt;\n"
                                      "    /*字体颜色为白色*/    \n"
                                      "    color:white;\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(67,67,67);\n"
                                      "    /*边框圆角半径为8像素*/ \n"
                                      "    border-radius:7px;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:hover\n"
                                      "{\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(90, 90, 90);\n"
                                      "}\n"
                                      "\n"
                                      "/*按钮按下态*/\n"
                                      "QPushButton:pressed\n"
                                      "{\n"
                                      "    /*背景颜色*/  \n"
                                      "    background-color:rgb(14 , 135 , 228);\n"
                                      "    /*左内边距为3像素，让按下时字向右移动3像素*/  \n"
                                      "    padding-left:3px;\n"
                                      "    /*上内边距为3像素，让按下时字向下移动3像素*/  \n"
                                      "    padding-top:3px;\n"
                                      "}")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("png/3E9]I~U815DD3{7OQXM]8JF.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.button1.setIcon(icon)
        self.button1.setIconSize(QtCore.QSize(40, 40))

    def SettingPBt_ClickEvent(self):
        if self.SettingMW.isVisible():#如果设置窗口可见，第一次应该返回false
            return self.SettingMW.close() and self.SettingMW.show()
        else:
            return self.SettingMW.show()
    def onClick_Button(self):
        sender = self.sender()  # sender获得发送消息的对象
        print(sender.text() + '按钮被按下')  # 输出Button的文本
        app = QApplication.instance()  # 获取app单例对象，也就是当前对象
        app.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QuitApplication()
    main.show()
    sys.exit(app.exec_())

~~~

