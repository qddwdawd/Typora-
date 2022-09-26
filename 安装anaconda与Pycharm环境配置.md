## Anaconda 安装

在浏览器中搜索Anaconda进行安装，跟随视频进行勾选。

而后进行Pytorch的安装，在Anaconda中有，进行进一步的环境设置。

![image-20220824202711752](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824202711752.png)

- 选择添加

![image-20220824202820062](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824202820062.png)

- 选择Conda环境，而后选择现有环境，在解释器下选择Anaconda安装目录下的Python.exe文件。

![image-20220824202916135](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824202916135.png)

#### 在进行完环境配置之后，进行导包

- 比如说一些matplotlib，numpy，pandas等，可以用cmd进行直接下载：pip install matplotlib，conda install matplotlib(注意环境变量)
- 也可以在Anaconda自带的Prompt中输入以上操作
- 仍可以在设置中的Conda环境配置中进行搜索安装，比较快捷如下：

![image-20220824203910176](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824203910176.png)

#### 配置Designer环境：

- 在下载完Anaconda之后就会有qtdesigner，找到目录之后进行操作

我的目录：

"E:\Anaconda\Library\bin\designer.exe"

"E:\Anaconda\pkgs\qt-5.9.7-vc14h73c81de_0\Library\bin\designer.exe"

![image-20220824220934549](C:/Users/DELL/AppData/Roaming/Typora/typora-user-images/image-20220824220934549.png)

![](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824220946502.png)

##### 配置pyuic

这个也是与上面相同，在下载Anaconda的时候自带

![image-20220824221959776](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824221959776.png)

我的地址：

- E:\Anaconda\python.exe
- -m PyQt5.uic.pyuic $FileName$ -o $FileNameWithoutExtension$.py
- $FileDir$

![image-20220824222100830](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220824222100830.png)

### 如果需要在cmd中使用还需要配置环境变量

