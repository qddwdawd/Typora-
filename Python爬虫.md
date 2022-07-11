# python爬虫

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/IQQQG]7WAW`L7%1Q7C%E@BD.png)

​    ![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/S16BVI(1BTG)655F~HPT~PS.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/YCT{2{7_U}BVN}I_[W_7%JV.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/T064KRRU7DCD_B93OT%5DVNA8.png)



![](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/T064KRRU7DCD_B93OT%5DVNA8.png)



![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/5N_A%60B5%5BSR84$J_IT6UC%60F5.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/B(DX@SM)W3Q653BAHU02~2L.png)



~~~python
import bs4 #网页解析，获取数据
import re  #正则表达式，进行文字匹配
import urllib.request,urllib.error   #制定URL，获取网页数据
import xlwt #进行excel操作
import sqlite3 #进行SQLite 数据库操作
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/}X8H%YYW2QGKUO~LP$[%@GH.png)

~~~python
from bs4   import BeautifulSoup
import re
import urllib.request,urllib.error
import xlwt
import sqlite3
def main():
    baseurl = "https://movie.douban.com/top250?start="
    #1.爬取网页
    datalist = getData(baseurl)
    savepath = ".\\豆瓣电影Top.250"
    #2.逐一解析数据
    #3.保存数据
#爬取网页
def getData(baseyul):
    datalist=[]
    #2.逐一解析数据
    return datalist
def saveData(savepath):
    print(save)
~~~

- 用get方法请求申请网页信息

~~~python
import urllib.request
#获取一个get请求
response = urllib.request.urlopen("http://www.baidu.com")
print(response.read().decode('utf-8'))#解码，防止乱码
#此代码能够成功的获取数据
~~~

- 用post方法请求申请网页信息

~~~python
import urllib.parse
data = bytes(urllib.parse.urlencode({"hello":"world"}),encoding = "utf-8")
response = urllib.request.urlopen("http://httpbin.org/post",data=data)
print(response.read().decode("utf-8"))
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/F%5D@I8MCNP%5D0~ST%60IY%60%5BRT8M.png)

这是一个测试网站，请求和响应的服务，当你向它发送一些请求的时候，他会告诉你他会给你一些什么相应，通过这样的方式测试你的请求是否得到了应有的响应。

当使用post方法进行网站访问时，

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/%5DQN%7DPEQCW@2W_%60U_F~G5RSX.png)

它会出现以上的操作，当没有参数的时候（No parameters），他会给你一个相应，响应的状态码是code：200，具体的信息点击Try it out

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/2QR@%5BTVZUC0%5B65JBYMDFP2G.png)

点击之后，他会弹出一个框Execute，叫做执行，每当你向它发送一次消息，他就会给你一个相应

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/U_Q%5DQIM_1$J%5BOI7K2ZF04OO.png)

当你以这种方式请求他，他会给你反应信息：

![image-20220707183327150](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/image-20220707183327150.png)

~~~python
import urllib.parse#这是解析器，帮你把键值对给你解析一下，按照一定格式来解析
data = bytes(urllib.parse.urlencode({"hello":"world"}),encoding = "utf-8")#bytes将你的信息转化成二进制数据包，放一些可以是编码解码，登录信息等等，键值对以字典的形式，最后说明你的二进制数组是以什么方式封装的，把前边的字典的内容进行封装成bytes数组形成一个data
response = urllib.request.urlopen("http://httpbin.org/post",data=data)#这个网址中末尾的/post就是说明post的方式来请求，但是不能够直接利用post来访问，你必须要给他一些post的表单信息，通过表单的封装你才可以顺利访问
print(response.read().decode("utf-8"))#解码，防止乱码
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/M~WIN%7BS%60%60LSR$O82919PL1J.png)

我们可以看到，返回的内容和测试的内容基本一致，并且它会将它收到的内容，即你封装的信息以form表单的形式展示出来，同时还能收到headers中的浏览信息，包括其他请求的一些内容。

所以这种方法用在我们模拟用户真实登录的时候

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/MUEHMV12HDWK5P3%E]6R~T7.png)

看标红所显示，如果是真正的网站，它会显示代理机构是你的计算机系统信息，而网络爬虫则会真正的反映出你的爬虫信息，所以说如果你访问的网站具有保护系统，则不能够进行访问。

### 超时

当系统发现你是一个爬虫或者你的网速比较慢，则会超时

~~~python
#超时处理：
try:#监测，异常检测
  response = urllib.request.urlopen("http://httpbin.org/get",timeout = 1)
  print(response.read().decode('utf-8'))
except urllib.error.URLError as e:
    print("time out !")

~~~

### 将网页内容进行解析

~~~python
response = urllib.request.urlopen("http://baidu.com")
print(response.getheaders())
#在访问过程中返回你的请求信息中的头部信息，状态码
~~~

## 当不想让网站知道你是爬虫时，要伪装成一个浏览器：

~~~python
url = "http://douban.com"
headers = {
"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36 Edg/103.0.1264.44"
}
data = bytes(urllib.parse.urlencode({"name":'eric'}),encoding="utf-8")
req = urllib.request.Request(url=url,data=data,headers = headers,method = "POST")
#这仅仅是构建了一个请求对象，还需要发出请求
response = urllib.request.urlopen(req)#req是封装的对象，包含了封装的头部信息，传递方式，封装以及url
print(response.read().decode("utf-8"))
~~~

说明：利用urllib.request.Request函数进行请求对象的封装，包括url网址，data利用post访问的封装，headers相应头部信息的封装，headers为字典类型，data仍然为二进制数组

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/PR%FX%MM_36{EQ1D%1LJ8NB.png)

## 进行函数的定义，因为我们要进行数据处理，且有好多页的数据，如果不定义将会很麻烦，所以：

~~~python
#得到指定一个URL的网页内容
def askURL(url):
    head = {   #模拟浏览器头部信息，向豆瓣服务器发送消息
        "user-agent": "Mozilla / 5.0(Windows NT 10.0;Win64;x64) AppleWebKit / 537.36(KHTML, likeGecko) Chrome / 103.0.5060.66Safari / 537.36Edg / 103.0.1264.44"
    }#用户代理，表示告诉豆瓣服务器，我们是什么类型的机器，浏览器不是爬虫，本质上是告诉浏览器我们需要什么水平的文件

    request = urllib.request.Request(url,headers = head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        print(html)
    except urllib.error.URLError as e: #遇到404，500等错误
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html
askURL("https://movie.douban.com/top250?start=0")

def getData(baseurl):
    for i in range(0,10):   #调用获取页面信息的函数10次
        url = baseurl +str(i*25)
        html = askURL(url)   #保存获取到的网页源码

getData("https://movie.douban.com/top250?start=")
~~~

## 关于解析的知识

```
BeautifulSoup4将复杂的HTML文档转化成一个复杂的树形结构，每一个节点都是Python对象，所有对象可以归纳为4种：
'''
-Tag
-NavigableString
-BeautifulSoup
-Comment
'''
```

~~~python
from bs4 import BeautifulSoup

file = open("./baidu.html,"rb")
html = file.read()
bs = BeautifulSoup(html,"html.parser")#用解析器解析
print(type(bs.head))
结果为<class 'bs4.element.Tag'>#很明显结果显示为标签
            #1.Tag,标签及其内容：拿到他所找到的第一个内容
print(bs.title.string)#单纯打标签里边的内容
print(type(bs.title.string))
结果为<class 'bs4.element.NavigableString'>#标签里的内容
            #NavigableString，标签里的内容（字符串）

~~~

~~~python
print(bs.a.attrs)#得到一组键值对，字典的形式，得到一个标签里的所有属性
~~~

~~~python
print(type(bs))
#3.BeautifulSoup   表示整个文档
print(bs.name)
~~~

~~~python
print(bs.a.string)
print(type(bs.a.string))
#4. Comment 是一个特殊的NavigableString，标签里的内容（字符串），输出的内容不包含特殊符号
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/_N%5B%7DN$ZDG~G730AS%5B9@IHAO.png)

### 文档的遍历

~~~python
print(bs.head.contents)
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/B5MT4PXDTG%7D~EO8%7B1F3%7D4DJ.png)

文档的遍历是指用contents的操作将你所需要内容以及标签以列表的形式返回。

~~~python
#用列表索引来获取它的元素
print(bs.head.contents[1])
~~~

#### find——all（）

~~~python
#字符串过滤：会查找与字符串完全匹配的内容
t_list = bs.find.all("a")
~~~

##### 正则表达式搜索：使用search（）方法来匹配内容

~~~python
import re
t_list = bs.find_all(re.compile("a"))#所有包含a的字样的内容都会被显示出来，并且返回列表的形式为以一个标签为一项内容（标签及其子内容）
print(t_list)
~~~

#方法  ： 传入一个函数（方法），根据函数的要求来搜索

~~~python
def name_is_exists(tag):
    return tag.has_arrt("name")
t_list = bs.find_all(name_is_exists)
print(t_list)
#返回标签中有“name”的所有内容
~~~

#2.kwargs    参数

~~~python
t_list = bs.find_all(id="head") #给一些参数会返回他所有包括的内容
t_list= bs.find_all(class_=True)
~~~

#3.text参数

~~~python
t_list = bs.find_all(text = "hao123")
t_list = bs.find_all(text = re.compile("/d"))#应用正则表达式来查找包含特定文本的内容（标签里的字符串）
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/0L%7DPIY%5BLA9S~%7BO1R_2F@6JL.png)

#4.limit参数

~~~python
t_list = bs.find_all("a",limit = 3)
for item in t_list:
    print(item)
~~~

#css选择器

~~~python
print(bs.select("title"))
~~~

![img](../../AppData/Roaming/Tencent/Users/731453367/QQ/WinTemp/RichOle/2SLP28`J%9_23_}E]$%{P`7.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/6G%}ZO}FZS7M3H35`[{3S35.png)

mnav前边的.表示类名

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/%7DQCR4L_3%7D%5D%5BC%7B@BSTXFA3$R.png)

#表示按ID来查找

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/H36%7BBI_S%60~%601%5D%5DW93L%60W8U4.png)![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/_2L~YLOP23@9Z25LMCKH9~L.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/IEAI%5DO%5D4Z$99~QY52~PZY0F.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/B1B0_%60ULC4MLVQA%5BGM%7DU%5B$N.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/SM@@S@291DQ@ZG}XM%N{I`9.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/PZQUIYHB%5D5~K06TV@KY5KPM.png)

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/%7D6HG($MI(HZPAK%7D)N4FKPMY.png)

~~~python
import re
#创建模式对象
pat = re.compile("AA")#此处的AA，是正则表达式，用来去验证其他的字符串#compile为编译
m = pat.search("CBA")  #search字符串被校验的内容
print(m)
#结果为None
~~~

~~~python
import re
#创建模式对象
pat = re.compile("AA")#此处的AA，是正则表达式，用来去验证其他的字符串#compile为编译
m = pat.search("ABCAA")  #search字符串被校验的内容
print(m)
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/NOVFJ4OMOZWS%5B2V%60J3LB%7DDH.png)

且当有多个AA出现时，search方法只能找到第一个符合要求的字符串位置

~~~python
#爬取网页
def getData(beaseurl):
    datalist = []
    for i in range(0,10):
        url = baseurl +str(i*25)
        html = askURL("https://movie.douban.com/top250?start=0")
        soup = BeautifulSou(html,"html.parser")
        for item in soup.find_all('div',class ="item"):
            print(item)
 def askURL(url):
    head = {   #模拟浏览器头部信息，向豆瓣服务器发送消息
        "user-agent": "Mozilla / 5.0(Windows NT 10.0;Win64;x64) AppleWebKit / 537.36(KHTML, likeGecko) Chrome / 103.0.5060.66Safari / 537.36Edg / 103.0.1264.44"
    }#用户代理，表示告诉豆瓣服务器，我们是什么类型的机器，浏览器不是爬虫，本质上是告诉浏览器我们需要什么水平的文件

    request = urllib.request.Request(url,headers = head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        #rint(html)
    except urllib.error.URLError as e: #遇到404，500等错误
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html

~~~

## 完整版爬取豆瓣电影数据

~~~python
import bs4 #网页解析，获取数据
import re  #正则表达式，进行文字匹配
import urllib.request,urllib.error   #制定URL，获取网页数据
import xlwt #进行excel操作
import sqlite3 #进行SQLite 数据库操作
from bs4   import BeautifulSoup
import re
findLink = re.compile(r'<a href="(.*?)">')   #创建正则表达式对象，表示规则（字符串的模式），影片详情的规则
#影片图片
findImgSrc = re.compile(r'<img.*src="(.*?)"',re.S)#re.S让换行符包含在字符中
#影片片名
findTitle = re.compile(r'<span class="title">(.*)</span>')
#影片评分
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
#找到评价人数
findJudge =re.compile((r'<span>(\d*)人评价</span>'))
#找到概况
findInq = re.compile(r'<span class="inq">(.*)</span>')
#找到影片的相关内容
findBd = re.compile(r'<p class="">(.*?)</p>',re.S)#忽视换行符
def askURL(url):
    head = {   #模拟浏览器头部信息，向豆瓣服务器发送消息
        "user-agent": "Mozilla / 5.0(Windows NT 10.0;Win64;x64) AppleWebKit / 537.36(KHTML, likeGecko) Chrome / 103.0.5060.66Safari / 537.36Edg / 103.0.1264.44"
    }#用户代理，表示告诉豆瓣服务器，我们是什么类型的机器，浏览器不是爬虫，本质上是告诉浏览器我们需要什么水平的文件

    request = urllib.request.Request(url,headers = head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        #rint(html)
    except urllib.error.URLError as e: #遇到404，500等错误
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html
#爬取网页
def getData(beaseurl):
    datalist = []
    for i in range(0,10):
        url = beaseurl +str(i*25)
        html = askURL(url)
        soup = BeautifulSoup(html,"html.parser")
        for item in soup.find_all('div',class_ ="item"):  #表示div中class仍有一个属性为item
            data=[]   #保存一部电影的所有信息
            item = str(item)
            #获取到影片详情的超链接
            link = re.findall(findLink,item)[0]# re库用来通过正则表达式查找指定的字符串,并且找到的数据会以列表的形式存在link
            data.append(link)
            imgSrc=re.findall(findImgSrc,item)[0]
            data.append(imgSrc)
            titles = re.findall(findTitle,item) #片名可能只有一个中国名没有外国名
            if(len(titles)==2):
                ctitle = titles[0]       #添加中文名
                data.append(ctitle)      #添加外国名
                otitle = titles[1].replace('/',"") #去掉无关的符号
                data.append(otitle)
            else:
                data.append(titles[0])
                data.append(' ')  #留空
            rating = re.findall(findRating,item)[0]
            data.append(rating)             #添加评分
            judgeNum = re.findall(findJudge,item)[0]
            data.append(judgeNum)
            inq = re.findall(findInq,item)
            if len(inq)  !=0:
                inq =inq[0].replace("。","") #去掉句号
                data.append(inq)
            else:
                data.append(" ")
            bd = re.findall(findBd,item)[0]
            bd = re.sub('<br(\s+)?/>(\s+?)'," ",bd)#替换的意思,去掉<br/>
            bd = re.sub('/'," ",bd)#替换/
            data.append(bd.strip())#去掉前后的空格
            datalist.append(data)
    print(datalist)
    return datalist
getData("https://movie.douban.com/top250?start=")

~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/5T%5BH~6BDBJ8MUAHH3$ML9%7B3.png)

## 保存数据

![img](file:///C:\Users\DELL\AppData\Roaming\Tencent\Users\731453367\QQ\WinTemp\RichOle\L%4%LZG_[RA%[D`XEO2`$GU.png)

~~~python
#保存数据
def saveData(datalist,savepath):
    print("save...")
    book = xlwt.Workbook(encoding="utf-8",style_compression=0)#创建workbook对象
    sheet = book.add_sheet('豆瓣电影Top250',cell_overwrite_ok=True)
    col = ("电影详情链接","图片链接","影片中文名","影片外国名","评分","评价数","概况","相关信息")
    for i in range(0,8):
        sheet.write(0,i,col[i])
    for i in range(0,250):
        print("第%d条"%(i+1))
        data = datalist[i]
        for j in range(0,8):
            sheet.write(i+1,j,data[j])
            book.save('豆瓣电影Top250.xls')
~~~

### 最终成品：

~~~python
import bs4 #网页解析，获取数据
import re  #正则表达式，进行文字匹配
import urllib.request,urllib.error   #制定URL，获取网页数据
import xlwt #进行excel操作
import sqlite3 #进行SQLite 数据库操作
from bs4   import BeautifulSoup
import re
findLink = re.compile(r'<a href="(.*?)">')   #创建正则表达式对象，表示规则（字符串的模式），影片详情的规则
#影片图片
findImgSrc = re.compile(r'<img.*src="(.*?)"',re.S)#re.S让换行符包含在字符中
#影片片名
findTitle = re.compile(r'<span class="title">(.*)</span>')
#影片评分
findRating = re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
#找到评价人数
findJudge =re.compile((r'<span>(\d*)人评价</span>'))
#找到概况
findInq = re.compile(r'<span class="inq">(.*)</span>')
#找到影片的相关内容
findBd = re.compile(r'<p class="">(.*?)</p>',re.S)#忽视换行符
def askURL(url):
    head = {   #模拟浏览器头部信息，向豆瓣服务器发送消息
        "user-agent": "Mozilla / 5.0(Windows NT 10.0;Win64;x64) AppleWebKit / 537.36(KHTML, likeGecko) Chrome / 103.0.5060.66Safari / 537.36Edg / 103.0.1264.44"
    }#用户代理，表示告诉豆瓣服务器，我们是什么类型的机器，浏览器不是爬虫，本质上是告诉浏览器我们需要什么水平的文件

    request = urllib.request.Request(url,headers = head)
    html = ""
    try:
        response = urllib.request.urlopen(request)
        html = response.read().decode("utf-8")
        #rint(html)
    except urllib.error.URLError as e: #遇到404，500等错误
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
    return html
#爬取网页
def getData(beaseurl):
    datalist = []
    for i in range(0,10):
        url = beaseurl +str(i*25)
        html = askURL(url)
        soup = BeautifulSoup(html,"html.parser")
        for item in soup.find_all('div',class_ ="item"):  #表示div中class仍有一个属性为item
            data=[]   #保存一部电影的所有信息
            item = str(item)
            #获取到影片详情的超链接
            link = re.findall(findLink,item)[0]# re库用来通过正则表达式查找指定的字符串,并且找到的数据会以列表的形式存在link
            data.append(link)
            imgSrc=re.findall(findImgSrc,item)[0]
            data.append(imgSrc)
            titles = re.findall(findTitle,item) #片名可能只有一个中国名没有外国名
            if(len(titles)==2):
                ctitle = titles[0]       #添加中文名
                data.append(ctitle)      #添加外国名
                otitle = titles[1].replace('/',"") #去掉无关的符号
                data.append(otitle)
            else:
                data.append(titles[0])
                data.append(' ')  #留空
            rating = re.findall(findRating,item)[0]
            data.append(rating)             #添加评分
            judgeNum = re.findall(findJudge,item)[0]
            data.append(judgeNum)
            inq = re.findall(findInq,item)
            if len(inq)  !=0:
                inq =inq[0].replace("。","") #去掉句号
                data.append(inq)
            else:
                data.append(" ")
            bd = re.findall(findBd,item)[0]
            bd = re.sub('<br(\s+)?/>(\s+?)'," ",bd)#替换的意思,去掉<br/>
            bd = re.sub('/'," ",bd)#替换/
            data.append(bd.strip())#去掉前后的空格
            datalist.append(data)
    print(datalist)
    return datalist
#保存数据
def saveData(datalist,savepath):
    print("save...")
    book = xlwt.Workbook(encoding="utf-8",style_compression=0)#创建workbook对象
    sheet = book.add_sheet('豆瓣电影Top250',cell_overwrite_ok=True)
    col = ("电影详情链接","图片链接","影片中文名","影片外国名","评分","评价数","概况","相关信息")
    for i in range(0,8):
        sheet.write(0,i,col[i])
    for i in range(0,250):
        print("第%d条"%(i+1))
        data = datalist[i]
        for j in range(0,8):
            sheet.write(i+1,j,data[j])
            book.save('豆瓣电影Top250.xls')
def main():
    baseurl = "https://movie.douban.com/top250?start="
    datalist = getData(baseurl)
    savepath="豆瓣电影Top250.xls"
    saveData(datalist,savepath)
#注意datalist为一个二维列表，其中每一个元素中又为一个一维列表
main()
print("爬取完毕")
~~~

![img](https://wuxidixi.oss-cn-beijing.aliyuncs.com/img/I4HGW61GTC50DOOOQ%60M5D82.png)

### 

