# Python 继承

1. 析构方法                                                      
2. 单继承
3. 多继承
4. 继承的传递
5. 重写父类方法
6. 调用父类的方法
7. 多态
8. 类属性和实例属性
9. 类方法和静态方法

### 重点

- 类的继承

- 弗雷德调用

- 静态方法

### 难点

- 继承与重写

- 静态方法

  ***

  ## 析构

  #### 析构方法的概述

  当一个对象被删除或者被销毁时，python解释器也会默认调用一个方法，这个方法为__ d e l__( )方法，也成为析构方法。(魔术方法)

  //创建对象时，需要进行实例化对象的创建，首先进行__ n e w__ 函数的操作，来进行创建，python解释器会自动进行此函数的调用。而当删除一个对象时，python解释器会自动用到析构方法  --d e l--来销毁这个对象。

  ~~~python
  class Animal(object):
      def __init__(self,name):   #构造方法，创建实例化属性#
          self.name = name
         print("__init__方法被调用")
      #析构方法，当对象被销毁时，python解析器会自动调用
      
      def __del__(self):
          print("__del__方法被调用")
          print("%s 对象被销毁"%self.name)
    dog = Animal("旺财")
  ~~~

  **程序执行结束自动调用 --  d e l --方法**

  可以看到输出结果为：

  ~~~python
  __init__方法被调用
  __del__方法被调用
  旺财对象被销毁
  ~~~

  当在某个作用域下面没有被使用，或者被引用的情况下，解析器会自动地调用 --del--函数来释放内存空间。

  ![del函数的使用](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\del函数的使用.png)  如图所示，当再加一个input函数时，系统会等待你输入，而在等待的过程中，代码并不会结束，所以del函数尚未执行，因为他不知道你所创造的cat实例对象是否还需要使用，所以并未销毁cat。

- 可以手动删除对象，用del 函数来实现。

  ~~~python
  del cat
  ~~~

- 如果在实例化过程中没有输入del()函数，他也会默认执行，就像是new()函数一样。
- del函数需要释放空间，一旦释放完毕，对象便不能使用。

比如：

~~~python
del cat
print(cat.name)
#这个是不能成功执行的
~~~

#### 析构方法总结

![del函数的总结](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\del函数的总结.png)

# 继承

- 在python中展现面向对象的三大特征：
- 封装、继承、多态
- 封装：指的是把内容封装到某个地方，便于后面的使用
- 它需要：
- 把内容封装到某个地方
- 从另外一个地方去调用被封装的内容
- 对于封装来说，其实就是使用初始化构造方法将内容封装到对象中，然后通过对象直接或者self简洁的获取对封装的内容
- 继承：和现实生活当中的继承是一样的：也就是 子可以继承父的内容[属性]和[行为] (爸爸有的儿子都有，而儿子有的爸爸不一定有)。

***

### 单继承

~~~python
//动物的例子
class Animal:
    def eat(self):
        print("吃饭啦")
        pass
    def drink(self):
        
        pass
class Dog(Animal):  #继承animal父类 此时dog就是子类
    def wwj(self):  #子类独有的实现
        print("小狗汪汪叫")
    pass
class Cat(Animal):
    def mmj(self):  #子类独有的实现
        print("小猫喵喵叫")
    pass
d1 = Dog()
d1.eat()   #具备了吃的行为 是继承了父类的行为
print("--------------------------")
c1 = Cat() 
c1.eat()
~~~

- 上述代码中，eat 和 drink为父类所有的行为，dog和 cat子类可以继承，这叫做单继承，而子类中独有的wwj和mmj是独有的实现，他们互不干扰。

- 所以，对于面向对象的继承来说，其实就是将多个类共有的方法提取到父类中，子类仅需要继承父类而不必一一地去实现，这样就可以极大的提高效率，减少代码的重复编写，还可以精简代码的层级结构，比较清晰，便于拓展。

~~~python
class 类名(父类)：
'''
子类就可以继承父类中公共的属性和方法
'''
class 子类(父类名)
~~~

***

## 多继承

- 多继承概念

![多继承的概念](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\多继承的概念.png)

~~~python
class shenxian:
    def fly(self):
        print('神仙都会飞')
    pass
class Monkey:
    def chitao(self)
        print('猴子喜欢吃桃')
class Sunwukong(shenxian,monkey): #既是神仙，同时也是猴子
    pass
swk = Sunwukong()
swk.chitao()
swk.fly()
#问题是，堂多个父类中存在相同方法的时候，应该去调用哪一个呢？
class D(object):
    def eat(self):
        print("D.eat")
class c(D):
    def eat(self):
        print("c.eat")
class B(D):
    pass
class A(B,C)



a = A()
a.eat()
~~~

- 在执行eat时，查找方法的顺序是：

首先到A中去查找，如果没有，则继续道B类中去查找，如果B中没有，则应该去C类中查找，如果c类中没有，则去D类中去查找，如果还没找到，那就会报错。

**__mro__**是可以查询运行的顺序

### 多重继承

~~~python
class GrandFather:
    def eat(self):
        print('吃的 方法')
        pass
    pass
class Father(GrandFather):
    pass
class Son(Father):
    pass
son = Son()
son.eat()   #此方法是从GrandFather中继承过来的。
~~~

**总结**：

类的传递过程中，我们把父类又称为基类，子类又称为派生类，父类的属性和方法可以一级一级的传递到子类。

***

## 重写父类的方法

所谓重写，就是子类中，有一个和父类相同名字的方法，在子类中的方法会覆盖掉父类中同名的方法，伪代码示例：

~~~python
class 父类：
  def 抽烟(self):
      print("抽芙蓉王")
  def 喝酒(self):
      print("喝二锅头")
class 子类(父类):
  def 抽烟(self)：
      print("抽华子")
son = 子类()
son.抽烟()
   
~~~

再例如：

~~~python
class GrandFather:
    def eat(self):
        print('吃的 方法')
        pass
    pass
class Father(GrandFather):
    def eat(self):
        print("吃的")  #因为父类中已经存在这个方法，在这里相当于方法重写[方法覆盖了]
    pass
class Son(Father):
    pass
son = Son()
son.eat()   
~~~

- 默认从下而上，从子类到父类来继承，由子类来覆盖父类。

- 如果在继承父类的过程中，父类中有init的实例属性构造方法， 而子类中没有init的实例属性构造方法，程序会自动报错如下图：

![init实例属性](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\重写父类init的构造实例属性的方法1.png)

~~~python
#如果你调用
kj = kejiquan()
kj.bark()
#会出现以上问题,它会显示init少两个参数，也就是说，无论你调用的实例方法与init有无关系，只要你父类中有init，你不给参数他就会报错，同时也是因为kj = kejiquan()的括号中应为init的参数。
~~~

- 重写父类的构造方法--init--（）
- ![重写父类的init](C:\Users\DELL\Desktop\笔记\笔记截图的保存地址\重写父类的init构造实例属性的方法2.png)

以上的在子类中同时定义一个构造方法(实例属性)，它是属于重写父类的方法，当在子类中重写了init之后，便不能再调用父类中的init方法的内容。

## 调用父类中init方法

~~~python
#如以下代码
class Dog:
    def __init__(self,name,color):
        self.name = name
        self.color = color
     def bark(self):
         print('汪汪叫...')
class kejiquan(Dog):
    def __init__(self,name,color):
         Dog.__init__(self,name,color)
    def bark(self):
         print("叫得很凶")
         print(self.name)
kj = kejiquan('柯基犬','红色')
kj.bark()
~~~

- 实现方法：调用父类中的init方法，需要在子类中的init下面调用Dog.__init__方法，参数在父类子类中都需要调用，这样能够通过子类转递给父类，一级一级向上传递。

- 在此基础之上，还可以书写子类独有的特征，如：

- ~~~python
  class Dog:
      def __init__(self,name,color):
          self.name = name
          self.color = color
      def bark(self):
           print('汪汪叫...')
  class kejiquan(Dog):
      def __init__(self,name,color):
           Dog.__init__(self,name,color)
           self.height = 90
           self.weight = 70
      def bark(self):
           print("叫得很凶")
           print(self.name)
      def __str__(self):
           return("柯基犬的身高是{}，体重是{}".format(self.height,self.weight))
  kj = kejiquan('柯基犬','红色')
  print(kj)
  ~~~

- 打印结果为：柯基犬的身高是90，体重是70

### super的调用方法

~~~python
super().__init__(name,colour)
~~~

- 使用此方法调用时，super会自动地找到父类，进而调用方法。假设继承了多个父类，那么，会按照顺序逐个的去找，找到之后去调用。

~~~python
class Dog:
    def __init__(self,name,color):
        self.name = name
        self.color = color
    def bark(self):
         print('汪汪叫...')
class kejiquan(Dog):
    def __init__(self,name,color):
         super().__init__(name,color)
         self.height = 90屁都
         self.weight = 70
    def bark(self):
         print("叫得很凶")
         print(self.name)
    def __str__(self):
         return("柯基犬的身高是{}，体重是{}".format(self.height,self.weight))
kj = kejiquan('柯基犬','红色')
print(kj)
~~~

- 两种调用方法的区别：

~~~python
Dog.__init__(self,name,color)#这种是手动调用的方法
super().__init__(name,color)#这种是自动调用的方法
~~~

***

## 多态

所谓的多态：就是多种状态、或者形态，就是同一种行为对于不同的子类[对象]有着不用行为表现，就是定义时的类型和运行时的类型不一样，此时就成为多态。

- 要想实现多态，必须有两个前提需要遵守：

1.继承：堕胎必须发生在父类和子类之间

2.重写：子类重写父类的方法
