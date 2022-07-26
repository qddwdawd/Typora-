# python进制转换

![python进制转换](C:\Users\DELL\Desktop\work\picture\python进制转换.png)

### 2进制转换成其他进制：

1. 2进制转化成8进制：oct(int(n,2))

~~~python
n = input()
print(oct(int(n,2)))
~~~

- 例如：输入：1010 输出：0o12

2. 2进制转化成10进制：int(n,2)

~~~python
n = input()
print(int(n,2))
~~~

- 例如：输入：1010 输出：10

3. 2进制转换成16进制： hex(int(n,2))

~~~python
n = input()
print(hex(int(n,2)))
~~~

- 例如：输入：1010输出：0xa

### 8进制转换成其他进制：

1. 8进制转换成2进制： bin(int(n,8))

~~~python
n = input()
print(bin(int(n,8)))
~~~

- 例如：输入：1010 输出：0b1000001000

2. 8进制转换成10进制：int(n,8)

~~~python
n = input()
print(int(n,8))
~~~

- 例如：输入：1010输出：520

3. 8进制转换成16进制：hex（int(n,16))

~~~python
n = input()
print(hex(int(n,8))
~~~

- 例如：输入1010输出：0x208

### 10进制转换成其他进制：

1. 10进制转换成2进制：bin(n)

~~~python
n = int(input())
print(bin(n))
~~~

- 例如：输入：10输出：0b1010

2. 10进制转换成8进制：oct(n)

~~~python
n = int(input())
print(oct(n))
~~~

- 例如：输入：1输出：0o12

3. 10进制转换成16进制：

~~~python
n = int(input())
print(hex(n))
~~~

- 例如：输入：10输出：0xa

### 16进制转换成其他进制：

1. 16进制转换成2进制：bin(int(n,16))

~~~python
n=input()
print(bin(int(n,16)))
~~~

- 例如：输入：a，输出：0b1010

2. 16进制转换成8进制：oct(int(n,16))

~~~python
n=input()
print(oct(int(n,16)))
~~~

- 例如：输入：a，输出：0o12

3. 16进制转换成10进制：int(n,16)

~~~python
n=input()
print((int(n,16)))
~~~

- 例如：输入：a，输出：10