# 从gcc/g++到Cmake学习笔记
<span style="font-size:8px;">注：Task任务的Makefile和CmakeLists在demo/Makefile目录下，另外,我先"个人练习"目录中写了cpp,hpp的Makefile练习</span>

## 一.gcc/g++
c/cpp代码的编译过程:
-   [1]预处理：根据条件编译展开宏、处理 #include 指令(hpp/h被识别)。
	操作：gcc -E main.c -o main.i
 ***关于宏***:
	-	define：```define  NAME  <value>```将一个value替换为NAME，但value并不确定类型，只是字符串之间的替换(代码本身都是字符串)，最终宏根据上下文确定类型:
	```cpp
	define V 5
	int a=V//V=5
	float b=V//V=5.0
	char c=V//根据ASCLL表替换为5对应的字符
	char d='V'//这里不是宏，只是单纯的'V'字母
	```
	-	ifndef与endif:
	```cpp
	ifndiff  M
	define M
	define Pi 3.1415926
	int max(int a,int b){
		return a>b ? a:b;
	}
	endif
	```
	如果此前M没有被宏定义，就执行ifndif~endif之间的代码，否则跳过；这样能在多处include这个头文件时，只会在第一个include的地方展开，否则函数max和宏PI在多个include这个头文件的地方被定义，触发重复定义的错误。

-    [2]编译：将预处理后的代码转换为汇编语言,进行词法、语法、语义分析，生成中间代码并优化
	 -	gcc -S main.o -o main.s
-    [3]汇编：将汇编代码转换为机器码(二进制)
	-	动词：gcc -c main.s -o main.o(Linux)/main.obj(Windows)
-    [4]链接：将多个 .o 文件和库文件(静态库与动态库)组合成可执行文件
	-	操作：```g++ a.o b.o -o project```后缀：不指定名字时，Linux下默认a.out,Windows下为a.exe，指定名字时不加后缀(如这里的project)

从.c直达.o再到executable(可执行的)的命令:
-	```gcc(g++) -c main.cpp -o name1.o```先完成预处理,main.cpp中的```#include <other.hpp>```将hpp内容插入main.cpp,把文本cpp编译为汇编语言,再编为机械码,生成.o目标文件.加上-o在后面给.o文件命名为name1.o,不加就是原名加.o后缀:**main.o** 
-	```g++ main.o other.o -o name```将多个.oi链接为可执行文件,并命名为name,不加-o就默认为a.out		
-	最后用```./ name```即可运行	

**教训:我一开始用gcc预处理，编译，汇编，没问题,结果链接时用gcc一堆报错,原来gcc和g++执行前几步都没问题,但是链接期只有g++才能链接c++文件**

````
gcc main.o -o myc
/usr/bin/ld: main.o: warning: relocation against _ZSt4cout in read-only section .text
/usr/bin/ld: main.o: in function main:
main.cpp:(.text+0x28): undefined reference to std::cout
/usr/bin/ld: main.cpp:(.text+0x30): undefined reference t......
````

更改:
```
g++ main.o -o myc
fushao@fushao-VMware-Virtual-Platform:~/Codes/杂七杂八$ ./myc
请输入两个数字
3
6
36
9
```
成功!	


<p style="color:yellow">
补充:
>>1:可以直接"gcc main.cpp other.cpp -o name"中间会自动编译出.o文件,生产name可执行文件,若还不写-o name,默认生产a.out可执行文件    	
>>2:删除文件用```rm name```,若要同时删除.o就:```rm name *.o```
</p>

---

## Makefile
### 基本操作
在gcc的基础上,根据每个命令都是几个文件生成一个文件,可以知晓文件的依赖关系,如.o依赖于.cpp和.hpp,可执行文件依赖于.o,那么Makefile中把各个依赖单独写命令
```
target:rely_file1,rely_file2
	gcc...#还是gcc的命令
```
如:
```
myc:main.o
	g++ main.o -o myc 
	
main.o:main.cpp func.hpp
	g++ main.cpp

clean:
	rm myc *.o
```
### 进阶1：
如果想在执行gcc/++之前先建一个文件夹等操作，那么让第一块依赖要依赖与加在最前面的操作：
```makefile
prepare:
	mkdir build#最先的操作
myc:main.o|prepare#此操作依赖于prepare操作
	g++ ....
```
### 进阶2：
makefile中可以给变量赋值，并在使用时```$(变量名)或${变量名}```来使用,类似于格式化，如
```makefile
CC=gcc
myc:main.o
	$(CC) main.o -o myc#等价于gcc -o myc
#文件也能赋值
files=main.o tool.o
project:$(files)
	gcc $(files) -o project
```
## CMake
### 基本操作
-	CMake中无需像Makefile里面手动再-c链接hpp,而是通过Cmake自带的${PROJECT_SOURCE_DIR}获取当前路径,再/include寻找当前文件夹中叫include的文件夹,用include_directories指定所有hpp都在这里找
-	add_executable(可执行文件名 源文件),不用管.o文件,CMake会自己创建文件夹放
-	为了避免Cmake创建的文件污染项目所在文件夹,可以先在项目文件夹中u创建一个叫builld的文件专门存放cmake的文件:
	-	先进入当前目录```cd ~/Codes/机械学习Warm\ Up\!/```
	-	再创建并进入build``` mkdir build && cd build```
	-	让cmake在上一级找CMakeLists文件:```cmake ..``` **注意camke和 .. 要空格,我没打空格到处找原因被整破防了~**
-	上一步已经生成了makefile文件,现在```make```就可以了
-	最后直接删除build文件夹即可```rm build```,其他文件没有受到任何污染

```CMake
cmake_minimum_required(VERSION 3.10)

project(MyProject)

include_directories(${PROJECT_SOURCE_DIR}/include)
#这里的${}时Cmake自动识别的当前路径，和上文makefile一样是格式化
add_executable(myc src/main.cpp)
```
### 进阶
#### 配对可执行程序找源文件的路径
include_directories()会作用于全局可执行程序的cpp文件搜索头文件方式，会导致相互污染，因此最好指定每个可执行程序搜索它的头文件的搜索方式
```
add_executable(project a.cpp b.cpp c.cpp)
#若a的源文件在include目录，b的源文件在src目录
target_include_directores(project PRIVATE
	include 
	src
	~/Libs/autodiff)
```
另外，在头文件include时，对于官方的头文件，他们在系统标准目录中，是gcc/g++默认搜索路径，直接```include<vector>```，而第三方文件的路径gcc/g++难找到，需要显式指出路径:
```
include "../include/tool.hpp"#相对于当前cpp文件的路径
include"~/autodiff/reverse/var.hpp"#绝对路径
```
#### 连接库
库是代码文件的文件夹，里面可能包含多个cpp/hpp文件及子文件夹；有的库里面有Cmake文件，可以在我们自己的Camke中直接找到它:
```
find_package(Eigen3 ...)
target_link_libiraries(project PRIVATE
	Eigen::Eigen)#和找源文件相似
```

## python与C/C++
### 基础概念：
强类型语言：不允许隐式转换，比如int a=32+"abb"报错，需要显示写int(c),d.ToString()等显示转换
弱类型语言：隐式转换较宽松，比如char可以转换为ASCll表对应的int索引
静态类型语言：变量声明时要写变量类型，编译期确定，之后不能变成其他类型
动态类型语言：运行时才绝定变量类型
### 为何C比python快
C是静态弱类型语言(C++是静态较弱类型)，编译期就确定类型，值类型在栈上，内存可以连续，CPU方便找
python是动态强类型语言，边执行边解释，实时进行类型检查;且一切皆对象，每个变量都是指向堆上对象的指针，可以随时更换指针指向的对象如```a=20;b="name";a=b```(注意，python是强类型，不能a+b)；另外，python无法实现并发，垃圾收集在主线程中进行。
### C++写核心逻辑，python负责
-	python简单灵活，生态丰富，能够快速做出demo，适合做上层业务逻辑;且作为胶水语言，python可以方便地做出接口,适合频繁更新优化
-	C++直接操作硬件，性能高，写性能要求高的计算逻辑，还能控制并发