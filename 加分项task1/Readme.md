# Task1加分项：用cpp写神经网络————实验报告
<p style="color:red">ΔΔΔ!</p>
<p style="color:yellow">process_moon.cpp中梯度计算的函数调用遇到了掉头发的问题，代码无法完整运行，但我把代码每一步的实现思路和解决问题的过程中学到的知识写在本文</p>

## 框架
-   加载数据：使用task2用python的sklearn来加载moon函数，再导出为moons.csv文件，cpp端用字符串流读取
-   include:由于隐藏层的激活函数是非线性的，无法简单计算，我使用autodiff库的自动微分函数(问题正出在它的函数调用);并用Eigen3库来使用矩阵
-   手搓各种函数：
    -   1.标准化Normalize:使用template泛型模板T,对矩阵MatrixXvar,Matrixxd,VectorXvar,VectorVd都能进行标准化
    -   2.打乱数据Shuffle:
        -   1.先用random_device建种子rd,再用mt19937生成引擎，当读取到样本数时再复制uniform_int_distribution分布器的取值范围
        -   2.准备一个哈希词典存放每个样本(Sanple结构体，包含特征和标签)，再准备一个哈希表存放已经取出的样本索引
        -   3.在while循环中生成(0~sample_size-1)的索引，若无法在unoredered_set中find索引则根据此索引在哈希词典中抽取，若未被抽取过，则技术++,并将把特征和标签抽出来放进向量，后续赋值给MatrixXvar和VectorXvar；
    -   3.激活函数:tanh在array中内置，只需Matrix->array->.tanh()->Matrix,而Sigmoid需要手写。sigmoid(x)=1/(1+e^(-x))
    -   4.准确率计算Acc:每轮需要储存本轮的正确率和总数，用完清零，因此我直接将其封装为一个类，用私有成员储存上述内容，并用公开函数每次根据标签和预测值计算正确数，最后打印正确率
    -   5.前向传播：由于训练集和测试集需求不同，前向传播到计算损失就截止，并加入参数isVal来选择模式(默认值为false);中间用MatrixXvar的计算和pytorch一样简单
    -   6.计算梯度：此处官方文档和Ai都没有给出清晰答案，我打开autodiff库的众多头文件，配置两三天，依然无果，以下说明遇到的阻碍和学到的内容：
-   读取文件：
用ifstream将文件中每一行变为数据流而非字符串的形式，可用getline分行读取，读取到的line是字符串，再用stringsream将string转换为流，把流再给到getling分","读取，最后用stod转换为值类型，每个样本存入结构体Sample中，所有Sample存如哈希字典中。
        -   1.阻碍：
            -   1.官方文档：计算梯度的函数有jacobian,gradient,derivative,然而官方文档实例中只include了forard的头文件，没有讲解反向模式的使用，讲解的部分也没有出现返回MatrixVar/MatrixXd，基本上都是var,少数是VectorXvar;autodiff
            -   2.Ai幻觉：或许由于此库不流行(我在b站上只看到一篇医学方面的文档提到了它，而没有pythorch那种系统讲解)，Ai自身的语料不包含它，而CoT也难以检索，我试着按照Ai的操作最终是反复绕圈子却无所获(emo++)
            -   3.autodiff库的文件夹：前两者都无济于事，我最终打开文件夹，文件目录分为common(基础函数模板)和forward(前向)和reverse(反向)，文件夹内出现互相include的操作，命名空间嵌套赋值，template一大堆，高级语法看不懂...最终找到了求导函数，却在forward/utils中,在include和using namespace时两者不统一，下述：
        -   2.学到的：
            -   1.include<头文件>时，gcc优先在系统默认目录中找，适合官方头文件<string>,<math>,STL等，但第三方库的路径需要用include"头文件"来让gcc搜索指定的路径，为了让头文件中不要直接指定路径，需要在cmake中用target_include_directories(相对路径)或target_include_libraries(若此库支持cmake,先用package_find找到，如Eigen3,而autodiff没有，要放到项目文件夹中用前者的相对路径);另外，反复折腾头文件配置的过程中，我从不适应每次```mkdir build && cd build```到熟悉Linux路径的许多操作了，此内容已加在机器学习Warm Up!/Cmake文件夹中
            -   2.autodiff的函数的大体概念：forward模式适合少量输入，多输出，而reverse模式适合多输入，少输出；每个函数有3个基本参数: ```f wrt(A) at(B)```:f是函数名，A是求导的参数，at是函数点的坐标，即所有参数，才能确定唯一点，此外，函数本身输出的是对wrt的梯度，但可输入一个参数用来存储函数值(类似C#的out)，只是不知wrt是否可以传矩阵甚至多个矩阵，整个函数返回的是标量var还是向量展平后的VectorXd or MatrixXvar还是矩阵MatrixXd or MatrixXvar,亦或是同wrt类型的元组tuple<Matrix,Matrix...,Vector>,我期望的是最后的，但并非如此
            -   3.头文件和命名空间：函数所在的头文件必须被识别,但如果它所在的头文件通过外层文件一层层include到最外层，则只需include最外层，因为编译器会展开所有include的文件，由外向内能追溯到内层，但是命名空间必须是它所在头文件中的namespace{}结构;我定位到了gradient的命名空间，可wrt和at和它居然不在一起?

