# coding=utf-8
""" PANDAS_NOTE """
import pandas as pd

# 单元7:Pandas库入门

# 一 Pandas库的介绍

# Pandas是Python第三方库，提供高性能易用数据类型和分析工具
# Pandas基于NumPy实现，常与NumPy和Matplotlib一同使用

# Pandas库的理解
# 两个数据类型：Series, DataFrame
# 基于上述数据类型的各类操作
# 基本操作、运算操作、特征类操作、关联类操作

# 二 Pandas库的Series类型
# Series类型由一组数据及与之相关的数据索引组成

# a = pd.Series([9, 8, 7, 6])
# print(a)

# a = pd.Series([9, 8, 7, 6], index=['a', 'b', 'c', 'd'])
# print(a)

# 1.Series类型可由如下类型创建
# Python列表，index与列表元素个数一致
# 标量值，index表达Series类型的尺寸
# Python字典，键值对中的“键”是索引，index从字典中进行选择操作
# ndarray，索引和数据都可以通过ndarray类型创建
# 其他函数，range()函数等

# a = pd.Series(25, index=['a', 'b', 'c', 'd'])
# print(a)

# 2.Series类型的基本操作
# Series类型包括index和values两部分

# Series类型的操作类似ndarray类型
# 索引方法相同，采用[]
# NumPy中运算和操作可用于Series类型
# 可以通过自定义索引的列表进行切片
# 可以通过自动索引进行切片，如果存在自定义索引，则一同被切片

# Series类型的操作类似Python字典类型
# 通过自定义索引访问
# 保留字in操作
# 使用.get()方法

# 3.Series类型的对齐操作
# Series类型在运算中会自动对齐不同索引的数据

# 4.Series类型的name属性
# Series对象和索引都可以有一个名字，存储在属性.name中

# 5.Series类型的修改
# Series对象可以随时修改并即刻生效

# 三 Pandas库的DataFrame类型

# DataFrame类型由共用相同索引的一组列组成
# DataFrame是一个表格型的数据类型，每列值类型可以不同
# DataFrame既有行索引、也有列索引
# DataFrame常用于表达二维数据，但可以表达多维数据

# 获得行的数据 .ix()

# DataFrame类型可以由如下类型创建：
# 二维ndarray对象
# 由一维ndarray、列表、字典、元组或Series构成的字典
# Series类型
# 其他的DataFrame类型

# DataFrame是二维带“标签”数组
# DataFrame基本操作类似Series，依据行列索引

# 四 Pandas库的数据类型操作

# 如何改变Series和DataFrame对象？
# 增加或重排：重新索引 --- .reindex()能够改变或重排Series和DataFrame索引
# 删除：drop

# .reindex(index=None, columns=None, …)的参数
# index, columns  新的行列自定义索引
# fill_value      重新索引中，用于填充缺失位置的值
# method          填充方法, ffill当前值向前填充，bfill向后填充
# limit           最大填充量
# copy            默认True，生成新的对象，False时，新旧相等不复制

# Series和DataFrame的索引是Index类型---Index对象是不可修改类型

# 索引类型的常用方法
# .append(idx)        连接另一个Index对象，产生新的Index对象
# .diff(idx)          计算差集，产生新的Index对象
# .intersection(idx)  计算交集
# .union(idx)         计算并集
# .delete(loc)        删除loc位置处的元素
# .insert(loc,e)      在loc位置增加一个元素e

# .drop()能够删除Series和DataFrame指定行或列索引
# 多维数组时, 需要指定axis
# a.drop("some", axis=1)

# 五 Pandas库的数据类型运算

# 算术运算法则
# 算术运算根据行列索引，补齐后运算，运算默认产生浮点数
# 补齐时缺项填充NaN (空值)
# 二维和一维、一维和零维间为广播运算
# 采用+ ‐ * /符号进行的二元运算产生新的对象

# 方法形式的运算
# .add(d, **argws) 类型间加法运算，可选参数
# .sub(d, **argws) 类型间减法运算，可选参数
# .mul(d, **argws) 类型间乘法运算，可选参数
# .div(d, **argws) 类型间除法运算，可选参数

# 比较运算法则
# 比较运算只能比较相同索引的元素，不进行补齐
# 二维和一维、一维和零维间为广播运算
# 采用> < >= <= == !=等符号进行的二元运算产生布尔对象

# 单元8:Pandas数据特征分析

# 一 数据的排序

# 数据形成有损特征的过程
# 一维数据->摘要->基本统计/分布/数据特征/数据挖掘

# Pandas库的数据排序

# .sort_index()方法在指定轴上根据索引进行排序，默认升序
# .sort_index(axis=0, ascending=True)

# .sort_values()方法在指定轴上根据数值进行排序，默认升序
# Series.sort_values(axis=0, ascending=True)
# DataFrame.sort_values(by, axis=0, ascending=True)
# by :axis轴上的某个索引或索引列表

# NaN统一放到排序末尾

# 二 数据的基本统计分析

# .sum()              计算数据的总和，按0轴计算，下同
# .count()            非NaN值的数量
# .mean() .median()   计算数据的算术平均值、算术中位数
# .var() .std()       计算数据的方差、标准差
# .min() .max()       计算数据的最小值、最大值
# .argmin() .argmax() 计算数据最大值、最小值所在位置的索引位置（自动索引）
# .idxmin() .idxmax() 计算数据最大值、最小值所在位置的索引（自定义索引）
# .describe()         针对0轴（各列）的统计汇总

# 三 数据的累计排序分析

# .cumsum()   依次给出前1、2、…、n个数的和
# .cumprod()  依次给出前1、2、…、n个数的积
# .cummax()   依次给出前1、2、…、n个数的最大值
# .cummin()   依次给出前1、2、…、n个数的最小值

# .rolling(w).sum()           依次计算相邻w个元素的和
# .rolling(w).mean()          依次计算相邻w个元素的算术平均值
# .rolling(w).var()           依次计算相邻w个元素的方差
# .rolling(w).std()           依次计算相邻w个元素的标准差
# .rolling(w).min() .max()    依次计算相邻w个元素的最小值和最大值

# 四 数据的相关分析

# 两个事物，表示为X和Y，如何判断它们之间的存在相关性？
# • X增大，Y增大，两个变量正相关
# • X增大，Y减小，两个变量负相关
# • X增大，Y无视，两个变量不相关

# 两个事物，表示为X和Y，如何判断它们之间的存在相关性？
# • 协方差>0, X和Y正相关
# • 协方差<0, X和Y负相关
# • 协方差=0, X和Y独立无关

# 皮尔森相关系数

# .cov()      计算协方差矩阵
# .corr()     计算相关系数矩阵, Pearson、Spearman、Kendall等系数
