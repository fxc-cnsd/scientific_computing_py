# coding=utf-8
""" MATPLOTLIB_NOTE """
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.rcParams["font.family"] = 'SimHei'

# Matplotlib库由各种可视化类构成，内部结构复杂，受Matlab启发
# matplotlib.pyplot是绘制各类可视化图形的命令子库，相当于快捷方式

# 单元4:Matplotlib库入门

# 一 Matplotlib库的介绍

# 1.matplotlib库小测

# plt.plot()只有一个输入列表或数组时，参数被当作Y轴，X轴以索引自动生成
# plt.plot([3, 1, 4, 5, 2])
# plt.ylabel("Grade")
# plt.show()

# plt.savefig()将输出图形存储为文件，默认PNG格式，可以通过dpi修改输出质量
# plt.plot([3, 1, 4, 5, 2])
# plt.ylabel("Grade")
# plt.savefig('test', dpi=600)
# plt.show()

# plt.plot(x,y)当有两个以上参数时，按照X轴和Y轴顺序绘制数据点
# plt.plot([0, 2, 4, 6, 8], [3, 1, 4, 5, 2])
# plt.ylabel("Grade")
# # 设置横纵坐标尺度
# plt.axis([-1, 10, 0, 6])
# plt.show()

# 2.pyplot的绘图区域

# 在全局绘图区域中创建一个 分区体系，并定位到一个子绘图区域
# 横轴数量/纵轴数量/定位哪个区域
# plt.subplot(nrows, ncols, plot_number)

# def f(t):
#     return np.exp(-t) * np.cos(2 * np.pi * t)

# a = np.arange(0.0, 5.0, 0.02)

# plt.subplot(211)
# plt.plot(a, f(a))

# plt.subplot(2, 1, 2)
# plt.plot(a, np.cos(2 * np.pi * a), "r--")
# plt.show()

# 二 pyplot的plot()函数

# plt.plot(x, y, format_string, **kwargs)
# x: x轴数据,列表或者数组,可选
# y: y轴数据,列表或数组
# format_string: 控制曲线的格式化字符串,可选
# **kwargs: 第二组或更多(x,y,format_string)     # 当绘制多条曲线时，各条曲线的x不能省略

# 绘制多条曲线
# a = np.arange(10)
# plt.plot(a, a * 1.5, a, a * 2.5, a, a * 3.5, a, a * 4.5)
# plt.show()

# format_string: 控制曲线的格式字符串，可选
# 由颜色字符、风格字符和标记字符组成

# 颜色字符
# 'b'         蓝色
# 'g'         绿色
# 'r'         红色
# 'c'         青绿色 cyan
# '#008000'   RGB某颜色
# 'm'         洋红色 magenta
# 'y'         黄色
# 'k'         黑色
# 'w'         白色
# '0.8'       灰度值字符串

# 风格字符
# '‐'     实线
# '‐‐'    破折线
# '‐.'    点划线
# ':'     虚线
# ''/' '  无线条

# 标记字符
# '.' 点标记
# ',' 像素标记(极小点)
# 'o' 实心圈标记
# 'v' 倒三角标记
# '^' 上三角标记
# '>' 右三角标记
# '<' 左三角标记
# '1' 下花三角标记
# '2' 上花三角标记
# '3' 左花三角标记
# '4' 右花三角标记
# 's' 实心方形标记
# 'p' 实心五角标记
# '*' 星形标记
# 'h' 竖六边形标记
# 'H' 横六边形标记
# '+' 十字标记
# 'x' x标记
# 'D' 菱形标记
# 'd' 瘦菱形标记
# '|' 垂直线标记

# a = np.arange(10)
# plt.plot(a, a * 1.5, "go-", a, a * 2.5, "rx", a, a * 3.5, "*", a, a * 4.5,
#          "b-.")
# plt.show()

# color           : 控制颜色, color='green'
# linestyle       : 线条风格, linestyle='dashed'
# marker          : 标记风格, marker='o'
# markerfacecolor : 标记颜色, markerfacecolor='blue'
# markersize      : 标记尺寸, markersize=20

# 三 pyplot的中文显示

# 1.第一种方法(改变全局字体)
# pyplot并不默认支持中文显示，需要rcParams修改字体实现
# import matplotlib
# matplotlib.rcParams["font.family"] = 'SimHei'

# font.family     字体(黑体,楷体,隶书,仿宋,幼圆,宋体)
# font.style      字体风格  normal正常 或者italic斜体
# font.size       字体大小  large/x-small

# a = np.arange(0.0, 5.0, 0.02)
# plt.xlabel("横轴:时间")
# plt.ylabel("纵轴:振幅")
# plt.plot(a, np.cos(2 * np.pi * a), "r--")
# plt.show()

# 2.第二种方法(改变部分字体)
# 在有中文输出的地方，增加一个属性：fontproperties
# a = np.arange(0.0, 5.0, 0.02)
# plt.xlabel("横轴:时间", fontproperties="SimHei", fontsize=20)
# plt.ylabel("纵轴:振幅", fontproperties="SimHei", fontsize=20)
# plt.plot(a, np.cos(2 * np.pi * a), "r--")
# plt.show()

# 四 pyplot的文本显示

# plt.xlabel()    对X轴增加文本标签
# plt.ylabel()    对Y轴增加文本标签
# plt.title()     对图形整体增加文本标签
# plt.text()      在任意位置增加文本
# plt.annotate()  在图形中增加带箭头的注解

# a = np.arange(0.0, 5.0, 0.02)
# plt.plot(a, np.cos(2 * np.pi * a), "r--")
# plt.xlabel("横轴:时间", fontproperties="SimHei", fontsize=15, color="green")
# plt.ylabel("纵轴:振幅", fontproperties="SimHei", fontsize=15)
# plt.title(r"正弦波实例 $y=cos(2\pi x)$", fontproperties="SimHei", fontsize=25)
# plt.text(2, 1, r"$\mu=100$", fontsize=15)
# plt.axis([-1, 6, -2, 2])
# plt.grid(True)
# plt.show()

# 箭头 # plt.annotate(s, xy=arrow_crd, xytext=text_crd, arrowprops=dict)
# a = np.arange(0.0, 5.0, 0.02)
# plt.plot(a, np.cos(2 * np.pi * a), "r--")
# plt.xlabel("横轴:时间", fontproperties="SimHei", fontsize=15, color="green")
# plt.ylabel("纵轴:振幅", fontproperties="SimHei", fontsize=15)
# plt.title(r"正弦波实例 $y=cos(2\pi x)$", fontproperties="SimHei", fontsize=25)
# # plt.text(2, 1, r"$\mu=100$", fontsize=15)
# plt.annotate(
#     r"$mu=100$",
#     xy=(2, 1),
#     xytext=(3, 1.5),
#     arrowprops=dict(facecolor="black", shrink=0.1, width=2))
# plt.axis([-1, 6, -2, 2])
# plt.grid(True)
# plt.show()

# 五 pyplot的子绘图区域

# 1.plt.subplot2grid()
# plt.subplot2grid(GridSpec, CurSpec, colspan=1, rowspan=1)
# 理念：设定网格，选中网格，确定选中行列区域数量，编号从0开始

# 2.GridSpec类
# import matplotlib.gridspec as gridspec

# gs = gridspec.GridSpec(3, 3)

# ax1 = plt.subplot(gs[0, :])
# ax2 = plt.subplot(gs[1, :-1])

# 单元5:Matplotlib基础绘图函数示例

# 一 pyplot基础图表函数概述

# plt.plot(x,y,fmt,…)                 绘制一个坐标图
# plt.boxplot(data, notch, position)  绘制一个箱形图
# plt.bar(left,height, width,bottom)  绘制一个条形图
# plt.barh(width,bottom, left,height) 绘制一个横向条形图
# plt.polar(theta, r)                 绘制极坐标图
# plt.pie(data, explode)              绘制饼图
# plt.psd(x,NFFT=256,pad_to,Fs)       绘制功率谱密度图
# plt.specgram(x,NFFT=256,pad_to,F)   绘制谱图
# plt.cohere(x,y,NFFT=256,Fs)         绘制X‐Y的相关性函数
# plt.scatter(x,y)                    绘制散点图，其中，x和y长度相同
# plt.step(x,y,where)                 绘制步阶图
# plt.hist(x,bins,normed)             绘制直方图
# plt.contour(X,Y,Z,N)                绘制等值图
# plt.vlines()                        绘制垂直图
# plt.stem(x,y,linefmt,markerfmt)     绘制柴火图
# plt.plot_date()                     绘制数据日期

# 二 pyplot饼图的绘制

# labels = ('Frogs', 'Hogs', 'Dogs', 'Logs')
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)
# plt.pie(
#     sizes,
#     explode=explode,
#     labels=labels,
#     autopct="%1.1f%%",
#     shadow=False,
#     startangle=90)
# plt.show()

# labels = ('Frogs', 'Hogs', 'Dogs', 'Logs')
# sizes = [15, 30, 45, 10]
# explode = (0, 0.1, 0, 0)
# plt.pie(
#     sizes,
#     explode=explode,
#     labels=labels,
#     autopct="%1.1f%%",
#     shadow=False,
#     startangle=90)
# # 正圆
# plt.axis("equal")
# plt.show()

# 三 pyplot直方图的绘制

# np.random.seed(0)
# mu, sigma = 100, 20
# a = np.random.normal(mu, sigma, size=100)
# # 数据/直方的个数/归一化/绘制类型/颜色/尺寸比例
# plt.hist(a, 40, normed=0, histtype="stepfilled", facecolor="b", alpha=0.75)
# plt.title("Histogram")
# plt.show()

# 四 pyplot极坐标图的绘制

# N = 20
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = 10 * np.random.rand(N)
# width = np.pi / 4 * np.random.rand(N)
# ax = plt.subplot(111, projection="polar")
# bars = ax.bar(theta, radii, width=width, bottom=0.0)
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.viridis(r / 10.))
#     bar.set_alpha(0.5)
# plt.show()

# 五 pyplot散点图的绘制

# fix, ax = plt.subplots()
# ax.plot(10 * np.random.randn(100), 10 * np.random.randn(100), "o")
# ax.set_title("Simple Scatter")
# plt.show()

# 实例2:引力波的绘制

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import wavfile
# rate_h, hstrain = wavfile.read(r"H1_Strain.wav", "rb")
# rate_l, lstrain = wavfile.read(r"L1_Strain.wav", "rb")
# # reftime, ref_H1 = np.genfromtxt(
# #     'GW150914_4_NR_waveform_template.txt').transpose()
# reftime, ref_H1 = np.genfromtxt('wf_template.txt').transpose()
# # 使用python123.io下载文件
# htime_interval = 1 / rate_h
# ltime_interval = 1 / rate_l
# fig = plt.figure(figsize=(12, 6))
# # 丢失信号起始点
# htime_len = hstrain.shape[0] / rate_h
# htime = np.arange(-htime_len / 2, htime_len / 2, htime_interval)
# plth = fig.add_subplot(221)
# plth.plot(htime, hstrain, 'y')
# plth.set_xlabel('Time (seconds)')
# plth.set_ylabel('H1 Strain')
# plth.set_title('H1 Strain')
# ltime_len = lstrain.shape[0] / rate_l
# ltime = np.arange(-ltime_len / 2, ltime_len / 2, ltime_interval)
# pltl = fig.add_subplot(222)
# pltl.plot(ltime, lstrain, 'g')
# pltl.set_xlabel('Time (seconds)')
# pltl.set_ylabel('L1 Strain')
# pltl.set_title('L1 Strain')
# pltref = fig.add_subplot(212)
# pltref.plot(reftime, ref_H1)
# pltref.set_xlabel('Time (seconds)')
# pltref.set_ylabel('Template Strain')
# pltref.set_title('Template')
# fig.tight_layout()
# plt.savefig("Gravitational_Waves_Original.png")
# plt.show()
# plt.close(fig)
