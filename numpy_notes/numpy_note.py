# coding=utf-8
"""NUMPY_NOTE"""

import numpy as np

# n维数组对象:ndarray = (1实际的数据 + 2描述这些数据的元数据)
# ndarray应由同质对象组成,否则无法发挥优势
# 轴(axis):数据的维度 / 秩(rank):轴的数量

ls = [[0, 1, 2, 3, 4], [9, 8, 9, 6, 5], [4, 6, 7, 9, 6]]
shape = (2, 3, 4)  # 一个尺寸
r_shape = (3, 8)  # 第二个尺寸
val = 3  # 一个值

# 单元1:numpy库入门
"""ndarray的创建"""

# 1.由列表创建
# a = np.array(ls)
# a = np.array(ls, dtype=np.float32)

# 2.numpy函数创建
# a = np.arange(24)           # 类似range
# a = np.ones(shape)         # 全1
# a = np.zeros(shape)        # 全0
# a = np.full(shape, val)    # val填充数组
# a = np.eye(5)  # 单位矩阵

# x = np.ones_like(a)
# x = np.zeros_like(a)  # 仿制
# x = np.full_like(a, val)

# a = np.linspace(1, 10, 4, endpoint=False)  # 等间距填充,endpoint-是否包含尾元素
# a = np.concatenate((a1, a2, ...))             # 合并数组

# 3.由字节流创建

# 4.由文件创建

# print(a)
# print("矩阵的秩 -> ", a.ndim)
# print("矩阵的尺寸-> ", a.shape)
# print("矩阵的元素个数 -> ", a.size)
# print("矩阵的元素类型 -> ", a.dtype)  # bool/int/uint/float/complex
# print("矩阵的元素大小 -> ", a.itemsize)
"""ndarray的变换"""

# 1.维度变换
# a = a.reshape(r_shape)     # 原数组不变,返回改变之后的数组
# a.resize(r_shape)          # 修改原数组
# a = a.swapaxes(0,1)        # 原数组不变,返回交换两个维度的数组
# a = a.flatten()            # 原数组不变,返回降维(1维)后的数组

# 2.类型变换
# a = a.astype(np.float32)   # 原数组不变,返回改变元素类型之后的数组

# 3.ndarray向list转换
# l = a.tolist()      # 转换为列表
# print(l)

# print(a)
# print("矩阵的秩 -> ", a.ndim)
# print("矩阵的尺寸-> ", a.shape)
# print("矩阵的元素个数 -> ", a.size)
# print("矩阵的元素类型 -> ", a.dtype)
# print("矩阵的元素大小 -> ", a.itemsize)
"""ndarray的操作"""

# 索引和切片,同列表
# print("一维切片-> ", a[1:2])    # 越界不报错,返回空列表
# print("多维索引-> ", a[1, 2])  # 多维数组的索引,以逗号分隔
# print("多维切片-> ", a[0:2, 2]) # 多维数组的切片
# print("跳过维度-> ", a[:, 2])   # 分号不可省略
"""ndarray的运算"""

# 1.数组与标量的运算,作用于每一个元素
# mn = a.mean()              # 均值
# print("数组均值-> ", mn)
# print("数组/均值->", a/mn)

# 2.numpy一元函数
# print("绝对值-> \n", np.abs(a))           # 求绝对值
# print("浮点绝对值-> \n", np.fabs(a))    # 求浮点绝对值
# print("平方根-> \n", np.sqrt(a))      # 求平方根
# print("平方-> \n", np.square(a))     # 求平方
# print("对数-> \n", np.log(a))        # 求对数{log10 / log2}
# print("ceiling-> \n", np.ceil(a))      # 求ceil
# print("floor-> \n", np.floor(a))       # 求floor
# print("四舍五入-> \n", np.rint(a))     # 四舍五入
# print("整数小数分离-> \n", np.modf(a))  # 求整数小数分离
# print("三角函数-> \n", np.cos(a))      # 求三角函数
# print("指数值-> \n", np.exp(a))       # 求指数值
# print("符号值-> \n", np.sign(a))      # 求符号

# 3.numpy二元函数
# b = np.sqrt(a)
# print("算术运算 -> \n", a+b)               # + - * / **
# print("最大最小值-> \n", np.maximum(a, b))  # np.maximum() / minimum() / fmax() / fmin()()
# print("模运算-> \n", np.mod(a, b))        # 模运算
# print("复制符号-> \n", np.copysign(a, b))  # b的符号赋给a
# print("关系运算-> \n", a > b)              # 关系运算

# 单元2:numpy数据存取与函数
"""numpy的数据存取"""

# 1.CSV(逗号分隔值)文件存取 --只能有效存取一维和二维数组
# np.savetxt('np_test_csv.csv', a, fmt='%d', delimiter=',')
# 文件 / 待存数组 / 格式 / 分割字符串

# c = np.loadtxt('np_test_csv.csv', dtype=np.float, delimiter=',', unpack=False)
#    # 文件 / 保存类型 / 分割字符串 /
# print(c)

# 2.多维数组的存取 --维度,元素类型丢失
# a.tofile('np_tofile.dat', sep=',', format='%s')
# 文件 / 数据分隔符(空串表示写入二进制) / 格式

# d = np.fromfile('np_tofile.dat', dtype=float, count=-1, sep=',')
#    # 文件 / 数据类型 / 读入元素数量(-1表示整个文件) / 数据分隔符
# print(d)

# 3.numpy的便捷文件存取
# np.save('np_npy.npy', a)
# 文件名(.npy / .npz) / 数组
# np.savez('np.npz.npz', a)
# 同上

# e = np.load('np_npy.npy')
# print(e)
"""numpy的随机数函数"""

# np.random.seed(1)
# print("rand -> \n", np.random.rand(3,4,5))     # 浮点数[0,1),均匀分布,生成参数类型的数组
# print("randn -> \n", np.random.randn(3,4,5))   # 正态分布,生成参数类型的数组
# a = np.random.randint(100, 200, (3,4))
# print("randint -> \n", a)                  # 根据shape创建在[l,r)范围内的数组,均匀分布

# np.random.shuffle(a)
# print("shuffle-> \n", a)                   # shuffle根据第0轴随机排列,改变原数组
# print("-> \n", np.random.permutation(a))   # permutation根据第0轴随机排列,不改变原数组
# choice从一维数组选取一个size的数组,选择是否重复以及概率
# print("-> \n", np.random.choice(a.flatten(), (3,2), replace=False, p=a.flatten()/np.sum(a) ) )

# print("uniform->\n", np.random.uniform(0, 10, (3,4)))  # 均匀分布
# print("normal->\n", np.random.normal(10, 5, (3,4)))  # 正态分布(均值,标准差,size)
# print("poisson->\n", np.random.poisson(0.5, (3,4)))  # 泊松分布
"""numpy的统计函数"""

# a = np.arange(15).reshape(3,5)
# print(a)
# print("sum->", np.sum(a))              # 求和,可指定axis
# print("mean->", np.mean(a, axis=0))    # 求期望
# print("average->", np.average(a, axis=0, weights=[10,5,1]))     # 求均值
# print("std->", np.std(a))              # 求标准差
# print("var->", np.var(a))          # 求方差

# print("min->", np.min(a))          # 求最小值
# print("max->", np.max(a))          # 求最大值
# print("argmax->", np.argmax(a))    # 求最大值降一维后的下标
# print("argmin->", np.argmin(a))    # 求最小值降一维后的下标
# print("unravel_index->", np.unravel_index(np.argmax(a), a.shape))    # 重塑多维下标
# print("ptp->", np.ptp(a))          # 求最大值与最小值的差
# print("median->", np.median(a))    # 求中位数
"""numpy的梯度函数"""

# a = np.random.randint(1, 20, (5))
# print(a)
# print("gradient->\n", np.gradient(a))  # 求梯度
# b = np.random.randint(1, 50, (3, 5))
# print(b)
# print("gradient->\n", np.gradient(b))  # 求梯度

# 实例1:图像的手绘效果
from PIL import Image
"""图像的数组表示"""

# RGB色彩模式 (0 - 255)
# im = np.array(Image.open(r"ali1.jpg"))
# print("阿狸图片->\n", im.shape, im.dtype)  # 图象是一个三维数组
"""图像的变换"""

# a = np.array(Image.open(r"ali1.jpg"))
# b = [255,255,255] - a
# im1 = Image.fromarray(b.astype('uint8'))
# im1.save(r'ali2.jpg')

# a = np.array(Image.open(r"ali1.jpg").convert('L'))   # 灰度处理
# b = 255 - a
# im2 = Image.fromarray(b.astype('uint8'))
# im2.save(r'ali3.jpg')

# a = np.array(Image.open(r"ali1.jpg").convert('L'))   # 灰度处理
# b = (100/255)*a +150   # 区间变换
# im3 = Image.fromarray(b.astype('uint8'))
# im3.save(r'ali4.jpg')

# a = np.array(Image.open(r"ali1.jpg").convert('L'))   # 灰度处理
# b = 255 * (a/255)**2  # 像素平方
# im4 = Image.fromarray(b.astype('uint8'))
# im4.save(r'ali5.jpg')
"""图像的手绘效果"""

# a = np.asarray(Image.open('ali1.jpg').convert('L')).astype('float')
# depth = 10.  # (0-100)
# grad = np.gradient(a) #取图像灰度的梯度值
# grad_x, grad_y = grad  #分别取横纵图像梯度值
# grad_x = grad_x*depth/100.
# grad_y = grad_y*depth/100.
# A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
# uni_x = grad_x/A
# uni_y = grad_y/A
# uni_z = 1./A
# vec_el = np.pi/2.2  # 光源的俯视角度，弧度值
# vec_az = np.pi/4.  # 光源的方位角度，弧度值
# dx = np.cos(vec_el)*np.cos(vec_az)  #光源对x 轴的影响
# dy = np.cos(vec_el)*np.sin(vec_az)  #光源对y 轴的影响
# dz = np.sin(vec_el)  #光源对z 轴的影响
# b = 255*(dx*uni_x + dy*uni_y + dz*uni_z)  #光源归一化
# b = b.clip(0,255)
# im = Image.fromarray(b.astype('uint8'))  #重构图像
# im.save('ali_shouhui.jpg')
