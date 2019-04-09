# -*- coding: UTF-8 -*-

# 获取手写数据。
# 28*28的图片对象。每个图片对象根据需求是否转化为长度为784的横向量
# 每个对象的标签为0-9的数字，one-hot编码成10维的向量
import numpy as np


# 数据加载器基类。派生出图片加载器和标签加载器
class Loader(object):
    # 初始化加载器。path: 数据文件路径。count: 文件中的样本个数
    def __init__(self, path, count):
        self.path = path
        self.count = count

    # 读取文件内容
    def get_file_content(self):
        print(self.path)
        f = open(self.path, 'rb')
        content = f.read()  # 读取字节流
        print(len(content))
        f.close()
        return content  # 字节数组

    # 将unsigned byte字符转换为整数。python3中bytes的每个分量读取就会变成int
    # def to_int(self, byte):
    #     return struct.unpack('B', byte)[0]


# 图像数据加载器
class ImageLoader(Loader):
    # 内部函数，从文件字节数组中获取第index个图像数据。文件中包含所有样本图片的数据。
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16  # 文件头16字节，后面每28*28个字节为一个图片数据
        picture = []
        for i in range(28):
            picture.append([])  # 图片添加一行像素
            for j in range(28):
                byte1 = content[start + i * 28 + j]
                picture[i].append(byte1)  # python3中本来就是int
                # picture[i].append(self.to_int(byte1))  # 添加一行的每一个像素
        return picture  # 图片为[[x,x,x..][x,x,x...][x,x,x...][x,x,x...]]的列表

    # 将图像数据转化为784的行向量形式
    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    # 加载数据文件，获得全部样本的输入向量。onerow表示是否将每张图片转化为行向量，to2表示是否转化为0,1矩阵
    def load(self, onerow=False):
        content = self.get_file_content()  # 获取文件字节数组
        data_set = []
        for index in range(self.count):  # 遍历每一个样本
            onepic = self.get_picture(content, index)  # 从样本数据集中获取第index个样本的图片数据，返回的是二维数组
            #print("[ImageLoader.get_picture]:"+onepic.__str__())
            if onerow:
                onepic = self.get_one_sample(onepic)  # 将图像转化为一维向量形式
            data_set.append(onepic)
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    # 加载数据文件，获得全部样本的标签向量
    def load(self, one_hot):
        content = self.get_file_content()  # 获取文件字节数组
        labels = []
        for index in range(self.count):  # 遍历每一个样本
            onelabel = content[index + 8]  # 文件头有8个字节
            if one_hot:
                onelabel = self.norm(onelabel)  # one-hot编码
            labels.append(onelabel)
        return labels

    # 内部函数，one-hot编码。将一个值转换为10维标签向量
    def norm(self, label):
        label_vec = []
        # label_value = self.to_int(label)
        label_value = label  # python3中直接就是int
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec


# 获得训练数据集。onerow表示是否将每张图片转化为行向量,one_hot代表编码
def get_training_data_set(num, onerow=False, one_hot=True): # ①什么是独热编码（One-Hot）
    image_loader = ImageLoader('train-images.idx3-ubyte', num)  # 参数为文件路径和加载的样本数量
    label_loader = LabelLoader('train-labels.idx1-ubyte', num)  # 参数为文件路径和加载的样本数量
    return image_loader.load(onerow), label_loader.load(one_hot)


# 获得测试数据集。onerow表示是否将每张图片转化为行向量
def get_test_data_set(num, onerow=False, one_hot=True):
    image_loader = ImageLoader('t10k-images.idx3-ubyte', num)  # 参数为文件路径和加载的样本数量
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', num)  # 参数为文件路径和加载的样本数量
    return image_loader.load(onerow), label_loader.load(one_hot)


# 将一行784的行向量，打印成图形的样式
def printimg(onepic):
    onepic = onepic.reshape(28, 28) #给数组一个新的形状而不改变其数据
    for i in range(28):
        for j in range(28):
            if onepic[i, j] == 0:
                print('  ', end='')
            else:
                print('* ', end='')
        print('')


if __name__ == "__main__":                                
    train_data_set, train_labels = get_training_data_set(100)  # 加载训练样本数据集，和one-hot编码后的样本标签数据集
    train_data_set = np.array(train_data_set)  # .astype(bool).astype(int)    #可以将图片简化为黑白图片
    train_labels = np.array(train_labels)

    for i in range(10):
        onepic = train_data_set[i]  # 取一个样本
        printimg(onepic)  # 打印出来这一行所显示的图片
        print(train_labels[i].argmax())  # 打印样本标签
        print("-"*100)




'''
①什么是独热编码（One-Hot）？

————————————————————————————————————————

One-Hot编码，又称为一位有效编码，主要是采用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。

One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数的索引之外，它都是零值，它被标记为1。

 

One-Hot实际案例

————————————————————————————————————————

就拿上面的例子来说吧，性别特征：["男","女"]，按照N位状态寄存器来对N个状态进行编码的原理，咱们处理后应该是这样的（这里只有两个特征，所以N=2）：

男  =>  10

女  =>  01

祖国特征：["中国"，"美国，"法国"]（这里N=3）：

中国  =>  100

美国  =>  010

法国  =>  001

运动特征：["足球"，"篮球"，"羽毛球"，"乒乓球"]（这里N=4）：

足球  =>  1000

篮球  =>  0100

羽毛球  =>  0010

乒乓球  =>  0001

所以，当一个样本为["男","中国","乒乓球"]的时候，完整的特征数字化的结果为：

[1，0，1，0，0，0，0，0，1]
'''
