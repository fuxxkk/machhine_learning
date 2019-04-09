import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
import cnn.MNIST as MNIST  # MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片

# 全局变量
batch_size = 128  # 批处理样本数量
nb_classes = 10  # 分类数目,0-9
epochs = 600  # 迭代次数
img_rows, img_cols = 28, 28  # 输入图片样本的宽高
nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (img_rows, img_cols, 1)  # 输入图片的维度

X_train, Y_train = MNIST.get_training_data_set(600, False)  # 加载训练样本数据集，和one-hot编码后的样本标签数据集。最大60000
X_test, Y_test = MNIST.get_test_data_set(100, False)  # 加载测试特征数据集，和one-hot编码后的测试标签数据集，最大10000
X_train = np.array(X_train).astype(bool).astype(float) / 255  # 数据归一化 (array将tuple和list, array, 或者其他的序列模式的数据转创建为 ndarray, 默认创建一个新的 ndarray. astype是实现变量类型转换)
X_train = X_train[:, :, :, np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
#print("X_train[:, :, :, np.newaxis]"+str(X_train[0]))
Y_train = np.array(Y_train)
X_test = np.array(X_test).astype(bool).astype(float) / 255  # 数据归一化
X_test = X_test[:, :, :, np.newaxis]  # 添加一个维度，代表图片通道。这样数据集共4个维度，样本个数、宽度、高度、通道数
Y_test = np.array(Y_test)
print('样本数据集的维度：', X_train.shape, Y_train.shape)
print('测试数据集的维度：', X_test.shape, Y_test.shape)
print(MNIST.printimg(X_train[1]))
print("Y_train[1]"+str(Y_train[1]))
# 构建模型
model = Sequential()  #创建 ①序贯模型
model.add(Conv2D(6, kernel_size, input_shape=input_shape, strides=1))  # 卷积层1
model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层
model.add(Conv2D(12, kernel_size, strides=1))  # 卷积层2
model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层
model.add(Flatten())  # 拉成一维数据
model.add(Dense(128))  # 全连接层1
model.add(Dense(nb_classes))  # 全连接层2
model.add(Activation('sigmoid'))  # sigmoid评分

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# 训练模型
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# verbose = 2 为每个epoch输出一行记录
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(X_test, Y_test))
# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


'''
①序贯模型
Sequential模型字面上的翻译是顺序模型，给人的第一感觉是那种简单的线性模型，但实际上Sequential模型可以构建非常复杂的神经网络，
包括全连接神经网络、卷积神经网络(CNN)、循环神经网络(RNN)、等等。这里的Sequential更准确的应该理解为堆叠，通过堆叠许多层，构建出深度神经网络。


'''