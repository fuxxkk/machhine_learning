# rnn 神经网络demo
import copy, numpy as np

np.random.seed(0)


# sigmoid函数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# sigmoid导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# 训练数据生成
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# 初始化一些变量
alpha = 0.1  # 学习率
input_dim = 2  # 输入的大小
hidden_dim = 8  # 隐含层的大小
output_dim = 1  # 输出层的大小

# 随机初始化权重
synapse_0 = 2 * np.random.random((hidden_dim, input_dim)) - 1  # (8, 2)
synapse_1 = 2 * np.random.random((output_dim, hidden_dim)) - 1  # (1, 8)
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1  # (8, 8)

synapse_0_update = np.zeros_like(synapse_0)  # (8, 2)
synapse_1_update = np.zeros_like(synapse_1)  # (1, 8)
synapse_h_update = np.zeros_like(synapse_h)  # (8, 8)

# 开始训练
for j in range(100000):

    # 二进制相加
    a_int = np.random.randint(largest_number / 2)  # 随机生成相加的数
    a = int2binary[a_int]  # 映射成二进制值

    b_int = np.random.randint(largest_number / 2)  # 随机生成相加的数
    b = int2binary[b_int]  # 映射成二进制值

    # 真实的答案
    c_int = a_int + b_int  # 结果
    c = int2binary[c_int]  # 映射成二进制值

    # 待存放预测值
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()  # 输出层的误差
    layer_2_values = list()  # 第二层的值（输出的结果）
    layer_1_values = list()  # 第一层的值（隐含状态）
    layer_1_values.append(copy.deepcopy(np.zeros((hidden_dim, 1))))  # 第一个隐含状态需要0作为它的上一个隐含状态

    # 前向传播
    for i in range(binary_dim):
        X = np.array([[a[binary_dim - i - 1], b[binary_dim - i - 1]]]).T  # (2,1)
        y = np.array([[c[binary_dim - i - 1]]]).T  # (1,1)
        layer_1 = sigmoid(np.dot(synapse_h, layer_1_values[-1]) + np.dot(synapse_0, X))  # (1,1)
        layer_1_values.append(copy.deepcopy(layer_1))  # (8,1)
        layer_2 = sigmoid(np.dot(synapse_1, layer_1))  # (1,1)
        error = -(y - layer_2)  # 使用平方差作为损失函数
        layer_2_error = y - layer_2
        layer_delta2 = error * sigmoid_output_to_derivative(layer_2)  # (1,1)
        layer_2_deltas.append(copy.deepcopy(layer_delta2))
        overallError +=np.abs(error)
        d[binary_dim - i - 1] = np.round(layer_2[0][0])
    future_layer_1_delta = np.zeros((hidden_dim, 1))
    # 反向传播
    for i in range(binary_dim):
        X = np.array([[a[i], b[i]]]).T
        prev_layer_1 = layer_1_values[-i - 2]
        layer_1 = layer_1_values[-i - 1]
        layer_delta2 = layer_2_deltas[-i - 1]
        layer_delta1 = np.multiply(np.add(np.dot(synapse_h.T, future_layer_1_delta), np.dot(synapse_1.T, layer_delta2)),
                                   sigmoid_output_to_derivative(layer_1))
        synapse_0_update += np.dot(layer_delta1, X.T)
        synapse_h_update += np.dot(layer_delta1, prev_layer_1.T)
        synapse_1_update += np.dot(layer_delta2, layer_1.T)
        future_layer_1_delta = layer_delta1
    synapse_0 -= alpha * synapse_0_update
    synapse_h -= alpha * synapse_h_update
    synapse_1 -= alpha * synapse_1_update
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    # 验证结果
    if (j % 100 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
