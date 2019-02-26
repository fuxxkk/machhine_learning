import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

'''超参数'''
num_steps = 8
batch_size = 200
num_classes = 2
state_size = 16
learning_rate = 0.1


def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    '''根据规则生成Y'''
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


'''生成batch数据'''


def gen_batch(raw_data, batch_size, num_step):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)
    batch_patition_length = data_length // batch_size  # ->5000
    data_x = np.zeros([batch_size, batch_patition_length], dtype=np.int32)  # ->(200, 5000)
    data_y = np.zeros([batch_size, batch_patition_length], dtype=np.int32)  # ->(200, 5000)
    '''填到矩阵的对应位置'''
    for i in range(batch_size):
        data_x[i] = raw_x[
                    batch_patition_length * i:batch_patition_length * (i + 1)]  # 每一行取batch_patition_length个数，即5000
        data_y[i] = raw_y[batch_patition_length * i:batch_patition_length * (i + 1)]
    epoch_size = batch_patition_length // num_steps  # ->5000/5=1000 就是每一轮的大小
    for i in range(epoch_size):  # 抽取 epoch_size 个数据
        x = data_x[:, i * num_steps:(i + 1) * num_steps]  # ->(200, 5)
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)  # yield 是生成器，生成器函数在生成值后会自动挂起并暂停他们的执行和状态（最后就是for循环结束后的结果，共有1000个(x, y)）


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


'''定义placeholder'''
x = tf.placeholder(tf.int32, [batch_size, num_steps], name="x")
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='y')
init_state = tf.zeros([batch_size, state_size]) #创建一个所有元素都设置为零的矩阵.
'''RNN输入'''
x_one_hot = tf.one_hot(x, num_classes)  #将input转化为one-hot类型数据输出，相当于将多个数值联合放在一起作为多个相同类型的向量,第一个参数表示输入的多个数值，通常是矩阵形式；第二个参数表示输出的尺寸。
rnn_inputs = tf.unstack(x_one_hot, axis=1) #分解矩阵

'''定义RNN cell  ###（关键步骤）### '''
with tf.variable_scope('rnn_cell'):  #绝大部分情况下会和tf.get_variable()配合使用，实现变量共享的功能。
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))


def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b) #tf.matmul（）将矩阵a乘以矩阵b，生成a * b


'''将rnn cell添加到计算图中'''
state = init_state
rnn_outputs = []
for rnn_input in rnn_inputs:
    state = rnn_cell(rnn_input, state)  # state会重复使用，循环
    rnn_outputs.append(state)
final_state = rnn_outputs[-1]  # 得到最后的state

'''预测，损失，优化'''
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]   #预测  (通过Softmax回归，将logistic的预测二分类的概率的问题推广到了n分类的概率的问题。)

y_as_list = tf.unstack(y, num=num_steps, axis=1)
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in  #计算logits 和 labels 之间的稀疏softmax 交叉熵
          zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses) #函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

'''训练网络'''


def train_rnn(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))  # ->(200, 4)
            if verbose:
                print('\nepoch', idx)
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _ = \
                    sess.run([losses, total_loss, final_state, train_step],
                             feed_dict={x: X, y: Y, init_state: training_state})
                training_loss += training_loss_
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print('第 {0} 步的平均损失 {1}'.format(step, training_loss / 100))
                    training_losses.append(training_loss / 100)
                    training_loss = 0
    return training_losses


training_losses = train_rnn(num_epochs=1, num_steps=num_steps, state_size=state_size)
print(training_losses[0])
plt.plot(training_losses)
plt.show()
