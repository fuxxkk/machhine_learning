import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt

'''超参数'''
num_steps = 5
batch_size = 200
num_classes = 2
state_size = 4
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
init_state = tf.zeros([batch_size, state_size])
'''RNN输入'''
rnn_inputs = tf.one_hot(x, num_classes)

'''定义RNN cell,并添加到计算图'''
cell = tf.contrib.rnn.BasicRNNCell(num_units=state_size)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)

'''预测，损失，优化'''
with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [state_size, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    '''因为rnn_outputs是三维的，这里需要将其转成2维的，
       矩阵运算后再转换回来[batch_size, num_steps, num_classes]'''
logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, \
                    shape=[batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)

y_as_list = tf.unstack(y, num=num_steps, axis=1)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
total_loss = tf.reduce_mean(losses)
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
                if  step > 0:
                    if verbose:
                        print('第 {0} 步的平均损失 {1}'.format(step, training_loss / 100))
                    training_losses.append(training_loss / 100)
                    training_loss = 0
    return training_losses


training_losses = train_rnn(num_epochs=1, num_steps=num_steps, state_size=state_size)
print(training_losses[0])
plt.plot(training_losses)
plt.show()
