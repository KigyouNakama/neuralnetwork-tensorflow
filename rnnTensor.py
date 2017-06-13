import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn
EPOCHS = 10000
PRINT_STEP = 1000

data = np.array([[[1, 2, 3, 4, 5],
                 [2, 3, 4, 5, 6],
                 [3, 4, 5, 6, 7]]])
target = np.array([[[6],
                   [7],
                   [8]]])
# data shape (1, 3, 5)
# target shape (1, 3, 1)
print("data shape", data.shape)
print("target shape", target.shape)

x_ = tf.placeholder(tf.float32, [None, data.shape[1], data.shape[2]])
y_ = tf.placeholder(tf.float32, [None, target.shape[1], target.shape[2]])
print("x_ shape", x_.get_shape())

cell = tf.contrib.rnn.BasicLSTMCell(num_units=10)
output, states = tf.nn.dynamic_rnn(cell, x_, dtype=tf.float32)
# outputs = output[-1]
outputs = output
print(outputs.get_shape())
W = tf.Variable(tf.random_normal([10, 1]))
b = tf.Variable(tf.random_normal([1]))

y = tf.matmul(tf.reshape(outputs, [-1, 10]), W) + b

cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHS):
        sess.run(train_op, feed_dict={x_: data, y_: target})
        if i % PRINT_STEP == 0:
            print("shape outputs", sess.run(tf.shape(output), feed_dict={x_: data, y_: target}))
            print(sess.run(output, feed_dict={x_: data, y_: target}))
        #    print(sess.run(y,  feed_dict={x_: data, y_: target}))
            c = sess.run(cost, feed_dict={x_: data, y_: target})
            print('training cost:', c * 100)
            response = sess.run(y, feed_dict={x_: data})
            print(response)

    # response = sess.run(y, feed_dict={x_: data})
    # print(response)
