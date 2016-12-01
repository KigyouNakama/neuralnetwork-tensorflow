import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
EPOCHS = 10000
PRINT_STEP = 1000

data = np.array([[[1,2,3],[4,5,6],[13,14,15]],[[7,8,9],[10,11,12],[16,17,18]]])

target = np.array([[[1],[2],[3]],[[6],[7],[8]]])
print("data shape", data.shape)

x_ = tf.placeholder(tf.float32, [None, data.shape[1], data.shape[2]])
y_ = tf.placeholder(tf.float32, [None, target.shape[1], target.shape[2]])
print("done placeholder")

cell = rnn_cell.GRUCell(num_units=data.shape[2])
outputs, states = rnn.dynamic_rnn(cell, x_, dtype=tf.float32)
outputs = tf.reshape(outputs, [-1, data.shape[2]])
print("done define ouput")

W = tf.Variable(tf.random_normal([data.shape[2], 1]))
b = tf.Variable(tf.random_normal([1]))
print("done params")

y = tf.matmul(outputs, W) + b
y = tf.reshape(y, [2,3,1])
cost = tf.reduce_mean(tf.square(y - y_))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)
print("start")

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print("for begin")
    for i in range(EPOCHS):
        #print("foring")
        #print("shape of y", sess.run(tf.shape(y), feed_dict={x_: data, y_: target}))
        #print("shape of y_", sess.run(tf.shape(y_), feed_dict={x_: data, y_: target}))
        sess.run(train_op, feed_dict={x_: data, y_: target})
        #print("done sess.run")
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={x_: data, y_: target})
            print('training cost:', c)
    response = sess.run(y, feed_dict={x_: data})
    print(response)
