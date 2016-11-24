import tensorflow as tf

learning_rate = 0.01
x = tf.Variable(tf.random_normal([1000,2],stddev = 0.1))
w = [[0.4], [0.6]]


y = tf.matmul(x, w)

#x_train = tf.slice(x, [0,0], [800,2])
x_train = x[:800]
y_real = y[0:800]

w_learn = tf.Variable(tf.truncated_normal([2,1],stddev = 0.1))

y_train = tf.matmul(x_train, w_learn)

cost = tf.reduce_sum(tf.pow(y_real-y_train, 2)/2)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()
# launch the graph
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(w_learn))
    for i in range(1000):
        sess.run(optimizer)
    print(sess.run(w_learn))
    #sess.run(optimizer)
