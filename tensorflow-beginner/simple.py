import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

temp = tf.placeholder(tf.float32, [10,1])
cake_doneness = tf.placeholder(tf.float32, [10,1])
temp_m = tf.get_variable('temp_m', [1,5])
temp_b = tf.get_variable('temp_b', [5])
predicted_output = tf.nn.xw_plus_b(temp, temp_m, temp_b)

temp_m1 = tf.get_variable('temp_m1', [5,1])
temp_b1 = tf.get_variable('temp_b1', [1])
predicted_output1 = tf.nn.xw_plus_b(predicted_output, temp_m1, temp_b1)

cost = tf.reduce_mean((cake_doneness-predicted_output1)**2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
temp_train = np.linspace(0,10,10).reshape(-1,1)
print("temp_train = ", temp_train)
doneness_train = temp_train*5. + 1. + np.random.randn(10,1)
print('doneness_train = ', doneness_train)
plt.plot(doneness_train, temp_train)
c = []

for i in range(10000):
    sess.run(optimizer, feed_dict={temp: temp_train,
                                   cake_doneness: doneness_train})
    predicted_doneness = sess.run(predicted_output, feed_dict={temp:temp_train})
    if (i + 1) % 1000 == 0:
        c.append(sess.run(cost, feed_dict={temp: temp_train,
                                   cake_doneness: doneness_train}))
        if (i+1) == 10000:
            print(c)
    if i == 9999:
        print("epoch: {:d}, predict: {}".format(i + 1, predicted_doneness))
        print('temp_m', sess.run(temp_m))
        print('temp_b', sess.run(temp_b))
        plt.plot(predicted_doneness, temp_train)
        #plt.plot(c, temp_train)
        plt.show()
