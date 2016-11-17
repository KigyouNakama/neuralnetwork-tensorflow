import tensorflow as tf

t = [[1,2,3],[2,3]]

with tf.Session() as sess:
    #print(sess.run(tf.shape(t)))
    print("mask is", sess.run(mask))
