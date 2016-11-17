import tensorflow as tf
# create a Variable
state = tf.Variable(0, name="counter")
# create an Op to add 1 to state
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# variables must be initialized by running an 'init' Op
# lauch the graph. Add 'init' Op to the Graph
init_op = tf.initialize_all_variables()

a = tf.constant([[1]])  # 1*2
b = tf.constant([[3, 4]]) # 1*2
mul = tf.matmul(a,b)

x = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]]
c = [[[1]],[[2]],[[3]]]
print(x)
y = tf.transpose(x)
z = tf.placeholder(tf.int32, [2,2,3])

ex = [[1,2],[4,5],[7,8]]
print(ex)
ex = tf.reshape(ex, [-1,2])
# Lauch the graph and run the Op
with tf.Session() as sess:
    # run 'init' Op
    #sess.run(init_op)
    # Print initial value of the 'state'
    #print(sess.run(state))
    # run the op that update 'state' and pr
    #for _ in range(3):
    #    sess.run(update)
    #    print(sess.run(state))
    print(tf.shape(ex))
#    print(sess.run(z, feed_dict={z:c}))
