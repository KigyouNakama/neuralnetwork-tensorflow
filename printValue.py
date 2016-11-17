import tensorflow as tf

#define a variable to hold normal random values
normal_rv = tf.Variable(tf.truncated_normal([2,3,4],stddev = 0.1))
#split0, split1, split2 = tf.split(0,3,normal_rv)
res = tf.reshape(normal_rv, [-1,6])

# exam = tf.Variable(tf.truncated_normal([2,2,3],stddev = 0.1))
exam1 = [[1,2,3],[4,5,6]]
exam2 = [[[1,2,3],[4,5,6],[13,14,15]],[[7,8,9],[10,11,12],[16,17,18]]]
test = [[[1,2,3],[4,5,6],[13,14,15],[16,17,18]],[[7,8,9],[10,11,12],[16,17,18],[16,17,18]]]
tran1 = tf.transpose(exam1, [0,1])
tran2 = tf.transpose(exam2, [1,0,2])

indice = 2
gather = tf.gather(tran2, indice)

exam3 = [2,2,2]
exam4 = [[2,2,2],[2,2,2]]

plus = tf.add(exam1, exam3)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    print(normal_rv.get_shape())
    sess.run(init_op)
    print(sess.run(normal_rv))
    print(sess.run(res))
    print(res.get_shape())
    print(sess.run(tf.shape(exam2)[0]))
    print(exam2)
    print("shape of trans2 ")
    print(sess.run(tf.shape(tran2)))
    print(sess.run(tran2))
    print(sess.run(gather))
    print(sess.run(plus))
    print("shape of exam3:", sess.run(tf.shape(exam3)))
    print("", sess.run(tf.sign(tf.reduce_max(tf.abs(test), reduction_indices=2))))
    print(sess.run(length(exam2)))
    #print the random values that we sample
"""    print(sess.run(normal_rv))
    print(sess.run(tf.shape(normal_rv)))
#    print(sess.run(tf.shape(split0)))
    print(sess.run(tf.shape(res)))
    print(sess.run(res))
"""

