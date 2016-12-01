import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])
# Reshape `a` as a vector. -1 means "set this dimension automatically".
a_as_vector = tf.reshape(a, [-1])

# Create another vector containing zeroes to pad `a` to (2 * 3) elements.
zero_padding = tf.zeros([2 * 3] - tf.shape(a_as_vector), dtype=a.dtype)

# Concatenate `a_as_vector` with the padding.
a_padded = tf.concat(0, [a_as_vector, zero_padding])

# Reshape the padded vector to the desired shape.
result = tf.reshape(a_padded, [2, 3])

with tf.Session() as sess:
    print("a as vector", sess.run(tf.shape(a_as_vector)))
    print("pad zero", sess.run(zero_padding))
    print("a_padded", sess.run(a_padded))
    print("ket qua", sess.run(result))

b = [[[1,2,3],[4,5,3]],[[6,2,2],[2,7,8],[9,10,11]]]
for x in b:
    pad = []
    if len(x) < 3:
        for word in range(3-len(x)):
            pad.append([0]*3)
            print("pad: ", pad)
        x+=pad
        print("x", x)
print("b", b)

inputSequence = [] # list of word in inputSequence
outputSequence = [] # list of word in outputSequence
inputBatch = [] # list of sequence in inputBatch
outputBatch = [] # list of sequence in outputBatch
for i in range(6):
    inputSequence.append([1]*3)
    print("inputSe", inputSequence)
    if i % 2 == 1:
        inputBatch.append(inputSequence)
        print("inputBa", inputBatch)
        inputSequence = []
print(inputBatch)
with tf.Session() as sess:
    print(sess.run(tf.shape(inputBatch)))

x = [[[1,1,1],[1,1,1],[1,1,1]]]
data = tf.placeholder(tf.float32, [None, 3, 3])
with tf.Session() as sess:
    print("shape of data", sess.run(tf.shape(data), feed_dict={data: x}))
    print(sess.run(tf.shape(x)))
    print(type(sess.run(data[0][0][0], feed_dict={data: x})))

x = list(x)
print("x = ", x)
x = x[0]
print("x[-1] = ", x)
y = [[1.,1.,0.],[1.,1.,1.]],[[1.,1.,1.],[1.,1.,1.]]]
z = [[[1.,1.,0.],[1.,1.,1.]],[[1.,1.,1.],[1.,1.,1.]]]
tf.sof
result = tf.matmul(y,z)
print(tf.Session().run(result))