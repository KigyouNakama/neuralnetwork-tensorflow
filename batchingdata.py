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

b = [[[1,2,3],[4,5]],[[6],[7,8],[9,10,11]]]
for x in b:
    for y in x:
        if len(y) < 3:
            p = [0]*(3-len(y))
            y += p
        print(y)