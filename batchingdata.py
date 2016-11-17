import tensorflow as tf

x = tf.range(1, 10, name="x")

range_q = tf.train.range_input_producer(limit=5, shuffle=False)
slice_end = range_q.dequeue()

y = tf.slice(x, [0], [slice_end], name="y")

batched_data = tf.train.batch(
    tensors=[y],
    batch_size=5,
    dynamic_pad=True,
    name="y_batch"
)

res = tf.contrib.learn.run_n({"y":batched_data}, n=1, feed_dict=None)

print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])
