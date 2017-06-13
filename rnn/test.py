import inline as inline
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from nltk.tokenize import word_tokenize

# Global config variables
num_steps = 10 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1

def gen_data(size=1000000):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)

gen_batch(gen_data(), batch_size, num_steps)

# for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
#     training_loss = 0
#     training_state = np.zeros((batch_size, state_size))
#     if verbose:
#         print("\nEPOCH", idx)
#     for step, (X, Y) in enumerate(epoch):
#         training_loss_, training_state, _ = \
#             sess.run([
#                 total_loss,
#                 final_state,
#                 train_step],
#                 feed_dict={x: X, y: Y, init_state: training_state})
#         training_loss += training_loss_
#         if step % 100 == 0 and step > 0:
#             if verbose:
#                 print("Average loss at step", step,
#                       "for last 250 steps:", training_loss / 100)
#             training_losses.append(training_loss / 100)
#             training_loss = 0

some_list = [[3,4],[5,6]]
for counter, value in enumerate(some_list):
    print(counter, value)

print(some_list[0])

def createGenerator():
    mylist = range(3)
    # return [i*i for i in mylist]
    for i in mylist:
        yield i*i

def dup(n):
    for i in n:
        yield createGenerator()

mygenerator = createGenerator()
# print(mygenerator)
for j,i  in enumerate(mygenerator):
    print(j)
    print(i)

x = [[0, 0, 2],
     [1, 0, 1]]
print(x)
x_ = tf.one_hot(x, 3)
x__ =  tf.unstack(x_, axis=2)
s = tf.Session()
print(s.run(x_))
print(s.run(x__))

string = "   i love you     "
print(word_tokenize(string))
token = "-love"
if token.startswith('-') or token.endswith('-'):
    token.replace('-', '')
print(token)
d = dict()
d[token] = 0
if token in d:
    d[token] += 1

for w,c in d.items():
    print("%s\t%d\n"%(w,c))
with open('/home/linhdang/Desktop/test', 'w') as frDict, \
     open('/home/linhdang/Desktop/test1', 'w') as fr :
    frDict.write("linh")
    frDict.write("\n")
    print(string.strip())

    l = ["linh", "dang"]
    frDict.write(" ".join(l)+"\n")
    frDict.write("linhdang")
    fr.write("linh")

so = [1,2,3,4]
print(" ".join(str(x) for x in so))

print("shuffle")
list1 = [11,12,13,14]
list2 = [4,3,2,1]
# Given list1 and list2
list1_shuf = []
list2_shuf = []
index_shuf = list(range(len(list1)))
random.shuffle(index_shuf)
for i in index_shuf:
    list1_shuf.append(list1[i])
    list2_shuf.append(list2[i])
print(list1_shuf)
print(list2_shuf)