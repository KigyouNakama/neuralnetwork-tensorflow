import csv, gzip, json
import re
import os, random, tensorflow as tf

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from numpy import array

csv.field_size_limit(500 * 1024 * 1024)

commonPath = '/home/linhdang/rnn/data/'
pEncodeData = os.path.abspath(commonPath+'encode_data')
pEncodeLabel = os.path.abspath(commonPath+'encode_label')

def loadTrainAndTest():
    data = []
    label = []
    count = 0
    MAX_POS = 110000
    MIN_NEG = 4080362
    MIN_NEU = 4416906
    TOTAL = 4437223
    MIDDLE = 20000
    END = 21000
    with open(pEncodeData, 'r') as fData, open(pEncodeLabel, 'r') as fLabel:
        for line in fData:
            count += 1
            if (count <= MAX_POS) or \
                    (count > MIN_NEG and count <= (MIN_NEG + MAX_POS)) or \
                    (count > MIN_NEU and count <= TOTAL):
                data.append(line)
        print(data.__len__())
        count = 0
        for line in fLabel:
            count += 1
            if (count <= MAX_POS) or \
                    (count > MIN_NEG and count <= (MIN_NEG + MAX_POS)) or \
                    (count > MIN_NEU and count <= TOTAL):
                label.append(line)
        print(label.__len__())
    # shuffle two lists
    data_shuf = []
    label_shuf = []
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    for i in index_shuf:
        data_shuf.append(data[i])
        label_shuf.append(label[i])
    # 240317
    return data[:MIDDLE], label[:MIDDLE], data[MIDDLE:END], label[MIDDLE:END]

class PaddedDataIterator():
    def __init__(self, data):
        self.data = data
        self.max_len = 50
        self.batch_size = 10
        self.vocab_size = 67259
        self.num_of_classes = 3

    def next_batch_data(self, idx):
        padded_data = []
        for line in self.data[idx*self.batch_size:(idx+1)*self.batch_size]:
            tokens = line.split(" ")
            tokens = [int(x) for x in tokens]
            if len(tokens) < self.max_len:
                pad = []
                pad += [0] * (self.max_len - len(tokens))
                tokens += pad
            padded_data.append(tokens)
        # print(array(padded_data).shape)
        # padded_data = tf.one_hot(padded_data, self.vocab_size)
        # print(padded_data.shape)
        return padded_data

    def next_batch_label(self, idx):
        label = []
        for line in self.data[idx*self.batch_size:(idx+1)*self.batch_size]:
            label.append(int(line))
        # label = tf.one_hot(label, self.num_of_classes)
        # print(label.shape[1])
        return label

train_data, train_label, test_data, test_label = loadTrainAndTest()

tr_data = PaddedDataIterator(train_data)
tr_label = PaddedDataIterator(train_label)
te_label = PaddedDataIterator(test_label)
te_data = PaddedDataIterator(test_data)

# print(tr_data.next_batch_data(0))
# print(tr_label.next_batch_label(0))

num_steps = 50
num_features = 67259
num_classes = 3
state_size = 50
# build placeholder
data_holder = tf.placeholder(tf.int32, [None, num_steps])
label_holder = tf.placeholder(tf.int32, [None])

rnn_ip = tf.one_hot(data_holder, num_features)
rnn_op = tf.one_hot(label_holder, num_classes)
rnn_inputs = tf.cast(rnn_ip, tf.float32)
rnn_outputs = tf.cast(rnn_op, tf.float32)

# build graph
cell = tf.contrib.rnn.LSTMCell(num_units=state_size)
output, states = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)
# outputs = output[-1]
outputs = output
# print(outputs.get_shape())
outputs = tf.transpose(outputs, [1, 0, 2])
# print(outputs.get_shape())
last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
print(last.get_shape())

W = tf.Variable(tf.random_normal([state_size, num_classes]))
b = tf.Variable(tf.random_normal([num_classes]))

# y = tf.matmul(tf.reshape(last, [-1, state_size]), W) + b
y = tf.matmul(last, W) + b

cost = tf.reduce_mean(tf.square(y - rnn_outputs))
train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

EPOCHS = 2
PRINT_STEP = 1
BATCH_SIZE = 10
NUM_OF_BATCHES = int(len(train_data)/BATCH_SIZE) # batch_size
print(NUM_OF_BATCHES)
print("start training")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHS):
        print("EPOCHS %d"%(i+1))
        for j in range(NUM_OF_BATCHES):
            batch_data = tr_data.next_batch_data(j)
            batch_label = tr_label.next_batch_label(j)
            sess.run(train_op, feed_dict={data_holder: batch_data, \
                                          label_holder: batch_label})
        if i % PRINT_STEP == 0:
            c = sess.run(cost, feed_dict={data_holder: batch_data, \
                                          label_holder: batch_label})
            print('training cost:', c * 100)

#
# def padd_test():
#     tokens = [1,2,3,4]
#     if len(tokens) < 10:
#         pad = []
#         pad += [0] * (10 - len(tokens))
#     tokens += pad
#     print(tokens)
#     tokens.append(int("4"))
#     print(tokens)
# padd_test()