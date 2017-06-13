import csv, gzip, json
import sys
import os, random, tensorflow as tf
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
from numpy import array
from pyasn1.type.char import TeletexString

csv.field_size_limit(500 * 1024 * 1024)

commonPath = '/home/linhdang/rnn/data/'
pEncodeData = os.path.abspath(commonPath+'encode_data')
pEncodeLabel = os.path.abspath(commonPath+'encode_label')
pTrainingCost = os.path.abspath(commonPath+'training_cost')
pTestingCost = os.path.abspath(commonPath+'tesing_cost')

def loadTrTe():
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    data_positive = []
    data_negative = []
    data_neural = []
    label_positive = []
    label_neural = []
    label_negative = []
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
            if (count <= MAX_POS):
                data_positive.append(line.strip())
            elif (count > MIN_NEG) and (count <= (MIN_NEG + MAX_POS)):
                data_negative.append(line.strip())
            elif (count > MIN_NEU) and (count <= TOTAL):
                data_neural.append(line.strip())
        count = 0
        for line in fLabel:
            count += 1
            if (count <= MAX_POS):
                label_positive.append(line.strip())
            elif (count > MIN_NEG) and (count <= (MIN_NEG + MAX_POS)):
                label_negative.append(line.strip())
            elif (count > MIN_NEU) and (count <= TOTAL):
                label_neural.append(line.strip())

    print("positive length", len(data_positive), len(label_positive))
    print("negative length", len(data_negative), len(label_negative))
    print("neural length", len(data_neural), len(label_neural))
    TRAIN_LIMIT = 20000
    TEST_LIMIT = 2000
    count = 0
    for i in range(len(data_positive)):
        if i >= (TRAIN_LIMIT + TEST_LIMIT):
            break
        if i < TRAIN_LIMIT:
            train_data.append(data_positive[i])
            train_label.append(label_positive[i])
            train_data.append(data_negative[i])
            train_label.append(label_negative[i])
        else:
            test_data.append(data_positive[i])
            test_label.append(label_positive[i])
            test_data.append(data_negative[i])
            test_label.append(label_negative[i])
            if count < len(data_neural):
                test_data.append(data_neural[count])
                test_label.append(label_neural[count])
                count+=1
        if i < 17000:
            train_data.append(data_neural[i])
            train_label.append(label_neural[i])
            count = i+1
    print("traindata length", len(train_data))
    print("trainlabel length", len(train_label))
    print("testdata length", len(test_data))
    print("testlabel length", len(test_data))
    # 240317
    return train_data, train_label, test_data, test_label

class PaddedDataIterator():
    def __init__(self, data):
        self.data = data
        self.max_len = 50
        self.batch_size = 100
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

train_data, train_label, test_data, test_label = loadTrTe()
# for i in range(19000, 19010):
#     print(train_label[i])
#     # print(test_label[i])
# sys.exit()
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
EVAL_STEP = 1
BATCH_SIZE = 100
NUM_OF_BATCHES_DATA = int(len(train_data)/BATCH_SIZE) # batch_size
NUM_OF_BATCHES_LABEL = int(len(test_data)/BATCH_SIZE)

print(NUM_OF_BATCHES_DATA)
print("start training")
training_cost = []
testing_cost = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(EPOCHS):
        print("EPOCHS %d"%(i+1))
        step, final_cost = 0, 0
        for j in range(NUM_OF_BATCHES_DATA):
            batch_data = tr_data.next_batch_data(j)
            batch_label = tr_label.next_batch_label(j)
            sess.run(train_op, feed_dict={data_holder: batch_data, \
                                          label_holder: batch_label})
            train_cost = sess.run(cost, feed_dict={data_holder: batch_data, \
                                                   label_holder: batch_label})
            final_cost+=train_cost
            step+=1
            print(step)

        print("training cost ",final_cost/step)
        training_cost.append(final_cost/step)

        if i % EVAL_STEP == 0:
            step, final_cost = 0, 0
            for j in range(NUM_OF_BATCHES_LABEL):
                test_cost = sess.run(cost, feed_dict={data_holder: te_data.next_batch_data(j), \
                                          label_holder: te_label.next_batch_label(j)})
                final_cost += train_cost
                step += 1
                # print(step)
            print('testing cost:', final_cost/step)
            testing_cost.append(final_cost/step)

with open(pTrainingCost, 'w') as fTrCost:
    for i in training_cost:
        fTrCost.write("%f\n"%i)
with open(pTestingCost, 'w') as fTeCost:
    for i in testing_cost:
        fTeCost.write("%f\n"%i)

plt.plot(training_cost)
plt.show()
plt.plot(test_cost)
plt.show()

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
"""
training cost  0.0212502378778
testing cost: 0.00919519457966
"""