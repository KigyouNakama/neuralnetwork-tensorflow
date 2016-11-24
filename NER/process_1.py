import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
dir1 = "/home/dhbk/rnn/neuralnetwork-tensorflow/dataset/NonTag4type.tag"
dir2 = "/home/dhbk/rnn/neuralnetwork-tensorflow/dataset/SubDict_vc.txt"

# is it better to eliminate from data (!?)
weak_stop_token = [',',';','...','"','(',')','[',']','{','}','<','>','/','*','@',
                   '#','$','%','^','&','-','_','+','=','|','~','`','“','”','-DOCSTART-']

# end of one sequence. Is it better to eliminate from data (!?)
strong_stop_token = ['.','\n',':','?','!']

# word classification: 11 classes
named_entity = ['O', 'I-PER', 'I-LOC', 'I-TOUR', 'I-ORG', 'I-PRO',
                     'B-PER', 'B-LOC', 'B-TOUR', 'B-ORG', 'B-PRO']

class dataset:
    # return @dict[word: listValueOfFloatTypeVector]
    def makeDictFromDataset(self):
        dictionary = {}
        with open(dir2) as f:
            #    num = tf.cast(float(split[1]), tf.float32)
            #    print(sess.run(num))
            for line in f:
                split = line.rstrip().lstrip().split(" ")
                word = split.pop(0).lower()
                # map string type vector to float type vector
                split = list(map(float, split))
                dictionary[word] = split
        return dictionary

    # wasted resource
    def getEntity(self, entity):
        list = [0] * 11
        list[named_entity.index(entity)] = 1
        return list

    #
    def data(self, dict):
        inputSequence = [] # list of word in inputSequence
        outputSequence = [] # list of word in outputSequence
        inputBatch = [] # list of sequence in inputBatch
        outputBatch = [] # list of sequence in outputBatch

        with open(dir1) as f:
            for line in f:
                split = line.rstrip().lstrip().split(" ")
                if split[0] in strong_stop_token:
                    inputSequence.append(dict.get(split[0].lower()))
                    if len(inputSequence) != 0:
                        inputBatch.append(inputSequence)
                        outputBatch.append(outputSequence)
                    inputSequence = []
                    outputSequence = []
                else:
                    if len(split) == 3:
                        inputSequence.append(dict.get(split[0].lower()))
                        outputSequence.append(self.getEntity((split[2])))
                #    if (split[0] not in weak_stop_token) and (len(split) == 3):
                #        inputSequence.append(dict.get(split[0].lower()))
                #        outputSequence.append(self.getEntity((split[2])))

        # each word represented by float type vector
        return inputBatch, outputBatch

    def getData(self, dict):
        inputSequence = [] # list of word in inputSequence
        outputSequence = [] # list of word in outputSequence
        inputBatch = [] # list of sequence in inputBatch
        outputBatch = [] # list of sequence in outputBatch

        with open(dir1) as f:
            for line in f:
                split = line.rstrip().lstrip().split(" ")
                if len(split) == 3:
                    wordVector = dict.get(split[0].lower())
                    if wordVector == None:
                        wordVector = np.random.uniform(-1,1,200)
                        wordVector = list(wordVector)
                #    if len(wordVector) == 200:
                #        print(type(wordVector))
                    inputSequence.append(wordVector)
                    outputSequence.append(self.getEntity((split[2])))
                    if split[0] in strong_stop_token:
                        inputBatch.append(inputSequence)
                        outputBatch.append(outputSequence)
                        inputSequence = []
                        outputSequence = []
        # each word represented by float type vector
        return inputBatch, outputBatch

    def length(self, input):
        used = tf.sign(tf.reduce_max(tf.abs(input), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

class SequenceLabelling:
    def __init__(self, data, target, num_hidden = 200, num_layers = 1):
        self.data = data
        self.target = target # batch size * timesteps(input sequence length) * features (word vector length)
        self.num_hidden = num_hidden # bactch size * timesteps * output size (classes size)
        self.num_layers = num_layers
        self.prediction
        self.error
        self.optimize
        self.cost

    def prediction(self):
        print("start prediction")
        output, _ = rnn.dynamic_rnn(
            rnn_cell.GRUCell(self.num_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=179
        )
        print("built rnn graph")
        # softmax layer
        max_length = int(self.target.get_shape()[1]) # timesteps
        num_classes = int(self.target.get_shape()[2]) # output size
        # weight [num_hidden, output size] bias [output size]
        weight, bias = self.weight_and_bias(self.num_hidden, num_classes)
        # Flatten to apply same weights to all time steps
        # nhưng nếu tổng số phần tử không chia hết cho số các ẩn số thì sao?
        print("done weight and bias")
        output = tf.reshape(output, [-1, self.num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight)+bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction())
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(179, tf.float32)
        print("cost")
        return tf.reduce_mean(cross_entropy)

    def optimize(self):
        learning_rate = 0.0003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        print("optimize")
        return optimizer.minimize(self.cost())

    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction(), 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(179, tf.float32)
        return tf.reduce_mean(mistakes)
    #
    def weight_and_bias(self, in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def pr(self):
        var = tf.Variable(tf.truncated_normal([2,3,4],stddev = 0.1))
        print("linhlinhlinh")
        return var


process = dataset()
dict = process.makeDictFromDataset()
print("kich co tu dien", len(dict))
x, y = process.getData(dict)

# batch size x, y = 22151, 22151
print("do dai cua batch input", len(x))
print("do dai cua batch output", len(y))
# x, y max_length of sequence = 179
print("do dai lon nhat cua moi xau", max(len(z) for z in x))
print("do dai nho nhat cua moi xau", min(len(z) for z in x[:100]))


# check for length
def check_length():
    for string in x[:1000]:
        if len(string) != 179:
            print("la")
        for word in string:
            if len(word) != 200:
                print("la")

# padding for input
def padding(x,type):
    feature_length = 0
    if type == "in":
        feature_length = 200
    else: feature_length = 11
    for string in x:
        pad = []
        if len(string) < 179:
            for word in range(179 - len(string)):
                pad.append([0] * feature_length)
            string += pad
    return x


x_train = x[:10000]
y_train = y[:10000]
x_test = x[10000:11000]
y_test = y[10000:11000]


x_test = padding(x_test, "in")
y_test = padding(y_test, "out")

#x_test = tf.convert_to_tensor(x_test)
#y_test = tf.convert_to_tensor(y_test)

"""
data = tf.placeholder(tf.float32, [None, 179, 200])
print(type(x_test))

with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())

    print("shape of x_placeholder", sess.run(tf.shape(data), feed_dict={data:x_test}))
    #print("shape of x_test", sess.run(tf.shape(x_test)))
    #print("shape of y_test", sess.run(tf.shape(y_test)))

"""
print("start")

#print(len(x_train[110]))

data = tf.placeholder(tf.float32, [None, 179, 200])
target = tf.placeholder(tf.float32, [None, 179, 11])
model = SequenceLabelling(data, target)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for epoch in range(10):
    for index in range(10):
        x_feed = x_train[index*1000:(index+1)*1000]
        y_feed = y_train[index*1000:(index+1)*1000]
        x_feed = padding(x_feed, "in")
        y_feed = padding(y_feed, "out")
        #x_feed = tf.convert_to_tensor(x_feed)
        #y_feed = tf.convert_to_tensor(y_feed)
        print("shape of x_feed", sess.run(tf.shape(x_feed)))
        print("shape of y_feed", sess.run(tf.shape(y_feed)))
        sess.run(model.optimize())
        #sess.run(model.optimize(),
        #        feed_dict = {data: x_feed, target: y_feed})
    error = sess.run(model.error(),
                feed_dict = { data: x_test, target: y_test})
    print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
