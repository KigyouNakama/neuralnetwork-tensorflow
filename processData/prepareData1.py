import csv, gzip, json
import re
import os, random

from nltk.tokenize import word_tokenize
from nltk import PorterStemmer
csv.field_size_limit(500 * 1024 * 1024)

class BuildDataset:
    commonPath = '/home/linhdang/rnn/data/'
    pAmazon = os.path.abspath(commonPath+'item_dedup.json.gz')
    pShortText = os.path.abspath(commonPath+'dataset.csv')
    pPositive = os.path.abspath(commonPath+'positive')
    pNegative = os.path.abspath(commonPath+'negative')
    pNeural = os.path.abspath(commonPath+'neural')

    pDictPos = os.path.abspath(commonPath+'dictPos')
    pDictNeg = os.path.abspath(commonPath+'dictNeg')
    pDictNeu = os.path.abspath(commonPath+'dictNeu')
    pDict = os.path.abspath(commonPath+'dict')

    pEncodeData = os.path.abspath(commonPath+'encode_data')
    pEncodeLabel = os.path.abspath(commonPath+'encode_label')

    porter = PorterStemmer()

    def __init__(self):
        print('start')

    def preprocess(self):
        readAmazon = gzip.open(self.pAmazon, 'rt')
        fPositive = open(self.pPositive, 'w')
        fNegative = open(self.pNegative, 'w')
        fNeural = open(self.pNeural, 'w')

        countShortText = neutralCountST = negativeCountST = positiveCountST = 0
        for line in readAmazon:
            # load line
            json_data = json.loads(line)
            review = json_data['reviewText']
            # process line
            review = re.sub("[^A-Za-z-' ]+", " ", review)
            review = ' '.join(review.split()).lower()
            # count word
            words_list = word_tokenize(review)
            length = len(words_list)
            if (length <= 50) and (length >= 1):
                countShortText += 1
                if json_data['overall'] < 3.0:
                    negativeCountST += 1
                    fNegative.write("\n"+review)
                elif json_data['overall'] > 3.0:
                    positiveCountST += 1
                    fPositive.write("\n"+review)
                else:
                    neutralCountST += 1
                    fNeural.write("\n"+review)
        print("shortText: %d\n negative: %d\n positive: %d\n neutral: %d\n"%(countShortText, negativeCountST, positiveCountST, neutralCountST))
        readAmazon.close()
        fNegative.close()
        fPositive.close()
        fNeural.close()

    def build_dictionary(self):
        frDict = open(self.pDict, 'w')
        files = [self.pPositive, self.pNegative, self.pNeural]
        dict_stats = dict()
        for file in files:
            with open(file, 'r') as f:
                count_line = 0
                for line in f:
                    count_line += 1
                    tokens = word_tokenize(line)
                    for token in tokens:
                        token = re.sub('\-+', '-', token)
                        if token.startswith('-') or token.endswith('-'):
                            token = token.replace('-', '')
                        if token in dict_stats:
                            dict_stats[token] += 1
                        else:
                            dict_stats[token] = 1
                print("dict_size %d"%dict_stats.__len__())
                print("line count in positive: %d"%count_line)

        for w, c in dict_stats.items():
            frDict.write("%s\t%d\n" % (w, c))
        frDict.close()

    def encode_data(self):
        word_dict = dict() # word_position
        sentiment_words = set()
        # stop_words = set()
        total_words = 0
        MIN_SENT_LENGTH = 4
        sentence_count = 0

        # create positive and negative list word
        with open('/home/linhdang/rnn/SentimentAnalysis/positive-words.txt', 'r') as fPos:
            for line in fPos:
                sentiment_words.add(line.strip())
        with open('/home/linhdang/rnn/SentimentAnalysis/negative-words.txt', 'r') as fNeg:
            for line in fNeg:
                sentiment_words.add(line.strip())
        print("sentiment size %d"%sentiment_words.__len__())
        # with open('/home/linhdang/rnn/SentimentAnalysis/stop_words', 'r') as fStop:
        #     for line in fStop:
        #         stop_words.add(line.strip())
        # print("stopword size %d" % stop_words.__len__())

        # make dictionary
        with open(self.pDict, 'r') as fDict:
            for line in fDict:
                word_count = line.split("\t")
                word_dict[word_count[0]] = total_words
                total_words += 1
        print("dict size %d"%total_words)

        # encode data and label
        with open(self.pEncodeData, 'w') as fEncodeData, \
                open(self.pEncodeLabel, 'w') as fEncodeLabel, \
                open("/home/linhdang/rnn/dataset/data", 'w') as fData:
            file_data = [self.pPositive, self.pNegative]
            for file_name in file_data:
                if file_name == self.pPositive:
                    label = 0
                else: label = 1
                with open(file_name, 'r') as fFile:
                    for line in fFile:
                        sentence_length = 0
                        sentence = []
                        tokens = word_tokenize(line)
                        for token in tokens:
                            if token in word_dict:
                                sentence.append(word_dict[token])
                                sentence_length += 1
                        if sentence_length > MIN_SENT_LENGTH:
                            fEncodeData.write(" ".join(str(x) for x in sentence)+"\n")
                            fEncodeLabel.write("%d\n"%label)
                            fData.write(line)
                            sentence_count += 1
            print("done positive and negative %d"%sentence_count)

            with open(self.pNeural, 'r') as fNeu:
                for line in fNeu:
                    sentence_length = 0
                    sentence = []
                    tokens = word_tokenize(line)
                    for token in tokens:
                        # loai bo sentiment word trong neural
                        # cach khac: line.notcontains(sentiment_word)
                        if token in sentiment_words:
                            sentence_length = 0
                            break
                        if token in word_dict:
                            sentence.append(word_dict[token])
                            sentence_length += 1
                    if sentence_length > MIN_SENT_LENGTH:
                        fEncodeData.write(" ".join(str(x) for x in sentence) + "\n")
                        fEncodeLabel.write("2\n")
                        fData.write(line)
                        sentence_count += 1
        print("sentence size %d"%sentence_count)

    def check1(self):
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        print("check")

        with open(self.pEncodeLabel, 'r') as f:
            for line in f:
                count3 += 1
                if int(line) == 0:
                    count+=1
                elif int(line) == 1:
                    count1+=1
                elif int(line) == 2:
                    count2+=1
        stop_words = set()
        with open('/home/linhdang/rnn/SentimentAnalysis/stop_words', 'r') as fStop:
            for line in fStop:
                stop_words.add(line.strip())
        with open(self.pDict, 'w') as fr:
            with open('/home/linhdang/rnn/dict', 'r') as fDict:
                for line in fDict:
                    word_count = line.split("\t")
                    if (int(word_count[1]) > 9) and (word_count[0] not in stop_words):
                        if int(word_count[1]) != 228442:
                            count+=1
                            fr.write(line)
                    # if (int(word_count[1]) <= 9) and (word_count[0] not in stop_words):
                    #     print(word_count[0])
        print(count)
        # print(count1)
        # print(count2)
        # print(count3)
        count = 0
        with open(self.pDict, 'r') as f:
            for line in f:
                count+=1
            print(count)

    # load as one hot vector
    # separate train and test data
    # padding to 50
    # // add more 2 dimension to represent positive and negative
    def loadTrainAndTest(self):
        data = []
        label = []
        count = 0
        MAX_POS = 110000
        MIN_NEG = 4080362
        MIN_NEU = 4416906
        TOTAL = 4437223
        SEPARATE = 200000
        with open(self.pEncodeData, 'r') as fData, open(self.pEncodeLabel, 'r') as fLabel:
            for line in fData:
                count+=1
                if (count <= MAX_POS) or \
                    (count > MIN_NEG and count <= (MIN_NEG+MAX_POS)) or \
                    (count > MIN_NEU and count <= TOTAL):
                    data.append(line)
            print(data.__len__())
            count = 0
            for line in fLabel:
                count +=1
                if (count <= MAX_POS) or \
                    (count > MIN_NEG and count <= (MIN_NEG+MAX_POS)) or \
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
        return data[:SEPARATE], label[:SEPARATE], data[SEPARATE:], label[SEPARATE:]


prepare = BuildDataset()
prepare.check1()
"""
stopword size 123
sentiment size 6785
dict size 67259
done positive and negative 4416906
sentence size 4437223

total 4437223
positive 4080362
negative 336544
neural 20317
"""