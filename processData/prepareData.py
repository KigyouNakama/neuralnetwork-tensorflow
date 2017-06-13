"""
thực ra trong mỗi câu ngắn đều có thể thể hiện cảm xúc của người nói ra, nhưng ở đây đang phân
tích những câu có độ dài dưới 140 ký tự

stemming ảnh hưởng như thế nào tới chất lượng dự đoán?

"""

import xml.etree.ElementTree as etree
import os
import json
import gzip
import csv
import re
import collections
import codecs

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import PorterStemmer

test = os.path.abspath('../dataamazon/Reviews-9-products/test.txt')
csv.field_size_limit(500 * 1024 * 1024)
class PreparingData:
    pathCustomerReviews = '../dataamazon/CustomerReviews-3domains(IJCAI2015)/'
    pathReviews = '../dataamazon/Reviews-9-products/'
    pathCustomers = '../dataamazon/customer review data/'
    pathPros = '../dataamazon/pros-cons/'
    pathResult = '/media/dhbk/New Volume/linh/result.csv'
    pathCorpus = '/media/dhbk/New Volume/linh/corpus.csv'
    pathShortText = '/media/dhbk/New Volume/linh/shortText.csv'
    pathSentDict = '/media/dhbk/New Volume/linh/SentimentDictionary.csv'
    pathCountedDict = '/media/dhbk/New Volume/linh/CountedDictionary.csv'
    pathOriginalDict = '/media/dhbk/New Volume/linh/OriginalDictionary.csv'
    pathPositive = '../english/positive-words.txt'
    pathNegative = '../english/negative-words.txt'
    pathEncodedData = '/media/dhbk/New Volume/linh/encoded_data'
    pathEncodedLabel = '/media/dhbk/New Volume/linh/encoded_label'

    computer = os.path.abspath(pathCustomerReviews+'Speaker.xml')
    reviews = os.path.abspath(pathReviews+'Canon S100.txt')
    customer = os.path.abspath(pathCustomers+'Nokia 6610.txt')
    proscons = os.path.abspath(pathPros+'IntegratedCons.txt')
    instantVideo = os.path.abspath('../dataamazon/Amazon_Instant_Video_5.json')
    pathAmazon = os.path.abspath('/media/dhbk/New Volume/linh/item_dedup.json.gz')
    path = os.path.splitdrive(os.path.dirname(__file__))

    porter = PorterStemmer()
    def __init__(self):
        print('start')

    # for CustomerReview
    def processReviews(self):
        f = open(self.customer, 'r')
        for line in f:
            line = line.strip()
            polarityIndex = line.find('[')
            if polarityIndex != -1:
                if line.endswith('.'):
                    line = line.rstrip(".") # strip dot from last position
                startIndex = line.find('##') + 2
                if line[polarityIndex + 1] == '-':
                    print(line[startIndex:] + ',', 'negative')
                elif line[polarityIndex + 1] == '+':
                    print(line[startIndex:] + ',', 'positive')
        f.close()

    # for Reviews
    def processCustomer(self):
        tree = etree.parse(self.computer)
        root = tree.getroot()

        for sent in root:
            if len(sent) > 1:
                count = 0
                for child in sent[1]:
                    if child.attrib['polarity'] == 'positive':
                        count = count + 1
                    else:
                        count = count + 1
                if count == len(sent[1]):
                    line = sent[0].text.strip()
                    if line.endswith('.'):
                        line = line.rstrip(".")  # strip dot from last position
                    print(line+',', sent[1][0].attrib['polarity'])

    # for customer reivew data
    def processProsAndCons(self):
        f = open(self.proscons, 'r')
        for line in f:
            closedIndex = line.find('</')
            if line.find('<Cons>') != -1:
                openedIndex = line.find('<Cons>')
                print(line[openedIndex+6:closedIndex], 'negative')
            elif line.find('<Pros>') != -1:
                openedIndex = line.find('<Pros>')
                print(line[openedIndex+6:closedIndex], 'positive')
        f.close()

    def processJsonData(self):
        f = open(self.instantVideo, 'r')
        f1 = open(self.pathResult, 'a')
        pCount = nCount = 0
        for line in f:
            json_data = json.loads(line)
            if json_data['overall'] < 3.0:
                nCount += 1
            else:
                pCount += 1
            if pCount == 5:
                break
            f1.write(json_data['reviewText'].rstrip(".")+', ' + ("negative\n" if json_data['overall'] < 3.0 else "positive\n"))
        #    print(json_data['reviewText'].rstrip(".")+',',"negative" if json_data['overall'] < 3.0 else "positive")
        print(nCount, pCount)

        f.close()
        f1.close()
        print(os.listdir(self.pathCustomerReviews))
    #        yield eval(l)

    #   82677139
    def processAmazonData(self):
        print("process gzip")
        print(self.pathAmazon)
        readAmazon = gzip.open(self.pathAmazon, 'rt')
        f = open(self.pathResult, 'w')
        count = nCount = pCount = 0
        for line in readAmazon:
            count += 1
            json_data = json.loads(line)
            if json_data['overall'] < 3.0:
                nCount += 1
                f.write(json_data['reviewText'] + ', negative\n')
            elif json_data['overall'] > 3.0:
                if pCount < 5000000:
                    pCount += 1
                    f.write(json_data['reviewText'] + ', positive\n')
            if nCount >= 5000000:
                break
        print(nCount, pCount, count)
        readAmazon.close()
        f.close()

    """
    not processing
    count, countShortText, negativeCountST, positiveCountST, neutralCountST
    82677139 20478923 2177522 16850793 1450608
    """
    def statitis(self):
        """
        shortText: 20.461.034
        negative: 2.174.524
        positive: 16.836.944
        neutral: 1.449.566

        write review with length <= 140 into file shortText with following format:
            sentiment   reviewText
        :return:
        """
        readAmazon = gzip.open(self.pathAmazon, 'rt')
        fShortText = open(self.pathShortText, 'w')
        countShortText = neutralCountST = negativeCountST = positiveCountST= 0
        for line in readAmazon:
            json_data = json.loads(line)
            s = json_data['reviewText']

            if len(s) > 0 and len(s) < 141:
                if "\0" in s:
                    s = s.replace("\0", "")
                    if len(s) < 1:
                        continue

                countShortText += 1
                if json_data['overall'] < 3.0:
                    negativeCountST += 1
                    polarity = "negative"
                elif json_data['overall'] > 3.0:
                    positiveCountST += 1
                    polarity = "positive"
                else:
                    neutralCountST += 1
                    polarity = "neutral"
                fShortText.write(polarity+"\t"+s+"\n")
        print("shortText: %d\n negative: %d\n positive: %d\n neutral: %d\n"%(countShortText, negativeCountST, positiveCountST, neutralCountST))
        readAmazon.close()
        fShortText.close()
    # 11856727 ['negative', 'Recieved the harmonica in a good case. but the 3rd reed wasnt working. Its has the same tone as the second reed.']
    # 11867527 positive	I love this app on my phone, but it freeze crashes everytime I use my Xoom. please fix this developers!
    def readDictionary(self):
        fOriginalDict = open(self.pathOriginalDict, 'r')
        reader = csv.reader(fOriginalDict, delimiter='\t')
        words_dict = dict()
        count = list()
        for row in reader:
            if len(row) < 2:
                continue
            if row[0] == "beautiful":
                print(row[0])
                break

    def readShortText(self):
        """
        build counted dictionary.
        :return: dictionary
        """
        words_dict = dict()
        stopwds = stopwords.words('english')
        fShortText = open(self.pathShortText, 'r')
        reader = csv.reader(fShortText, delimiter='\t')

        for row in reader:
            if len(row) < 2:
                continue

            words_list = word_tokenize(row[1].lower())
            for word in words_list:
                if (len(word) == 1) or (word in stopwds) or (not re.search(r'^[a-zA-Z]', word)):
                    continue
                if word not in words_dict.keys():
                    words_dict[word] = 1
                else:
                    words_dict[word] += 1

        fShortText.close()
        #counted        dict        size        1842829
        #print("counted dict size %d"%len(words_dict))
        return words_dict

    def readShortText1(self):
        fShortText = open(self.pathShortText, 'r')
        reader = csv.reader(fShortText, delimiter='\t')

        for row in reader:
            if len(row) < 2:
                continue

    def preprocessing(self):
        original_dict = self.readShortText()
        fpos = open(self.pathPositive, 'r')
        fneg = open(self.pathNegative, 'r')
        fCountedDict = open(self.pathCountedDict, 'w')
        fSentDict = open(self.pathSentDict, 'w')
        fOriginalDict = open(self.pathOriginalDict, 'w')

        total_word_count = 0
        words_dict = dict()
        pos_list = list()
        neg_list = list()

        for line in fpos:
            words_dict[line.split()[0]] = total_word_count
            total_word_count += 1
            fSentDict.write("%s\t%d\n" % (line.split()[0], 0))
            pos_list.append(line.split()[0])
        for line in fneg:
            words_dict[line.split()[0]] = total_word_count
            total_word_count += 1
            fSentDict.write("%s\t%d\n" % (line.split()[0], 1))
            neg_list.append(line.split()[0])

        print("write original dict to file")
        for w, c in original_dict.items():
            fOriginalDict.write("%s\t%d\n"%(w,c))

        counted_dict = {w:c for w,c in original_dict.items() if c > 3} # delete word happend less than 3 times
        print("size dict of word happended more than 3 times %d"%len(counted_dict))

        print("write shorted dict to file")
        for w, c in counted_dict.items():
            fCountedDict.write("%s\t%d\n"%(w,c))

        # starting to create sentiment dictionary
        words_set = counted_dict.keys()
        words_set = words_set - set(pos_list) - set(neg_list)

        for word in words_set:
            words_dict[word] = total_word_count
            total_word_count += 1
            fSentDict.write("%s %d\n" % (word, 2))

        # encode data
        sent_dict = dict()
        sent_dict["positive"] = 0
        sent_dict["negative"] = 1
        sent_dict["neutral"] = 2

        fDataFile = open(self.pathEncodedData, 'w')
        fLabelFile = open(self.pathEncodedLabel, 'w')
        fReader = csv.reader(self.pathShortText, delimiter='\t')
        for row in fReader:
            if len(row) < 2:
                continue

            words_list = word_tokenize(row[1].lower())
            for word in words_list:
                if word in words_dict.keys():
                    if word in pos_list:
                        sent = 0
                    elif word in neg_list:
                        sent = 1
                    else:
                        sent = 2
                    fDataFile.write(str(words_dict[word])+'/'+str(sent) + ' ')

            fLabelFile.write(sent_dict[row[0]])
            fLabelFile.write('\n')
            fDataFile.write('\n')

prepare = PreparingData()
prepare.readDictionary()
"""
wd, ow, owk = prepare.preprocessing()
print(ow, owk) # n_time_happened_word 45788 word_in_nplist_after_stemming 822
print(wd.get('cramps'))
print(wd.get('cramp'))
print(wd.get('disrupted'))
print(wd.get('disrupt'))
"""
# words   number of sentences
# 1730000 9541086
# 1800000 10004818
# 6190000 82551854
# dict size 6195542