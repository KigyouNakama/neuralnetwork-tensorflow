import re, sys, numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

word = "linh linh.linh.linh linh%%%%linh%linh"
review = re.sub('[^A-Za-z- ]+', ' ', word)
review = ' '.join(review.split()).lower()
print(review)
words_list = word_tokenize(review)
print(words_list)
wordVector = np.random.uniform(-1,1,200)
print(wordVector)

l = []
print(len(l))

li = [1,2,3]
l.append(wordVector)
print(len(l))
l.append([4,5,6])
print(l)
lis = []
lis.append(l)
print(len(l))

output = [1,2]
print(output[-1])