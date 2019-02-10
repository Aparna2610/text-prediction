
# coding: utf-8

# In[1]:


import time
start_time = time.time()

from os.path import join, isfile
from os import listdir

# Load dataset
books = []
mypath = '../Dataset/Holmes_Training_Data'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
headerSeparator = "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*"
noOfFiles = 522
for fname in onlyfiles:
    fp = open(mypath + "/" + fname,"r",encoding='cp1252')
    bookContent = ""
    headerSeparatorDetected = False
    for line in fp.readlines():
        if not headerSeparatorDetected:
            if line.strip() == headerSeparator:
                headerSeparatorDetected = True
            continue
        line = line.strip()
        if not (len(line) == 0):
            bookContent = bookContent + " " + line.lower()
    books.append(bookContent)
    fp.close()
    noOfFiles = noOfFiles - 1
    if noOfFiles == 0:
        break

    # Number of books read
    print("No of books read",len(books))


# In[2]:


stoplist = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
texts = [[word for word in book.split() if word not in stoplist] for book in books]


# In[3]:


from gensim import corpora, models
from scipy import spatial

# Create a dictionary of the text corpus
dictionary = corpora.Dictionary(texts)
dictionary.save('deerwester.dict')


# In[4]:


# Create bag of words
corpus = [dictionary.doc2bow(text) for text in texts]


# In[5]:


corpora.MmCorpus.serialize('deerwester.mm', corpus)
corpus = corpora.MmCorpus('deerwester.mm') 
dictionary = corpora.Dictionary.load('deerwester.dict') 


# In[6]:


# Create TFIDF matrix
tfidf = models.TfidfModel(corpus) 
corpus_tfidf = tfidf[corpus] 


# In[120]:


# Create LSI model of the TFIDF matrix
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20) 
corpus_lsi = lsi[corpus_tfidf] 
lsi.save('model.lsi')  


# In[121]:


lsi = models.LsiModel.load('model.lsi') 


# In[122]:


# Filename: QuestionProcessor
# Description : Processes all questions and provides instance
# Author : Vaibhaw Raj
# Created on 22nd Feb 2018

import csv

testing_data_FN = "../Dataset/testing_data.csv"
testing_data_answer_FN = "../Dataset/test_answer.csv"

class QuestionProcessor:
    def __init__(self):
        self.questionSet = {}
        self.loadQuestion()

    def loadQuestion(self):
        with open(testing_data_FN,'r') as csvfile:
            qReader = csv.reader(csvfile,delimiter=',',quotechar='"')
            isHeader = True
            for row in qReader:
                if isHeader:
                    isHeader = False
                    continue
                q = Question()
                q.qNo = int(row[0])
                q.question = row[1]
                q.options["a"] = row[2]
                q.options["b"] = row[3]
                q.options["c"] = row[4]
                q.options["d"] = row[5]
                q.options["e"] = row[6]
                self.questionSet[q.qNo] = q
        with open(testing_data_answer_FN,'r') as csvfile:
            aReader = csv.reader(csvfile,delimiter=',',quotechar='"')
            isHeader = True
            for row in aReader:
                if isHeader:
                    isHeader = False
                    continue
                qNo = int(row[0])
                answer = row[1]
                self.questionSet[qNo].answer = answer

class Question:
    def __init__(self):
        self.qNo = None
        self.question = ""
        self.options = {"a":None,"b":None,"c":None,"d":None,"e":None}
        self.answer = ""
    def __repr__(self):
        return self.question


# In[123]:


qp = QuestionProcessor()


# In[124]:


from nltk.tokenize import word_tokenize

def getPositionOfBlank(question):
    qToken = word_tokenize(question.lower())
    index = 0
    for token in qToken:
        if token == "_____":
            return index
            break
        index += 1
    return -1


# In[125]:


import sys
import operator

def getAnswer(q):
    blankPos = getPositionOfBlank(q.question)
    #print(blankPos)
    stoplist.add("_____")
    qsTokens = [word for word in q.question.split() if word not in stoplist]
#    print(qsTokens)
    qs_vectors = []
    for tokens in qsTokens:
        qs_bow = dictionary.doc2bow(tokens.lower().split())
        qs_tfidf = tfidf[qs_bow]
        qs_lsi = lsi[qs_tfidf]
        qs_vectors.append(qs_lsi)

    optToInd = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
    IndToOpt = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}
    options_vectors = [[] for i in range(5)]

    for key,option in q.options.items():
        option_bow = dictionary.doc2bow(option.lower().split())
        option_tfidf = tfidf[option_bow]
        option_lsi = lsi[option_tfidf]
        options_vectors[optToInd[key]] =  option_lsi

    qs_features = []
    option_features = []
    qs_features = [[tup[1] for tup in qsVec] for qsVec in qs_vectors if len(qsVec) > 0]
    option_features = [[tup[1] for tup in opVec] for opVec in options_vectors if len(opVec) > 0]

    resultForQues = {}
    for i in range(len(option_features)):
        resultForOpt = 0
        for j in range(len(qs_features)):
            resultForOpt += 1 - spatial.distance.cosine(qs_features[j], option_features[i])
        resultForQues[IndToOpt[i]] = resultForOpt
    ans = max(resultForQues.items(), key=operator.itemgetter(1))[0]
    return ans


# In[126]:


def runTest():
    correct = 0
    total = 0
    for qNo in range(1,1041):
        q = qp.questionSet[qNo]
        ans = getAnswer(q)
        if ans:
            total += 1
            if q.answer == ans:
                correct += 1
            print(qNo, ans, q.answer, correct,correct/total)
    print("Achieved Accuracy with LSA",correct*100/total)


# In[127]:


if __name__ == '__main__':
    if 'runTest' in sys.argv:
        runTest()
    else:
        print("Usage: python3 model.py runTest")

