################################################################################
# Filename: PMI_preprocessor.py
# Description : PMI preprocessor                                                        
# Author : Aparna Gangwar                                                      
# Created on 28th Feb 2018  
################################################################################

from nltk.tokenize import sent_tokenize,word_tokenize
import sys
from os import listdir
from os.path import isfile, join
import timeit
import numpy as np

from QuestionProcessor import QuestionProcessor

def processor(datasetFolder,test_data,test_answer,vocabFilename,coMatFilename):
    qp = QuestionProcessor(test_data,test_answer)

    # Preparing Question Set Vocabulary
    # Voc

    words = []

    for qNo in range(1,1041):
        q = qp.questionSet[qNo]
        question = q.question
        for sent in sent_tokenize(question.lower()):
            for word in word_tokenize(sent):
                words.append(word)
        for opt in q.options.keys():
            words.append(q.options[opt])

    vocabulary = list(set(words))
    words = np.zeros(len(vocabulary)+2)

    indexToWord = {}
    wordToIndex = {}
    for word,index in zip(vocabulary,range(2,len(vocabulary)+2)):
        indexToWord[index] = word
        wordToIndex[word] = index

    del qp

    vocabSize = len(indexToWord)+2
    print("Length of Vocab",vocabSize)

    coMat = np.ones([vocabSize,vocabSize],dtype=np.int32)

    # Load devDataSet

    devDataSet = [f for f in listdir(datasetFolder) if isfile(join(datasetFolder, f))]
    headerSeparator = "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*"

    noOfBooks = 2
    count = 0

    readTimer = timeit.default_timer()
    for fname in devDataSet:
        tokenizeTimer = timeit.default_timer()

        # read training file
        fp = open(datasetFolder + fname,"r",encoding='cp1252')

        bookContent = ""
        headerSeparatorDetected = False
        print("Reading",fname)
        for line in fp.readlines():
            if not headerSeparatorDetected:
                if line.strip() == headerSeparator:
                    headerSeparatorDetected = True
                continue
            line = line.strip()
            if not (len(line) == 0):
                bookContent = bookContent + " " + line.lower()
        fp.close()

        # tokenize and encode

        print("Tokenizing & Encoding",fname)
        bookSentences = []
        for sent in sent_tokenize(bookContent):
            temp_sent = []
            for word in word_tokenize(sent):
                if word in vocabulary:
                    temp_sent.append(wordToIndex[word])
                    words[wordToIndex[word]] += 1
            if(len(temp_sent)>0):
                # Calculate cooccurance
                for i in range(len(temp_sent)):
                  for j in range(i+1,len(temp_sent)):
                    coMat[temp_sent[i]][temp_sent[j]] += 1
                    coMat[temp_sent[j]][temp_sent[i]] += 1

        print("Elapsed time for this book\t\t\t\t",timeit.default_timer() - tokenizeTimer)
        count += 1
        if noOfBooks == 0:
        	break
        noOfBooks = noOfBooks-1

    # Number of books read
    print("No of books read & Tokenized",count,"in",timeit.default_timer() - readTimer)

    # Saving vocabulary
    fp = open(vocabFilename,"w")

    for item in vocabulary:
        fp.write(str(item) + "," + str(words[wordToIndex[item]]) + "," + str(wordToIndex[item]) + "\n")
    fp.close()
    print("Vocabulary saved to",vocabFilename)

    # Saving co-occurrance matrix
    #coMatFilename = "model/coMat"
    np.save(coMatFilename,coMat)
    print("Co-occurance matrix saved to",coMatFilename)

if __name__ == "__main__":
    datasetFolder = "../Dataset/Holmes_Training_Data/"
    test_data = "../Dataset/testing_data.csv"
    test_answer = "../Dataset/test_answer.csv"
    vocabFilename = "model/Vocab.txt"
    coMatFilename = "model/coMat"
    processor(datasetFolder,test_data,test_answer,vocabFilename,coMatFilename)