################################################################################
# Filename: Model.py
# Description : Main script for preprocessing and computing performance using  
#               NGram                                                            
# Author : Vaibhaw Raj                                                      
# Created on 28th Feb 2018                                                     
# Usage :                                                                      
#           $ python3 model.py
################################################################################

from QuestionProcessor import QuestionProcessor
import sys
from os import listdir
from os.path import isfile, join
import re
from nltk.tokenize import word_tokenize
import numpy as np

# Load question dataset and read correct answers against those questions
test_data = "../Dataset/testing_data.csv"
test_answer = "../Dataset/test_answer.csv"

qp = QuestionProcessor(test_data,test_answer)

nGrams = []
modelLoaded = False
if isfile("model/ngrams.npy"):
    nGrams = np.load("model/ngrams.npy")
    modelLoaded = True
    print(len(nGrams))

# Load Dataset
books = []
if not modelLoaded:
    mypath = '../Dataset/Holmes_Training_Data'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    headerSeparator = "*END*THE SMALL PRINT! FOR PUBLIC DOMAIN ETEXTS*Ver.04.29.93*END*"
    noOfFiles = 522
    for fname in onlyfiles:
        fp = open(mypath + "/" + fname,"r",encoding='cp1252')
        bookContent = ""
        headerSeparatorDetected = False
        print(fname)
        for line in fp.readlines():
            if not headerSeparatorDetected:
                if line.strip() == headerSeparator:
    #                print(line)
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


# Get position p of blank in the sentence
# Input:
#      question(str): Question having blank
# Output:
#      positionNo(int)

def getPositionOfBlank(question):
    qToken = word_tokenize(question.lower())
    index = 0
    for token in qToken:
        if token == "_____":
            return index
            break
        index += 1
    return -1

# Generate N-Grams from list of tokens of size n
# Input:
#        tokens(list) : List of tokens using word tokenizer
#        n : Size of n in N-Gram
# Output:
#        List of ngrams of size n separated by " " as a separator
getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]

def findAnswer(q,givenBooks):
    ques = q.question
    blankPos = getPositionOfBlank(ques)
    
    ngrams = [[],[],[]]
    qToken = word_tokenize(ques)

    getValidIndex = lambda blankPos,tokenLength,nGram:(max(blankPos-(nGram-1),0),min(blankPos+nGram,tokenLength-1))
    
    # Generate all 4-Grams
    (minIndex,maxIndex) = getValidIndex(blankPos,len(qToken),4)
    ngrams[0] = getNGram(qToken[minIndex:maxIndex],4)

    # Generate all trigrams
    (minIndex,maxIndex) = getValidIndex(blankPos,len(qToken),3)
    ngrams[1] = getNGram(qToken[minIndex:maxIndex],3)

    # Generate all bigrams
    (minIndex,maxIndex) = getValidIndex(blankPos,len(qToken),2)
    ngrams[2] = getNGram(qToken[minIndex:maxIndex],2)
    
    # Prepare list of all three n-grams with words in option
    options = q.options
    listOfNgrams = {}
    for key in options.keys():
        opt = options[key]
        listOfNgrams[key] = [[],[],[]]
        for index in range(0,len(ngrams)):
            listOfNgrams[key][index] = []
            for ngram in ngrams[index]:
                listOfNgrams[key][index].append(re.sub("_____",opt,ngram))

    # Evaluate Scores for sentence by considering each word one by one
    scoreIndex = 0
    scores = [0,0,0,0,0]
    for key in listOfNgrams.keys():
        # For each list of ngram
        ngramScore = 3
        #print("Debug",key)
        for ngramList in listOfNgrams[key]:
            #print("Debug",ngramList)
            for ngram in ngramList:
                #print("Debug",ngram)
                if modelLoaded:
                    if ngram in nGrams:
                        scores[scoreIndex] += ngramScore
                else:
                    for bookNo in range(0,len(givenBooks)):
                        #print("Debug",bookNo,ngram)
                        text = givenBooks[bookNo]
                        if len(re.findall(ngram,text))>0:
                            #print(">>> Matched",ngram,"[",bookNo,"]")
                            nGrams.append(ngram)
                            scores[scoreIndex] += ngramScore
                            break
            ngramScore = ngramScore - 1
        scoreIndex += 1
    #print(scores)

    # Return the option having maximum score
    
    return list(options.keys())[np.argmax(scores)]

def computePerformance(qp,givenBooks):
    correct = 0
    total = 0
    for qNo in range(1,1041):
        q = qp.questionSet[qNo]
        answer = findAnswer(q,givenBooks)
        if(answer == q.answer):
            correct += 1
        total += 1
        print(qNo,answer,q.answer,correct/total)
    print("Achieved accuracy for nGrams",correct*100/total)
        
computePerformance(qp,books)
if not modelLoaded:
    print(len(nGrams))
    np.save("model/ngrams",nGrams)