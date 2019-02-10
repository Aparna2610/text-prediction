################################################################################
# Filename: Model.py
# Description : Main script for preprocessing and computing performance using  
#               PMI                                                            
# Author : Aparna Gangwar                                                      
# Created on 28th Feb 2018                                                     
# Usage :                                                                      
#           $ python3 model.py command [options]                                
#       commands:
#           preprocess: to preprocess and compute model parameters for PMi
#               -d path     path of folder of dataset of books for traning
#                           default : '../Dataset/Holmes_Training_Data'
#
#               -q path     path dataset of question set
#                           default : '../Dataset/testing_data.csv'
#
#               -a path     path dataset of answers to question set
#                           default : '../Dataset/test_answer.csv'
#
#               -c path     path to save cooccurance matrix
#                           default : 'model/coMat'
#
#               -v path     path to save Vocabulary
#                           default : 'model/Vocab.txt'
#
#           NOTE: this step takes 3-4 hours on regular CPU and 1-2 hour on
#           googles colaboratory
#
#           runTest: to compute performance of PMI model
#               -c path     path to cooccurance matrix obtained from preprocess
#                           step with name of "coMat.npy"
#                           default : 'model/coMat.npy'
#
#               -v path     path to Vocabulary obtained from preprocess step
#                           default : 'model/Vocab.txt'
#
#               -s path     path to stopword list, it has been provided with
#                           model
#                           default : 'model/stopword.npy'
#
#               -verbose    print intermediate accuracy
#
# Examples:
#       To initiate preprocessing
#           $ python3 model.py preprocess
#
#       To compute performance
#           $ python3 model.py runTest -verbose
#
#       To get answer of question N [1 to 1040]
#           $ python3 model.py getAnswer -qNo N
#
################################################################################

# import libraries
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
from os import listdir
from os.path import isfile
import numpy as np
from QuestionProcessor import QuestionProcessor
import sys
from PMI_preprocessor import processor

# Config Variable and shared variable
verbose = False
indexToWord = {}
wordToIndex = {}
Freq = {}
coMat = None
vocabFilename = "model/Vocab.txt"
coMatFilename = "model/coMat.npy"
stopwordsFilename = "model/stopwords.npy"
test_data = "../Dataset/testing_data.csv"
test_answer = "../Dataset/test_answer.csv"
datasetFolder = "../Dataset/Holmes_Training_Data/"
vocabFilename = "model/Vocab.txt"
unk = []
qp = None
allOptions = []

# Load Vocab
def loadVocabulary(vocabFilename):
    fp = open(vocabFilename,"r")
    min_threshold = 5
    for line in fp.readlines():
        tok = line.strip().split(",")
        index = int(tok[-1])
        freq = int(float(tok[-2]))
        word = ",".join(tok[:-2])
        indexToWord[index] = word
        wordToIndex[word] = index
        Freq[index] = freq
    fp.close()
    return indexToWord, wordToIndex, Freq

def loadQuestionSet(test_data,test_answer):  
    qp = QuestionProcessor(test_data,test_answer)

    allOptions = []
    for qNo in range(1,1041):
            q = qp.questionSet[qNo]
            allOptions.extend([w for w in ([ q.options[z] for z in q.options.keys()])])
    allOptions = list(set(allOptions))
    return qp, allOptions

def loadCoMat(coMatFilename):
    coMat = np.load(coMatFilename)
    coMat = np.int64(coMat)
    if len(unk) > 0:
        coMat[unk,:] = 0
        coMat[:,unk] = 0
    return coMat

def loadStopwords(stopwordsFilename):
    if isfile(stopwordsFilename):
        unk = np.load(stopwordsFilename)
    else:
        stop = set(stopwords.words('english'))
        unk = [wordToIndex[i] for i in wordToIndex.keys() if i in stop and not i in allOptions]
        unk.extend([wordToIndex[i] for i in wordToIndex.keys() if not re.match("[\w\d]+",i) and not i in allOptions])
        unk.append(3)
    return unk

def computeTotalFreq(I,J,coMat):
    T = 0
    for i in I:
        for j in J:
            T += coMat[i][j]
    return T

def PMI(i, j, s_i, s_j, coMat, T):
    return np.log2((coMat[i][j]*T)/(s_i*s_j))

def computeSimilarity(I,J,coMat):
    T = computeTotalFreq(I,J,coMat)
    
    sim = np.zeros(len(I))
    for i in range(len(I)):
        s_i = np.sum(coMat[I[i]][:])+1
        for j in J:
            s_j = np.sum(coMat[:][j])+1
            sim[i] += PMI(I[i],j,s_i,s_j,coMat,T)
    return sim

def getAnswer(q):
    q1 = q.question
    q_token = word_tokenize(q1.lower())
    
    # Tokenize & Encode question
    J = [wordToIndex[x] for x in q_token if x in wordToIndex.keys() and not wordToIndex[x] in unk]
    
    # Encode options
    options = [w for w in ([ q.options[z] for z in q.options.keys()])]
    I = [ wordToIndex[x] for x in options if x in wordToIndex.keys() and not wordToIndex[x] in unk]
    
    s = computeSimilarity(I,J,coMat)
    if(len(s) == 0):
        return None
    
    return [a for a,opt in zip(q.options.keys(),options) if opt == indexToWord[I[np.argmax(s)]]][0]


def computePerformance():
    correct = 0
    total = 0
    for qNo in range(1,1041):
        q = qp.questionSet[qNo]
        ans = getAnswer(q)
        #if(np.std(ans[1])>38 or np.std(ans[1])<0.72):
        #    continue
        if ans:
            total += 1
            if q.answer == ans:
                #cVar.append(np.var(ans[1]))
                correct += 1
            #else:
                #iVar.append(np.var(ans[1]))
            if verbose:
                print(qNo, ans, q.answer, correct,correct/total)
    print("Achieved Accuracy with PMI",correct*100/total)
    return correct*100/total

def usage():
    print()
    print("# Usage :\n\
#           $ python3 model.py command [options]                                \n\
#       commands:\n\
#           preprocess: to preprocess and compute model parameters for PMi\n\
#               -d path     path of folder of dataset of books for traning\n\
#                           default : '../Dataset/Holmes_Training_Data'\n\
#\n\
#               -q path     path dataset of question set\n\
#                           default : '../Dataset/testing_data.csv'\n\
#\n\
#               -a path     path dataset of answers to question set\n\
#                           default : '../Dataset/test_answer.csv'\n\
#\n\
#               -c path     path to save cooccurance matrix\n\
#                           default : 'model/coMat'\n\
#\n\
#               -v path     path to save Vocabulary\n\
#                           default : 'model/Vocab.txt'\n\
#\n\
#           NOTE: this step takes 3-4 hours on regular CPU and 1-2 hour on\n\
#           googles colaboratory\n\
#\n\
#           runTest: to compute performance of PMI model\n\
#               -c path     path to cooccurance matrix obtained from preprocess\n\
#                           step with name of \"coMat.npy\"\n\
#                           default : 'model/coMat.npy'\n\
#\n\
#               -v path     path to Vocabulary obtained from preprocess step\n\
#                           default : 'model/Vocab.txt'\n\
#\n\
#               -s path     path to stopword list, it has been provided with\n\
#                           model\n\
#                           default : 'model/stopword.npy'\n\
#\n\
#               -verbose    print intermediate accuracy\n\
#\n\
# Examples:\n\
#       To initiate preprocessing\n\
#           $ python3 model.py preprocess\n\
#\n\
#       To compute performance\n\
#           $ python3 model.py runTest -verbose\n\
#\n\
#       To get answer of question N [1 to 1040]\n\
#           $ python3 model.py getAnswer -qNo N\n\
#")
    exit()

if __name__ == '__main__':
    command = None
    if(len(sys.argv)>1):
        command = sys.argv[1]

    if not command == None:
        # Parse for options
        options = {}
        for i in range(2,len(sys.argv),2):
            options[sys.argv[i]] = None
            if(i+1 < len(sys.argv)):
                options[sys.argv[i]] = sys.argv[i+1]

        # Parsed Options
        if("-d" in options.keys()):
            if(not options["-d"]):
                print("Missing argument for -d")
                usage()
            datasetFolder = options["-d"]

        if("-q" in options.keys()):
            test_data = options["-q"]
            if(not options["-q"]):
                print("Missing argument for -q")
                usage()

        if("-a" in options.keys()):
            test_answer = options["-a"]
            if(not options["-a"]):
                print("Missing argument for -a")
                usage()

        if("-v" in options.keys()):
            vocabFilename = options["-v"]
            if(not options["-v"]):
                print("Missing argument for -v")
                usage()

        if("-s" in options.keys()):
            stopwordsFilename = options["-s"]
            if(not options["-s"]):
                print("Missing argument for -s")
                usage()

        if("-verbose" in options.keys()):
            verbose = True

        #print("options",options)
        if command.lower() == "preprocess":
            #datasetFolder,test_data,test_answer,vocabFilename,coMatFilename
            
            saveCoMatFilename = "model/coMat"
            if("-c" in options.keys()):
                if(not options["-c"]):
                    print("Missing argument for -c")
                    usage()

                saveCoMatFilename = options["-c"]
            processor(datasetFolder,test_data,test_answer,vocabFilename,saveCoMatFilename)
            exit()
            
        elif command.lower() == "runtest":
            if("-c" in options.keys()):
                if(not options["-c"]):
                    print("Missing argument for -c")
                    usage()
                coMatFilename = options["-c"]

            # loadQuestion
            qp,allOptions = loadQuestionSet(test_data,test_answer)

            # loadVocab
            indexToWord, wordToIndex, Freq = loadVocabulary(vocabFilename)
            # loadStopwords
            unk = loadStopwords(stopwordsFilename)

            # loadComat
            coMat = loadCoMat(coMatFilename)
            
            # computePerformance
            computePerformance()
            
        elif command.lower() == "getanswer":
            if("-c" in options.keys()):
                if(not options["-c"]):
                    print("Missing argument for -c")
                    usage()
                coMatFilename = options["-c"]

            no = 0
            if("-qNo" in options.keys()):
                no = int(options["-qNo"])
                if(not options["-qNo"]):
                    print("Missing argument for -s")
                    usage()
            else:
                print("Missing -qNo N , where N ranges from 1 to 1040")
                usage()

            if no < 1 or no > 1040:
                print("Question not available")
                exit()

            # loadQuestion
            qp,allOptions = loadQuestionSet(test_data,test_answer)

            # loadVocab
            indexToWord, wordToIndex, Freq = loadVocabulary(vocabFilename)
            # loadStopwords
            unk = loadStopwords(stopwordsFilename)

            # loadComat
            coMat = loadCoMat(coMatFilename)

            print("Question:",qp.questionSet[no].question)
            print("Options")
            for key in qp.questionSet[no].options.keys():
                print("\t\t",key,qp.questionSet[no].options[key])
            print()
            print("Actual Answer:",qp.questionSet[no].answer,qp.questionSet[no].options[qp.questionSet[no].answer])
            predicted_answer = getAnswer(qp.questionSet[no])
            print("Predicted Answer:",predicted_answer,qp.questionSet[no].options[predicted_answer])

        exit()
    usage()