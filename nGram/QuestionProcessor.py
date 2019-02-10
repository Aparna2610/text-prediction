################################################################################
# Filename: QuestionProcessor.python3                                                      
# Description : Question Processor loads question data set and answers
# Author : Vaibhaw Raj                                                      
# Created on 21st Feb 2018  
################################################################################

import csv

#testing_data_FN = "../Dataset/testing_data.csv"
#testing_data_answer_FN = "../Dataset/test_answer.csv"

class QuestionProcessor:
	def __init__(self,testing_data_FN,testing_data_answer_FN):
		self.questionSet = {}
		self.testing_data_FN = testing_data_FN
		self.testing_data_answer_FN = testing_data_answer_FN
		self.loadQuestion()

	def loadQuestion(self):
		with open(self.testing_data_FN,'r') as csvfile:
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
		with open(self.testing_data_answer_FN,'r') as csvfile:
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