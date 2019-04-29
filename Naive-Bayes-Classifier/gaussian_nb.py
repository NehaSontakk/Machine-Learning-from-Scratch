#Gaussian Naive Bayes from scratch

#PREPROCESSING

#import dataset
#The dataset is divided into two classes and has 5 attributes
#The dataset describes how likely it is that a person will buy a piece of clothing given these attributes  
import pandas as pd
import numpy as np

data = pd.read_csv("Randomdataset.csv",names=['Id','Age','Income','Gender','Marital-Status','Label'])
print data.head()
print data.info()

#Drop id column
data = data.drop(columns=['Id'])
print data.head()

#Convert all the categorical values into numerical
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
data['Age'] = lbl.fit_transform(data['Age'])
data['Income'] = lbl.fit_transform(data['Income'])
data['Gender'] = lbl.fit_transform(data['Gender'])
data['Marital-Status'] = lbl.fit_transform(data['Marital-Status'])
data['Label'] = lbl.fit_transform(data['Label'])
print data.head()

data = np.array(data)
#seperate by class
def seperate_class(data):
	seperate = {}						#dictionary of seperated data
	for i in range(len(data)):
		row = data[i]
		if row[-1] not in seperate:			#if label not already present in the list then 
			seperate[row[-1]]=[]			#create a new index of that element and initialize a list
		seperate[row[-1]].append(row)			#append elements of each class found
	return seperate
print "\nSeperated by class:\n",seperate_class(data)

#calculate mean and standard deviation for each attribute in each class
import math
def mean(num):
	return sum(num)/len(num)
def stdev(num):
	num_m = mean(num)
	var = sum([pow(x-num_m,2) for x in num])/float(len(num)-1)
	return math.sqrt(var)

#make summaries of data
def summary(dataset):
	summary = [(mean(attribute),stdev(attribute)) for attribute in zip(*dataset)]#zip each attribute in the dataset together and iterate
	del summary[-1]	#don't need the last label columns summaries
	return summary

#calculate mean and stuff for every class
def calc_for_class(data):
	seperated_data = seperate_class(data)			#first seperate the classes
	classcalcs = {}
	for classvalue,row in seperated_data.iteritems():	#for each class (0 or 1) take class and the rows in it
		classcalcs[classvalue] = summary(row)		#append summary of each row to the particular class
	return classcalcs
		

data_details = calc_for_class(data)
print "\nMean and Standard deviations for each class:\n",data_details


#make predictions
#calculate that a given instance belongs to a particular class

def calcprobability(value,mean,stddev):		#take a value mean and standard deviations to calculate gaussian normal distribution
	numerator = math.exp(-((value-mean)**2/(2*stddev**2)))
	denominator = math.sqrt(2*math.pi*(stddev**2))
	return numerator/denominator



#now find probabilities for each class and multiply those with each other
def calc_prob_class(summaries,testcase):
	probs = {}
	for classvalue,calcs in summaries.iteritems():	#take the means for each class
		probs[classvalue] = 1		#because we have to multiply it later
		for i in range(len(calcs)):		#unbox all the values in each row
			mean,stddev = calcs[i]
			value = testcase
			probs[classvalue] *= calcprobability(value,mean,stddev)
	return probs

#find the highest probability class for testcase
def predict(summaries,testcase):
	#find probabilities for classes
	probs = calc_prob_class(summaries,testcase)
	final_label = None
	current_probab = -1
	for classvalue,prob in probs.iteritems():
		if final_label == None or prob > current_probab:
			current_probab = prob
			final_label = classvalue
			print "label and probability is:",final_label,current_probab
	return final_label

def use_predict(summaries,testcase):
	predictions =[]
	for i in range(len(testcase)):
		result = predict(summaries,testcase)
		predictions.append(result)
	return predictions

#test data is : Age <21 , Income is high, gender is male , marital status is married
test_data = np.array([1,0,1,0,'?'])
test_data = np.reshape(-1,1)


final_predictions = use_predict(data_details,test_data)
print final_predictions
if final_predictions == [1]:
	print "\nAccording to the test data person buys the item"
else:
	print "\nAccording to this person does not buy item"
