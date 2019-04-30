#Naive bayes
#Numpy array manipulation

from __future__ import division
import pandas as pd
import numpy as np

data = pd.read_csv("Randomdataset.csv",names=['Id','Age','Income','Gender','Marital-Status','Label'])
print data.head()
print data.info()

#drop id column
data = data.drop(columns ='Id')

#Get overall yes and no probability

data['Label'] = np.array(data['Label'])
print "\nUnique values in the final classification column: ",np.unique(data['Label'])
unique, counts = np.unique(data['Label'], return_counts=True)
overall_count = {}

for i in range(len(unique)):
	overall_count[unique[i]] = counts[i]
print "\nOverall count for yes and no:",overall_count

#seperate all the values into two classes i.e. yes and no
data = np.array(data)
seperated = {}
for i in range(len(data)):
	row = data[i]
	if row[-1] not in seperated:			#if label not already present in the list then 
		seperated[row[-1]]=[]			#create a new index of that element and initialize a list
	seperated[row[-1]].append(row)			#append elements of each class found
print "\nThe yes values:\n",seperated['Yes']
print "\nThe no values:\n",seperated['No']

print "\n"
#find the number of rows and columns in the numpy array
yes_column_probs =[]
for column in zip(*seperated['Yes']):
	unique_val,count = np.unique(column,return_counts=True)
	#print unique_val,count
	for i in range(len(unique_val)):
		yes_column_probs.append([unique_val[i],count[i],count[i]/overall_count['Yes']])
	
print "\nYes counts and probabilities:",yes_column_probs

no_column_probs =[]
for column in zip(*seperated['No']):
	unique_val,count = np.unique(column,return_counts=True)
	#print unique_val,count
	for i in range(len(unique_val)):
		no_column_probs.append([unique_val[i],count[i],count[i]/overall_count['No']])
	
print "\nNo counts and probabilities:",no_column_probs


#if we find the test case for this: Age = <21, gender = female, income = low, marital status = married
#P(test|yes)*p(yes)
#this means  P(testcase|yes) = p(Age = <21 | yes)*p(gender = female | yes)*p(income = low | yes)*p(marital status = married | yes)
yes_test = 1
for i in range(len(yes_column_probs)):
	if yes_column_probs[i][0]== '<21':
		x = yes_column_probs[i][2]
		yes_test =yes_test*x
	if yes_column_probs[i][0]== 'Female':
		x = yes_column_probs[i][2]
		yes_test =yes_test*x
	if yes_column_probs[i][0]== 'Low':
		x = yes_column_probs[i][2]
		yes_test =yes_test*x
	if yes_column_probs[i][0]== 'Married':
		x = yes_column_probs[i][2]
		yes_test =yes_test*x
test_yes =  yes_test*(overall_count['Yes']/len(data))

#P(test|no)*p(no)
#this means  P(testcase|no) = p(Age = <21 | no)*p(gender = female | no)*p(income = low | no)*p(marital status = married | no)
yes_test = 1
no_test = 1
for i in range(len(no_column_probs)):
	if no_column_probs[i][0]== '<21':
		x = no_column_probs[i][2]
		no_test =no_test*x
	if no_column_probs[i][0]== 'Female':
		x = no_column_probs[i][2]
		no_test =no_test*x
	if no_column_probs[i][0]== 'Low':
		x = no_column_probs[i][2]
		no_test =no_test*x
	if no_column_probs[i][0]== 'Married':
		x = no_column_probs[i][2]
		no_test =no_test*x
test_no = no_test*(overall_count['No']/len(data))

print "For the testcase Age = <21, gender = female, income = low, marital status = married"
print "\nProbability of yes: ",test_yes,"\tProbability of no: ",test_no

if test_yes>test_no:
	print "Since yes probability is greater, person will buy item"
else:
	print "No probability is greater, person will NOT buy item"



