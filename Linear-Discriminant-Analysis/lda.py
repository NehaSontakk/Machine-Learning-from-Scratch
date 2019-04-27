from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

#Read iris data
data = pd.read_csv('Iris.csv')
print data.head()

#Drop Id it's useless
data = data.drop(columns='Id')
print data.head()

#X values are columns 0-3
# values creates a numpy array
X = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
print "Numpy array with attributes:"
print 'Sample:\n',X[:5]

#Y has classes of each row in attributes
Y = data[['Species']].values
print 'Sample:\n',Y[:5]

#Iris has 3 classes and 4 attributes of each class 
#the last column has to have numerical classes

lbl = LabelEncoder()
Y = lbl.fit_transform(Y)
print "Classes after encoding: \n",Y

#LDA START

#STEP 1: Find mean vectors of each attribute for each class
mean_array = []
for i in np.unique(Y):
	mean_array.append(np.mean(X[Y==i],axis=0))
print "Mean array:",mean_array
print "For class 1 means are: ",mean_array[0]
print "For class 2 means are: ",mean_array[1]
print "For class 3 means are: ",mean_array[2]


#STEP 2: Find Scatter Matrix
#Too many classes to calculate for covariance so using scatter matrix
#Within class scatter matrix is 
print "\nBegin Step 2:\n"
scatter_within_class = np.zeros((4,4))
#For each class in our data
for classes in range(len(np.unique(Y))):
	print "Class:",classes
	print "Mean array for current class:",mean_array[classes]
	#Take the mean vector for that class
	#matrix for each class
	each_class_sc = np.zeros((4,4))
	#For a row of data belonging to current class
	for row in X[Y==classes]:
		#make sure both row in x and mean_vec are same size
		row = row.reshape(4,1)
		mean_array[classes] = mean_array[classes].reshape(4,1)
		#within class scatter matrix
		each_class_sc += (row-mean_array[classes]).dot((row-mean_array[classes]).T)
	#add it to scatter for each class
	scatter_within_class += each_class_sc

print "\nScatter matrix within class: \n",scatter_within_class



#STEP 3: Find between class scatter matrix

#Find the mean for everything
all_mean = np.mean(X,axis=0)
print "Overall mean for every class:",all_mean

#between class scatter
scatter_between_class = np.zeros((4,4))
for classes in np.unique(Y):
	N = len(X[Y==classes])
	print "\nClass number:",classes," size:",N
	mean_array[classes] = mean_array[classes].reshape(4,1)
	all_mean = all_mean.reshape(4,1)
	print "Mean for current class:",mean_array[classes]
	scatter_between_class += N*(mean_array[classes]-all_mean).dot((mean_array[classes]-all_mean).T)

print "\nScatter Matrix in between classes:\n",scatter_between_class


#STEP 4: Eigan Values and Vectors
#We need to use a generalized eigan values and vector formula for this as we have multiple dimensions and classes
#The matrix we use is : inverse of scatter matrix within class * scatter matrix between classes
eigan_val,eigan_vec = np.linalg.eig(np.linalg.inv(scatter_within_class).dot(scatter_between_class))
  
for i in range(len(eigan_val)):
	eigvecs = eigan_vec[:,i].reshape(4,1)  
	print "\nEigenvalue",i,eigan_val[i]
	print "Eigenvector",i,eigvecs


#STEP 5: Check if we're correct till now
# Av = lambda(v) i.e. inverse(Sw)*Sb*eigan_vectors == eigan_value*eigan_vectors

for i in range(len(eigan_val)):
	eigvecs = eigan_vec[:,i].reshape(4,1) 
	lhs = (np.linalg.inv(scatter_within_class).dot(scatter_between_class))*eigvecs[i]
	rhs = eigan_val[i]*eigvecs[i]
	if lhs.all()==rhs.all():
		print "All cool for ",i,", proceed."


#STEP 6: Sort Eigan values in decreasing order and then sort eigan vectors 
#We actually start reducing our dimensions here

#new list of pairs
eigan_pair = []
for i in range(len(eigan_val)):
	eigan_pair.append([eigan_val[i],eigan_vec[:,i]])

#print eigan_pair

sorted_eigan_pair = sorted(eigan_pair,reverse=True)
print "\nSorted Vals: \n",sorted_eigan_pair

#STEP 7: Find which eigan values matter the most
sum_eigan_val = sum(eigan_val)
for i in range(len(eigan_pair)):
	print "\nEigan value",eigan_pair[i][0]," accounts for ",float(eigan_pair[i][0]/sum_eigan_val*100)," of the variance."

###### We can see that last two barely matter in our data set

#STEP 8: We consruct the final matrix that will transform our data
#Column 1 : eigan vector 1 and Column 2: eigan vector 2
Final_Matrix = np.hstack((eigan_pair[0][1].reshape(4,1), eigan_pair[1][1].reshape(4,1)))
print '\nFinal Matrix:\n', Final_Matrix

#STEP 9: New Coordinates
Xlda = X.dot(Final_Matrix)


#STEP 10: Plot Graph
import matplotlib.pyplot as plt

color1 = ['r','g','b']
for i in range(len(np.unique(Y))):
	plt.scatter(x=Xlda[:,0].real[Y==i],y=Xlda[:,1].real[Y==i],color = color1[i])
plt.xlabel('LD1')
plt.ylabel('LD2')

plt.show()

#Conclusion
#We have reduced data and seperated it in two dimensions
#LD1 seperates classes well

