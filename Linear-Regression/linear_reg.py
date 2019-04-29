def mean(values):							#Find the mean for x and y
	return sum(values)/float(len(values))

def standard_dev(values,mean):
	step1 = sum([(x-mean)**2 for x in values])			#sum all values
	step2 = step1/(len(values)-1)					#divide sum by number of 'x' or 'y' values
	step3 = step2**0.5						#whole root
	return step3
	
def covariance(x,x_mean,y,y_mean):					#Covariance gives relative variance of x and y
	covar1 = 0.0
	for i in range(len(x)):
		covar1 += (x[i]-x_mean)*(y[i]-y_mean)
	covar2 = covar1/(len(x)-1)
	return covar2

#Find the correlation between our x and y values.
#This gives the strength of the relation between the variables.
#Using Pearson's correlation formula

def correlation(covariance,std_dev_x,std_dev_y):
	cor = covariance/(std_dev_x*std_dev_y)
	return cor	
	
#Find the coefficients of the regression line
#Formula of the line : y = b0 + b1(x)
# b1 can be written as covariance divided by variance of both variables
# variance is square of standard deviation
# b0 is just using the line equation and replacing y and x with it's mean and b1 with b1

def line_coeff(dataset):
	x=[row[0] for row in dataset]
	y=[row[1] for row in dataset]
	x_m,y_m = mean(x),mean(y)
	b1 = covariance(x,x_mean,y,y_mean)/(standard_dev(x,x_mean)*standard_dev(y,y_mean))
	b0 = y_m - b1*x_m
	return [b0,b1]


# Simple Linear Regression Algorithm
# Algorithm Function
def simple_linear_reg(train,test):
	predictions = list()						#List to save predicted values of y
	b0,b1 = line_coeff(train)					#use training data to find our coefficients
	for row in test:
		y_new = b0 + b1*row[0]					#row[0] are our x values
		predictions.append(y_new)
	return predictions

from math import sqrt
# Root mean squared error gives us the error between the actual and predicted values of y
def rmse(actual,predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]		#the difference between actual y and predicted y
		sum_error += (prediction_error ** 2)			#square it
	mean_error = sum_error / float(len(actual))			#mean of the error values
	return sqrt(mean_error)

#evaluate values using simple_linear_reg function on the dummy dataset
def evaluate_algorithm(dataset):
	test_set = list()						#Create a list of test values
	for row in dataset:
		row_copy = list(row)					#Keep the x values and delete y values from the dataset
		row_copy[-1] = None
		test_set.append(row_copy)				#That's the test data
	predicted = simple_linear_reg(dataset, test_set)		#Call the algorithm function to predict new y values
	actual = [row[-1] for row in dataset]				#The actual values are the y values
	error=rmse(actual,predicted)					#find the error using root mean squared method
	print("Root mean squared error: ",error)
	print("Predicted values: ",predicted)



#This part is just to test the math in the functions on a dummy dataset 

dataset = [[34,5],[108,17],[64,11],[88,8],[99,14],[51,5]]
x=[row[0] for row in dataset]
y=[row[1] for row in dataset]
x_mean = mean(x)
print("Mean of 'x' values:",x_mean)
y_mean = mean(y)
print("Mean of 'y' values:",y_mean)
cov = covariance(x,x_mean,y,y_mean)
print("Covariance is: ",cov)
std_dev_x = standard_dev(x,x_mean)
print("Standard deviation of 'x': ",std_dev_x)
std_dev_y = standard_dev(y,y_mean)
print("Standard deviation of 'y': ",std_dev_y)
correl = correlation(cov,std_dev_x,std_dev_y)
print("Pearson Correlation(greater than 0.5 is good enough): ",correl)

coeff = line_coeff(dataset)
print("Coefficients b0 and b1 of regression line: ",coeff)


#Check everything
#the root mean squared error gives us an idea of how accurate we are
evaluate_algorithm(dataset)





"""
("Mean of 'x' values:", 74.0)
("Mean of 'y' values:", 10.0)
('Covariance is: ', 123.0)
("Standard deviation of 'x': ", 29.003448070875987)
("Standard deviation of 'y': ", 4.898979485566356)
('Pearson Correlation(greater than 0.5 is good enough): ', 0.865664999629448)
('Coefficients b0 and b1 of regression line: ', [-54.059209972579154, 0.865664999629448])
('Root mean squared error: ', 19.179446683258696)
('Predicted values: ', [-24.626599985177922, 39.43260998740122, 1.3433500037055168, 22.11930999481227, 31.64162499073619, -9.910294991477308])

"""
