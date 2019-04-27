
#We have two dimensions of data 
x1 = [1.40,1.60,-1.40,-2.00,-3.00,2.40,1.50,2.30,-3.20,-4.10]
x2 = [1.65,1.97,-1.77,-2.52,-3.95,3.07,2.02,2.75,-4.05,-4.85]

#Vizualize it
import matplotlib.pyplot as plt
plt.scatter(x1,x2)
plt.grid()
plt.show()

#STEP 1: COVARIANCE MATRIX
#Calculate mean of x1 and x2
#Calculate variance of x1 and x2
#Calculate covariance of x1 and x2

def mean(values):
	num1 = sum(values)/len(values)
	return num1

x1_mean = float(mean(x1))
x2_mean = float(mean(x2))

print "Mean of x1 and x2: ",x1_mean,x2_mean

def variance(values):
	num1=0.0
	value_mean = mean(values)
	for i in range(len(values)):
		num1 += (values[i]-value_mean)**2
	num2 = num1/len(values)
	return num2

x1_var = variance(x1)
x2_var = variance(x2)

print "Variance of x1 and x2: ",x1_var,x2_var

def covariance(value1,value2):
	value1_mean = mean(value1)
	value2_mean = mean(value2)
	cov1=0.0
	for i in range(len(value1)):
		cov1 += (value1[i]-value1_mean)*(value2[i]-value2_mean)
	cov2 = cov1/(len(value1)-1)
	return cov2

covarx1_x2 = covariance(x1,x2)
print "Covariance of x1 and x2: ",covarx1_x2

#Covariance matrix :
#  var1		covar1,2
#  covar1,2	var2
cov_mat = [[[],[]],[[],[]]]
cov_mat[0][0] = x1_var
cov_mat[0][1] = covarx1_x2
cov_mat[1][0] = covarx1_x2
cov_mat[1][1] = x2_var

print "Covariance matrix:",cov_mat

#STEP 2: EIGAN VALUES
#Solve equation A-(lambda(L))I=0
#I = [[[1],[0]],[[0],[1]]]
#After solving equation L^2 - L*(var1+var2) + [var1*var2 - covar1,2^2].........................(1)
# i.e. ax^2 + bx + c
# x = [-b + or - sqrt(b^2-4ac)] / 2a
a = 1
b = -(x1_var+x2_var)
c = x1_var*x2_var - covarx1_x2**2
print a,"L^2+",b,"L+",c
import math
L1 = (-b + math.sqrt(b**2 - 4*a*c))/2*a
L2 = (-b - math.sqrt(b**2 - 4*a*c))/2*a
print "Eigan Values: ",L1,L2

#Check if we're correct till now
if L1+L2 == x1_var+x2_var:
	print "Checked if lambda1 +2 = var1+var2..correct...proceed ..."
else:
	print "Wrong."

#STEP 3: EIGAN VECTORS
#(A-(L1 then L2)I)(X) = 0
#X = [a,b]
#put L1 and L2 in equation 1
#use numpy to find eigan vectors and crosscheck eigan values
import numpy as np
check_eigans = np.linalg.eig(cov_mat)
print "Eigan vectors and values:",check_eigans
print "Eigan vectors for lambda1/L1\n"
X1 = check_eigans[1][0]
print X1
print "\nEigan vectors for lambda2/L2\n"
X2 = check_eigans[1][1]
print X2

#STEP 4: FIND NEW COORDINATES
#center data of x1
centered_x1 = []
for i in range(len(x1)):
	centered_x1.append(x1[i]-x1_mean)
print "Cenetered x1:",centered_x1
#center data of x2
centered_x2 = []
for i in range(len(x2)):
	centered_x2.append(x2[i]-x2_mean)
print "Cenetered x1:",centered_x2
#multiply
new_x1=[]
new_x2=[]
for i in range(len(x1)):
	new_x1.append(x1[i]*X1[0]+x2[i]*X1[1])
	new_x2.append(x1[i]*X2[0]+x2[i]*X2[1])

print "New x1: ",new_x1
print "New x2: ",new_x2



#STEP 5: FIND AND PLOT PC1 AND PC2
#Plot lines PC1 and PC2
#PC1 gives us the direction of most variance in the data about 83% of variance is captured here
#PC2 is just perpendicular to PC1 and captures more variance if we woud have had more dimensions
new_y1=[]
for i in range(len(new_x1)):
	new_y1.append(new_x1[i]*(X1[1]/X1[0]))

new_y2=[]
for i in range(len(new_x2)):
	new_y2.append(new_x2[i]*(X2[1]/X2[0]))
	

plt.scatter(new_x1,new_x2)
plt.plot(new_x1,new_y1,'r-',label='PC1')
plt.plot(new_x2,new_y2,'y-',label='PC2')
plt.grid()
plt.legend()
plt.show()

