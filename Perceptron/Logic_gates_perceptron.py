import numpy as np

#Logic gate 3 input And using perceptron

inputs = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0,1,1],[1,0,1],[1,1,1]])

print "Inputs:",inputs

#Choose data
choice = input("Choose between \n1) or \n2) and \n3) nor \n4) nand")

if choice == 1:
	outputs = np.array([0,1,1,1,1,1,1]).reshape(-1,1)
elif choice == 2:
	outputs = np.array([0,0,0,0,0,0,1]).reshape(-1,1)
elif choice == 3:
	outputs = np.array([1,0,0,0,0,0,0]).reshape(-1,1)
elif choice == 4:
	outputs = np.array([1,1,1,1,1,1,0]).reshape(-1,1)
	


#outputs = np.array([0,0,0,0,0,0,1]).reshape(-1,1)
inputs_shape = inputs.shape

class perceptron:
	def __init__(self):
		#initialize weights by some random number
		#weights are to be multiplied with each row so the shape of the array = the number of columns*1 array
		self.weights = np.random.rand(inputs_shape[1],1)
	
	#sigmoid function is the activation function used
	#input is an input value
	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
		
		
	#the partial derivative of the sigmoid function is used to calculate and readjust the weights
	#used during back propagation
	def sig_der(self,x):
		return np.exp(-x)/((1+np.exp(-x))**2)
		
		
	#Train Function:
	def train(self,input_values,expected_output,learning_rate,iterations):
		#Inputs: input_values = values used to train
		# expected_output = real ouputs in the training phase i.e. supervised learning during backpropagation
		#learning_rate = rate at which the weights are updated
		#iterations = number of iterations to keep updating weights
		
		#This is going to store all the changed weight derivatives
		delta_weights = np.zeros((input_values.shape[1],input_values.shape[0]))
		
		for iteration in range(iterations):
			
			#Forward Pass
			
			#Z stores the dot products of all the input rows multiplied by weights
			z = np.dot(input_values,self.weights)
			#Activation decides where the sigmoid value of z is in range 0 to 1
			activation = self.sigmoid(z)
			#print ("z",z," Activaton",activation)
		
			#Backward Pass
			
			for row in range(input_values.shape[0]):
				
				#Find the deviation of found values from real/expected values
				cost_derivation = 2*(activation[row]-expected_output[row])
				#find the partial derivative of z
				z_partial_derv = self.sig_der(z[row])
				
				for column in range(input_values.shape[1]):
					#Update the delta weights array
					delta_weights[column][row] = cost_derivation*input_values[row][column]*z_partial_derv
				
				#Find average delta of all the mean values found
				delta_average = np.array([np.mean(delta_weights,axis=1)]).reshape(-1,1)
				
				#Update the weights with the average value subtracted
				#Learning rate enhances the impact of subtraction
				self.weights = self.weights - delta_average*learning_rate
		
	#Result function is to test a new case on these optimized weights			
	def test(self,testcase):
		return self.sigmoid(np.dot(testcase,self.weights))
		
		
#Class object		
p = perceptron()
print "Weights: \t",p.weights
p.train(inputs,outputs,10,1000)

#alter to give any other test case 
testcase = np.array([0,0,1])	
print "Testcase sigmoid output",p.test(testcase)	

#if sigmoid output is close to 1 then the output is 1
#threshold set to 0.5
x = float(p.test(testcase))
if x <= 0.5:
	print "Final output for testcase",testcase.reshape(1,-1),"is 0"	
elif x > 0.5:
	print "Final output for testcase",testcase.reshape(1,-1),"is 1"
