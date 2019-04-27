import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from math import sqrt
from collections import Counter
style.use('fivethirtyeight')

dataset = {'y':[[2,4],[4,6],[4,2],[6,4]], 'n':[[4,4],[6,2]]}
new_point = [6,6]


for i in dataset:
	for j in dataset[i]:
		plt.scatter(j[0],j[1],s=100,color='red')


plt.scatter(new_point[0],new_point[1],s=100,color='green')
plt.show()

def k_nn(data,predict,k=3):
	if(len(data))>=k:
		print("error")
	
	distance=[]
	for group in data:
		for features in data[group]:
			euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
           		print(euclidean_distance)
           		distance.append([euclidean_distance,group])
        votes = [i[1] for i in sorted(distance)[:k]]	#for i[1] ie group in the list sorted by distances till range k
        print votes
        vote_result = Counter(votes).most_common(1)[0][0]
        return vote_result
			
			
			
result = k_nn(dataset, new_point)
print(result)
