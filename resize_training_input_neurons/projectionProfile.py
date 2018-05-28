import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np

train, test = chainer.datasets.get_mnist()



#reconstruct the (784,) array to shape of (28,28)
def reconstruct(Chainer_trainset):

	trainsetRecon = []
	for items in Chainer_trainset:
		mnArray = items[0]
		mnArray = mnArray.reshape((28,28))
		mnLabel = items[1]
		intermi = (mnArray,mnLabel)
		trainsetRecon.append(intermi)
	return trainsetRecon

		


reshapedMNISTdataset = reconstruct(train)

# Next we need to resize the (28,28) array to 49 number of (7,7) small array
# numpy offers methods to fetch rows and columns with ease


def reshapedMN_to_zones(reshapedMNISTdataset):

      	#each item[0] is an (28,28) array
	zones = []
	labels = []
	for items in reshapedMNISTdataset:
		b = []
		for i in range(4):
			rows=7
			cols=7
			for k in range(4):
				a = np.empty((7,7))
				a[0] = items[0][rows*i+0][cols*k:cols*k+7]
				a[1] = items[0][rows*i+1][cols*k:cols*k+7]
				a[2] = items[0][rows*i+2][cols*k:cols*k+7]
				a[3] = items[0][rows*i+3][cols*k:cols*k+7]
				a[4] = items[0][rows*i+4][cols*k:cols*k+7]
				a[5] = items[0][rows*i+5][cols*k:cols*k+7]
				a[6] = items[0][rows*i+6][cols*k:cols*k+7]
				b.append(a)
		
		zones.append(b)
		labels.append(items[1])

	return zones, labels


zones, labels = reshapedMN_to_zones(reshapedMNISTdataset)

print(len(zones))



"""
60000 items, they are 16 arrays with each of shape of (7,7)
this function aims to compute the projection profiles of each array
we will compute 4 projection profiles (horizontally, vertically, left diagnosly and right diagnosly).
For each array, we get 4 projection profiles, then we store peak values for each projection profile
"""	


#zones have 60000 items
#for each item,  
def computeProjectionProfile(zones):
	for items in zones:
		#print(len(items))
		
		for subitem in items:

			horizon = []
			for i in range(7):
				#print(subitem[i])
				#break
				horizon.append(np.sum(subitem[i]))
			hori_max = max(horizon)


			vertical = []
			for j in range(7):
				vertical = subitem.sum(axis=0)
			vert_max = max(vertical)



computeProjectionProfile(zones)








































