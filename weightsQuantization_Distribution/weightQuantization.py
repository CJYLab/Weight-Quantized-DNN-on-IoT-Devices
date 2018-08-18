"""
Author: CAI JINGYONG @ BeatCraft, Inc & Tokyo University of Agriculture and Technology

This scripts is not yet generalized.

Here we give general algorithm of log quantization, for details please refer to paper:
Convolutional Neural Networks using Logarithmic Data Representation
url:https://arxiv.org/pdf/1603.01025.pdf

parameters:
	FSR: full scale range, equals to log2(max-min), max and min are in linear space
	bitwidth
	x: input


LogQuant(x,bitwidth,FSR)= 0         x=0,
			  2^x'      otherwise.

where

x'=Clip(Round(log2|x|),FSR-2^bitwidth,FSR),

		Clip(x,min,max)= 0        x<=min,
				 max-1    x>=max,
				 x	  otherwise.
Round(x)

	for Round() function, the direction of round is decided by the fractional part of the parameter.
	say fractional part is F,
		ROUND UP if F >= math.sqrt(2)-1
		else
		ROUND DOWN

"""
from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('../projectionProfileAlgorithm')
import projectionProfileMultiProcessVersion as pro

pro.dataPreparation()
pro.trainMnist()

chain = pro.chain_mnist

weights_layer1 = np.array(chain.l1.W.data)
biases_layer1 = np.array(chain.l1.b.data)
weights_layer2 = np.array(chain.l2.W.data)
biases_layer2 = np.array(chain.l2.b.data)

maxima1 = np.amax(weights_layer1)
minima1 = np.amin(weights_layer1)
maxima2 = np.amax(weights_layer2)
minima2 = np.amin(weights_layer2)

def Clip(x, maxima, minima):
	FSR = maxima - minima
	MIN = FSR-(2**4)
	if(x <= MIN):
		return MIN
	elif(x >= FSR):
		return FSR - 1
	else:
		return x


def Round(num):
	bridge = math.sqrt(2)-1
	decimalPart, intPart = np.modf(num)
	if decimalPart >= bridge:
		return math.ceil(num)
	else:
		return math.floor(num)

vRound = np.vectorize(Round)
vClip = np.vectorize(Clip)

def outputWeightsArrayFile(weights):
    # check whether the parameter is numpy, add laterly
    for k, v in list(locals().iteritems()):
        if v is weights:
            a_as_str = k
    filename = a_as_str + '_file.txt'
    arrayname = a_as_str + 'quantized_weights_layer1_array'
    rows = weights.shape[0]
    cols = weights.shape[1]
    with open(filename,'w+') as f:
        f.write("int " + arrayname + "[" + str(rows) + "]" + "[" + str(cols) + "] = {")
        for index, _ in enumerate(weights[:,]):
            f.write("{")
            for i in _[0:-1]:
                f.write(str(i.item()))
                f.write(", ")
            f.write(str(_[-1].item()))
            if(index == (rows - 1)):
                f.write("}")
            else:
                f.write("}, ")
        f.write("}")
    f.close()

#mask (sign bits of the original weights, these informatin will get lost after quantization)
def signBitsMaskArray(layer):
	mask  = layer < 0
	mask = mask * (-1)
	mask[mask > -1] = 1
	return mask

#for layer1:


quantized_weights_layer1 = vClip(vRound(np.log2(abs(weights_layer1))), maxima1, minima1)
layer1_mask = signBitsMaskArray(weights_layer1)
anti_quantized_weights_layer1 = np.power(2.0,quantized_weights_layer1)
anti_quantized_weights_layer1 = anti_quantized_weights_layer1 * layer1_mask



#for layer2:

quantized_weights_layer2 = vClip(vRound(np.log2(abs(weights_layer2))), maxima2, minima2)
layer2_mask = signBitsMaskArray(weights_layer2)
anti_quantized_weights_layer2 = np.power(2.0, quantized_weights_layer2)
anti_quantized_weights_layer2 = anti_quantized_weights_layer2 * layer2_mask


# print("layer1 mask")
# print(layer1_mask)
#
# print("weights_layer1")
# print(weights_layer1)
#
# print("anti_quantized_weights_layer1")
# print(anti_quantized_weights_layer1)


randint = np.random.randint(0,10000,size=1000)

resizedInputs_t = np.array(pro.resizedInputs_t, dtype=np.float32)
labels_t = np.array(pro.labels_t, dtype=np.int32)
test_items = resizedInputs_t[randint]
test_items_labels = labels_t[randint]


#accuracy tests


#forward computation
count = 0
for index, inputs in enumerate(test_items):
	layer1_out = np.dot(anti_quantized_weights_layer1, inputs) + biases_layer1
	#relu here
	layer1_out[layer1_out < 0] = 0
	layer2_out = np.dot(anti_quantized_weights_layer2,layer1_out) + biases_layer2
	output = np.argmax(layer2_out)
	#print(int(output))
	#print(int(test_items_labels[index]))
	#print("----------")
	if int(output) == int(test_items_labels[index]):
		count = count+1

accuracy = count/1000
print("--------------accuracy--------------")
print(accuracy)
print("--------------accuracy--------------")

#visualization


# print("weights of layer1, before quantization")
# print(weights_layer1)
#
# print("weights of layer1, after quantization")
# print(anti_quantized_weights_layer1)
#
# print("weights of layer2, before quantization")
# print(weights_layer2)
#
# print("weights of layer2, after quantization")
# print(anti_quantized_weights_layer2)

figure = plt.figure()

plt.subplot(2,1,1)

#compoute the desity of the weights
density_x = np.linspace(round(minima1,2),round(maxima1,2),100)
y, x = np.histogram(weights_layer1, bins=density_x)
x = x + (maxima1-minima1)/200
plt.plot(x[:-1], y,'k', linewidth=1)
plt.scatter(x[:-1], y, s=5, c='k', label='Weights count within intervals')
#compute each interval size
interval = round((maxima1-minima1)/100, 3)
plt.ylabel('density: count within {} interval'.format(interval))
plt.xlabel('value of weights')
ticks_to_be_showed = []

for _ in range(100):
	ticks_to_be_showed.append('')

for _ in range(0,100,20):
	ticks_to_be_showed[_] = round(density_x[_],2)
ticks_to_be_showed[99] = round(density_x[99],2)
#print(ticks_to_be_showed)
# x_label = np.linspace(round(minima1,2),round(maxima1,2),100)


plt.xticks(density_x, ticks_to_be_showed)
plt.legend(loc='best')

plt.subplot(2,1,2)
unique, counts_ = np.unique(anti_quantized_weights_layer1, return_counts= True)
for index, counts in enumerate(unique):
	plt.plot([unique[index],unique[index]], [0, counts_[index]],'k-',linewidth=1.5)
plt.plot([], [], c='k', label='Count of quantized weights')



plt.legend()
randint2 = 16*np.random.random_sample(size=1024)
#plt.scatter(weights_layer1,randint2,alpha=0.4,s=10)
plt.ylabel('count')
plt.xlabel('value of weights(log quantized)')
#plt.grid(True)




plt.tight_layout()
#plt.show()
figure.savefig('my_image.svg', format='svg', dpi=1200)
figure.savefig('my_image_png.png')
