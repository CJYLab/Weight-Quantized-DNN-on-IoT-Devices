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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import collections
from collections import Counter
import seaborn as sns
from resize_training_input_neurons import projectionProfileMultiProcessVersion as pro

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
		return FSR-1
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


#for layer1:


quantized_weights_layer1 = vClip(vRound(np.log2(abs(weights_layer1))), maxima1, minima1)
print(quantized_weights_layer1)



#for layer2:

quantized_weights_layer2 = vClip(vRound(np.log2(abs(weights_layer2))), maxima2, minima2)
print(quantized_weights_layer2)
