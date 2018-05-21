import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import collections
from collections import Counter
import seaborn as sns 


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

#oneDemArray=np.array([])
repFile=open("cjyreplaced","r")

Maxima=0
Minima=0


#print(repFile.tell())
for line in repFile:
	content=line.split()
	#print(content)
	for numbers in content:
		numbers=float(numbers)
		if(numbers>Maxima):
			Maxima=numbers
		elif(numbers<Minima):
			Minima=numbers
		else:
			doNothing=0

#print(repFile.tell())

FSR=Maxima-Minima


#print(2**(FSR-1))
MIN=FSR-(2**4)



def Clip(x):
	if(x<=MIN):
		return MIN
	elif(x>=FSR):
		return FSR-1
	else:
		return x

bridge=math.sqrt(2)-1

def Round(num):
	intPart=int(num)
	decimalPart=num-intPart
	if(decimalPart>=bridge):
		return math.ceil(num)
	else:
		return math.floor(num)

QuanWeights = []

repFile1=open("cjyreplaced_dom","r")

repFile2=open("quantified_data","w+")

#print(repFile1.tell())
for line2 in repFile1:
	content2=line2.split()
	for numbers2 in content2:
		numbers2=float(numbers2)
		Quantee=Clip(Round(math.log(abs(numbers2))))
		if Quantee==0:
			QuanWeights.append(float(0))
		else:
			QuanWeights.append(2**Quantee)
			repFile2.write(str(2**Quantee))
#a counter is a dict subclass for counting hashable objects.

outCount=Counter(QuanWeights)



sortedOutCount = collections.OrderedDict(sorted(outCount.items()))




wKeys=np.array(sortedOutCount.keys())
wValues=np.array(sortedOutCount.values())



#print(QuanWeights)
#print(outCount)

#sns.factorplot(outCount, rug=True)
plt.plot(wKeys,wValues,label='Frist line',color='r') 

#plt.hist(od, bins=50, color='steelblue', normed=True)


plt.bar(wKeys,wValues,width=0.001,align='edge')
plt.xlabel('Log Quantified data')    
plt.ylabel('Count')

#plt.yticks(wValues)
#print(type(wKeys))
#print(len(QuanWeights))

plt.show()
#plt.savefig('sns')
#print(Clip(1))
#print(Round(1.415))

