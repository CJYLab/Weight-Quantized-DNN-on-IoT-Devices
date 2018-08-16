# Weight Quantized DNN on IoT Devices



Workflow, methods used in this repository will be published as an IEEE GCCE2018 conference paper, citation will be updated later.

## Introduction:

Storing the weights in the flash then **read and compute piece by piece is extremely low-efficient**. As we know the processor inside hifive1 is quite powerful while, however, 16KB of RAM makes DNNs on this device not so feasible.

Our purpose is to **reduce the weights size greatly without affecting the accuracy.**

How this is possible? The answer is **logarithmic Quantization**. 

Say the original weights are 32bit float numbers, after quantization they are reduced to 3bit. in our case, we reduced the weights size of one hidden layer of MNIST from 50KB+ to 1KB- which make it possible to store whole matrices in the tiny RAM. 



## An algorithm for inputs reduction

![four direction projection profiles](doc/images/matrix_image.jpg)
## Quantization: Before & After


Quantization in 7bits:
-----------------------
before

[ 0.125      -0.5        -2.         -0.5        -0.25       -0.03125
  -0.03125     0.5        -0.5         0.125      -0.125      -1.
   0.5         0.125       0.0625      0.0078125 ]

after

[0.21458377 -0.7497542  -1.5595171  -0.9316343  -0.27410102 -0.04780952
  -0.04013866  0.65799063 -0.8927019   0.13269071 -0.17763455 -1.010546
   0.6517413   0.20199347  0.11478033  0.01444888]
   
4bits:

3bits:

## Accuracy:



