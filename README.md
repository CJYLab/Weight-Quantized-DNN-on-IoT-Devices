# Weight Quantized DNN on IoT Devices


make sure the following modules are installed:

Chainer

numpy

matplotlib

multiprocessing

## Quantization: Before & After:


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



