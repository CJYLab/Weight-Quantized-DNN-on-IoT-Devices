# Weight Quantized DNN on IoT Devices

## Update

This repo is outdated, please refer to [DLQ](https://github.com/CJYLab/Logarithmic-Quantization-of-Parameters-in-Neural-Networks) which contains one of the best quantization approach (simple yet powerful, even without retraining).

If you find our method helpful, please considering cite:

A Deep Look into Logarithmic Quantization of Model Parameters in Neural Networks, Jingyong Cai, Masashi Takemoto and Hironori Nakajo, Proceedings of The 10th International Conference on Advances in Information Technology (IAIT2018)

Workflow, methods used in this repository will be published as an IEEE GCCE2018 conference paper, citation will be updated later.

## Introduction:

Storing the weights in the flash then **read and compute piece by piece is extremely low-efficient**. As we know the processor inside hifive1 is quite powerful while, however, 16KB of RAM makes DNNs on this device not so feasible.

Our purpose is to **reduce the weights size greatly without affecting the accuracy.**

How this is possible? The answer is **logarithmic Quantization**. 

Say the original weights are 32bit float numbers, after quantization they are reduced to 3bit. in our case, we reduced the weights size of one hidden layer of MNIST from 50KB+ to 1KB- which make it possible to store whole matrices in the tiny RAM. 



## An algorithm for inputs reduction

![four direction projection profiles](doc/images/matrix_image0.jpg)
## Quantization: Before & After


Quantization in 3bits: how weights look like?
-----------------------
![before quantization](doc/images/before_quanti_3bits_50.png)
![after quantization](doc/images/after_quanti_3bits_50.png)   

## Results(Distribution-accuracy):

By default, hidden layer contains 50 neurons.

**4bits**

![4bits](doc/images/my_image_4bits_50.png)

**accuracy：** 0.887

**7bits:**

![7bits](doc/images/my_image_7bits_50.png)

**accuracy：** 0.929

**3bits**

![3bits](doc/images/my_image_3bits_50.png)

**accuracy：** 0.921
