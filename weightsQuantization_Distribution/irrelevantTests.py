#!/usr/bin/env python
import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import math


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)


def Clip(x,maxima,minima):
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

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=16,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    K=MLP(args.unit, 10)
    model = L.Classifier(K)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)




    item1 = train[7][0]
    item1_label = train[7][1]
    print(item1_label)


    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    #trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    f= open("cjycjy_layer1.txt","w+")

    trainer.run()

    T1 = np.array(K.l1.W.data)
    T1_b = np.array(K.l1.b.data)
    T2 = np.array(K.l2.W.data)
    T2_b = np.array(K.l2.b.data)

    print("real weights of layer 1")
    print(T1)
    mask = T1<0
    print("real weights of layer 2")
    print(T2)
    mask2 = T2<0

    maxima1 = np.amax(T1)
    minima1 = np.amin(T1)
    maxima2 = np.amax(T1)
    minima2 = np.amin(T1)




    vRound = np.vectorize(Round)
    vClip = np.vectorize(Clip)
    quantized_weights_layer1 = vClip(vRound(np.log2(abs(T1))),maxima1,minima1)

    quantized_weights_layer2 = vClip(vRound(np.log2(abs(T2))),maxima2,minima2)


    antiQuanti1 = np.power(2.0,quantized_weights_layer1)


    antiQuanti2 = np.power(2.0,quantized_weights_layer2)




#    mask   antiQuanti1
    print("replace elements of mask")
    mask = mask*(-1)
    mask[mask>-1] = 1

    antiQuanti1 = antiQuanti1*mask
    print("antiQuant1: weights with inputs")
    print(antiQuanti1)


#    mask   antiQuanti2
    print("replace elements of mask 2")
    mask2 = mask2*(-1)
    mask2[mask2>-1] = 1
    antiQuanti2 = antiQuanti2*mask2



    layer1_out = np.dot(antiQuanti1,item1) + T1_b



#relu here




    print("layer1 out")
    print(layer1_out)
    layer1_out = F.relu(layer1_out)
    layer2_out = np.dot(antiQuanti2,layer1_out) + T2_b

    print("layer2 out")
    print(layer2_out)






"""
dot product will be able to compute the output of each layer

测试准确率：
把输入转化成numpy数组   做dot product, bias 得到的数就是这个节点的activation


def forward(self,inputs)

    x = _as_mat(inputs[0])
    W = inoputs[1]
    y = x.dot(W.T)
    if len(inputs) == 3:
        b = inputs[2]
        y += b
    return y
    
    
    
    
0809
找到问题了    被quantized量化了之后   原有的负号全部丢失了
"""




if __name__ == '__main__':
    main()


