import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Process

t1 = time.time()


train, test = chainer.datasets.get_mnist()


# train = train[0:1000]
# test = test[0:1000]
# reconstruct the (784,) array to shape of (28,28)
def reconstruct(Chainer_trainset):
    trainsetRecon = []
    for items in Chainer_trainset:
        mnArray = items[0]
        mnArray = mnArray.reshape((28, 28))
        mnLabel = items[1]
        intermi = (mnArray, mnLabel)
        trainsetRecon.append(intermi)
    return trainsetRecon


reshapedMNISTdataset = reconstruct(train)
reshapedMNISTdataset_t = reconstruct(test)


# Next we need to resize the (28,28) array to 49 number of (7,7) small array
# numpy offers methods to fetch rows and columns with ease
def reshapedMN_to_zones(reshapedMNISTdataset):
    # each item[0] is an (28,28) array
    zones = []
    labels = []
    for items in reshapedMNISTdataset:
        b = []
        for i in range(4):
            rows = 7
            cols = 7
            for k in range(4):
                a = np.empty((7, 7))
                a[0] = items[0][rows * i + 0][cols * k:cols * k + 7]
                a[1] = items[0][rows * i + 1][cols * k:cols * k + 7]
                a[2] = items[0][rows * i + 2][cols * k:cols * k + 7]
                a[3] = items[0][rows * i + 3][cols * k:cols * k + 7]
                a[4] = items[0][rows * i + 4][cols * k:cols * k + 7]
                a[5] = items[0][rows * i + 5][cols * k:cols * k + 7]
                a[6] = items[0][rows * i + 6][cols * k:cols * k + 7]
                b.append(a)

        zones.append(b)
        labels.append(items[1])

    return zones, labels


zones, labels = reshapedMN_to_zones(reshapedMNISTdataset)
zones_t, labels_t = reshapedMN_to_zones(reshapedMNISTdataset_t)

# zones = zones[0:3000]
# print(len(zones))
# print(len(zones))


"""
60000 items, they are 16 arrays with each of shape of (7,7)
this function aims to compute the projection profiles of each array
we will compute 4 projection profiles (horizontally, vertically, left diagnosly and right diagnosly).
For each array, we get 4 projection profiles, then we store peak values for each projection profile
"""

# f = open("resizedInputs_t","w+")

# zones have 60000 items
# each item has 16 groups(array)
def computeProjectionProfile(zones):
    fourProjectionProfile = []
    for subitem in zones:

        horizon = []
        for i in range(7):
            horizon.append(np.sum(subitem[i]))
        hori_max = max(horizon)
        fourProjectionProfile.append(hori_max)

        vertical = []
        for j in range(7):
            vertical = subitem.sum(axis=0)
        vert_max = max(vertical)
        fourProjectionProfile.append(vert_max)

        leftdiag = []
        for k in range(-6, 7):
            leftdiag.append(subitem.diagonal(k).sum())
        left_dia_max = max(leftdiag)
        fourProjectionProfile.append(left_dia_max)

        rightdiag = []
        for l in range(-6, 7):
            rightdiag.append(np.fliplr(subitem).diagonal(l).sum())
        right_dia_max = max(rightdiag)
        fourProjectionProfile.append(right_dia_max)
    return fourProjectionProfile

pool = mp.Pool()

resizedInputs = pool.map(computeProjectionProfile, zones)

resizedInputs_t = pool.map(computeProjectionProfile, zones_t)


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_out)
    #self.l3 = L.Linear(None, n_out)  # n_units -> n_out
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
    #h2 = F.relu(self.l2(h1))
        return self.l2(h1)




train_ = tuple_dataset.TupleDataset(np.array(resizedInputs, dtype = np.float32), np.array(labels, dtype = np.int32))
test_t = tuple_dataset.TupleDataset(np.array(resizedInputs_t, dtype = np.float32), np.array(labels_t, dtype = np.int32))




#print(train_[0])

print(train_.__getitem__(0))


model = L.Classifier(MLP(10, 10))


optimizer = chainer.optimizers.Adam()

optimizer.setup(model)


train_iter = chainer.iterators.SerialIterator(train_, 100)

test_iter = chainer.iterators.SerialIterator(test_t, 100, repeat=False, shuffle=False)

updater = training.updaters.StandardUpdater(train_iter, optimizer, device=-1)

trainer = training.Trainer(updater, (20, 'epoch'), out='result')

trainer.extend(extensions.Evaluator(test_iter, model, device=-1))

trainer.extend(extensions.LogReport())


trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.extend(extensions.ProgressBar())


trainer.run()

t2 = time.time()

print(t2-t1)