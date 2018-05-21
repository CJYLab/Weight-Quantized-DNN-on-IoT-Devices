# Chainer Learning Notes


we will go through an official example to dive into Chainer source code.

The example we used here is the MNIST : https://github.com/chainer/chainer/tree/master/examples/mnist

Note when we search items from official documents, there are links which point to source code on github.

![source code pointer](images/source_link.png)

```python
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
```

