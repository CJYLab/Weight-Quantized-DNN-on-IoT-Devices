In the cifar10 example inside the Chainer, architecture of the network is defined in [VGG.py](https://github.com/chainer/chainer/blob/master/examples/cifar/models/VGG.py).

Python basics:

### Context Manager

A splendid full version tutorial is here: [Pthon with Context Managers](https://jeffknupp.com/blog/2016/03/07/python-with-context-managers/).

Any object that needs to have a close function called on it after use is (or should be) a context manager. 

`contextlib`contains tools for creating and working with context managers. One nice shortcut to creating a context manager from a class is to use the `@contextmanager` decorator. To use it, decorate a generator function that calls `yield` exactly *once*. Everything *before* the call to `yield` is considered the code for `__enter__()`. Everything after is the code for `__exit__()`. 

The context manager: init_scope().

```python
 @contextlib.contextmanager
 def init_scope(self):
    old_flag = self.within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            self._within_init_scope = old_flag
```

