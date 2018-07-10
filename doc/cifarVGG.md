In the cifar10 example inside the Chainer, architecture of the network is defined in [VGG.py](https://github.com/chainer/chainer/blob/master/examples/cifar/models/VGG.py).

Python basics:

### Context Manager

A fantastic full version tutorial is here: [Pthon with Context Managers](https://jeffknupp.com/blog/2016/03/07/python-with-context-managers/).

> Any object that needs to have a close function called on it after use is (or should be) a context manager. 
>
> `contextlib`contains tools for creating and working with context managers. One nice shortcut to creating a context manager from a class is to use the `@contextmanager` decorator. To use it, decorate a generator function that calls `yield` exactly *once*. Everything *before* the call to `yield` is considered the code for `__enter__()`. Everything after is the code for `__exit__()`. 

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

we open a file, yield it, then close it.

> In a nutshell, the goal of context manager: to make working with resources and creating managed contexts easier. 

### @ Symbol in Python (Decorators)

@ symbol is called decorator in Python language. In the piece of code above you might feel confuse about ` @contextlib.contextmanager`, here we give explanation of decorator in Python(note decorator means differently among languages):

>  Decorators allow you to inject or modify code in functions or classes, the @ indicates the application  of the decorator(from [Bruce Eckel ](https://www.artima.com/weblogs/viewpost.jsp?thread=240808)).

static methods were added into Python 2.2,  to realize it we need:

```python
class MyClass(object):
	def staticFoo():
		:
	staticFoo = staticmethod(staticFoo)
	: 
```

Here we want the staticFoo() becomes a static method therefore we eliminate the `self` parameter. However, it seems so redundant to use:

```python
staticFoo = staticmethod (sta- ticFoo)
```

Instead we use decorator to make it simple.

```python
class MyClass(object):
	@staticmethod
	def staticFoo():
		: 
```

While, I still don't get the idea，，，***<u>so what on earth is decorator?</u>***

> By definition, a decorator is a function that takes another function and extends the behavior of the latter function *without* explicitly modifying it. 

An article explain decorators well: [Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/#decorators). 

Copy the best example to here:

```python
def my_decorator(some_function):
    def wrapper():
        num = 10
        if num == 10:
            print("Yes!")
        else:
            print("No!")

        some_function()

        print("Something is happening after some_function() is called.")

#notice here we don't use wapper() so that way we can use them in the future.
    return wrapper

def just_some_function():
    print("Wheee!")

just_some_function = my_decorator(just_some_function)

just_some_function()
```

output:

```
Yes!
Wheee!
Something is happening after some_function() is called.
```

How @ functions？  Compare the following piece of code.

```python
def my_decorator(some_function):
    def wrapper():
        num = 10
        if num == 10:
            print("Yes!")
        else:
            print("No!")

        some_function()

        print("Something is happening after some_function() is called.")

    return wrapper

if __name__ == "__main__":
    my_decorator()
```

Call the function with the decorator:

```python
from decorator07 import my_decorator

@my_decorator
def just_some_function():
    print("Wheee!")

just_some_function()
```

output:

```
Yes!
Wheee!
Something is happening after some_function() is called.
```

So, `@my_decorator` is just an easier way of saying 

`just_some_function = my_decorator(just_some_function)`. 

It’s how you apply a decorator to a function. 