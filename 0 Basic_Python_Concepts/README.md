# Understanding Basic Python Concepts

This module covers the fundamental concepts of Python programming, including data types, data structures, functions, and error handling. By the end of this module, you should have a solid understanding of the basics of Python and be able to write simple programs.

## Topics Covered

- [Data Types](#data-types)
- [Data Structures](#data-structures)
- [Functions](#functions)
- [Error Handling](#error-handling)

## Data types

Python has several built-in data types, including:

- **Numeric Types**: `int`, `float`, and `complex`
- **Sequence Types**: `list`, `tuple`, and `range`
- **Text Type**: `str`
- **Mapping Type**: `dict`
- **Set Types**: `set` and `frozenset`
- **Boolean Type**: `bool`
- **Binary Types**: `bytes`, `bytearray`, and `memoryview`

## Data Structures

Python includes several built-in data structures, such as lists, tuples, sets, and dictionaries. Understanding these data structures is crucial for effective programming in Python. There are also many third-party libraries that provide additional data structures, such as NumPy arrays, pandas DataFrames, and PyTorch tensors.

* See [Python Documentation](https://docs.python.org/3/tutorial/datastructures.html) for detailed information.
* For detailed implementation, refer to [data_structures.py](https://github.com/ikathuria/python-to-ai/blob/main/0%20Basic_Python_Concepts/data_structures.py).

### Lists
- Ordered, mutable collections of items.
- Defined using square brackets: `my_list = [1, 2, 3]`

### Tuples
- Ordered, immutable collections of items.
- Defined using parentheses: `my_tuple = (1, 2, 3)`

### Sets
- Unordered collections of unique items.
- Defined using curly braces: `my_set = {1, 2, 3}`

### Dictionaries
- Unordered collections of key-value pairs.
- Defined using curly braces: `my_dict = {'key': 'value'}`

### Linked Lists
- Ordered collections of items, where each item points to the next.
- Defined using a class:
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### NumPy Arrays
- Homogeneous collections of items (all elements are of the same type).
- Defined using the NumPy library:
```python
import numpy as np
my_array = np.array([1, 2, 3])
```

### NumPy matrices
- 2-dimensional homogeneous collections of items.
- Defined using the NumPy library:
```python
import numpy as np
my_matrix = np.array([[1, 2], [3, 4]])
```

### Pandas DataFrames
- 2-dimensional labeled data structures with columns of potentially different types.
- Defined using the pandas library:
```python
import pandas as pd
my_df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
```

### PyTorch Tensors
- Multi-dimensional arrays with GPU acceleration.
- Defined using the PyTorch library:
```python
import torch
my_tensor = torch.tensor([[1, 2], [3, 4]])
```

## Functions

Functions are reusable blocks of code that perform a specific task. They help organize code, make it more readable, and allow for code reuse. In Python, you can define a function using the `def` keyword, followed by the function name and parentheses.

### Defining a Function
```python
def my_function(param1, param2):
    # Function body
    return result
```

### Calling a Function
```python
result = my_function(arg1, arg2)
```

### Built-in Functions
Python provides many built-in functions, such as `print()`, `len()`, and `range()`. You can also create your own custom functions.

### Lambda Functions
Lambda functions are small anonymous functions defined using the `lambda` keyword. They can take any number of arguments but can only have one expression.
```python
my_lambda = lambda x: x * 2
```

## Error Handling

Error handling is an essential part of programming that allows you to manage and respond to errors gracefully. In Python, you can handle errors using `try`, `except`, and `finally` blocks.

### Try Block
```
try:
	# Code that may raise an exception
except ExceptionType:
	# Code to handle the exception
finally:
	# Code that will run regardless of whether an exception occurred
```

## Extra resources
- [Python Official Documentation](https://docs.python.org/3/)
- [Learn Python - Full Course for Beginners](https://www.youtube.com/watch?v=rfscVS0vtbw)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
