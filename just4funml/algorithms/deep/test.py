import numpy as np

a = np.zeros((10, 10))
print(id(a))
print(a)
print(id(a[0, 0]))

a[:] = np.ones((5, 5))
print(id(a))
print(a)
print(id(a[0, 0]))