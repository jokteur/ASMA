import numpy as np

a = np.array([0, 1, 2])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(np.einsum("ji,i->i", b, a))
print(b @ a)