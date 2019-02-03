import numpy as np

largest_number = pow(2, 8)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
print(binary)