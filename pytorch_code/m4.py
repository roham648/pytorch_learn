import numpy as np

data = np.array([3, 7, 2, 9, 5, 11, 4, 8])
answers = np.array([9, 21, 6, 27, 15, 33, 12, 24])

mean = np.mean(data)
print(mean)

std = np.std(answers)
print(std)

multiplied = data * 3
print(multiplied)