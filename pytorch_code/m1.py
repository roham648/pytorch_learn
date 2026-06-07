data = [5, 10, 15, 20, 25, 30]

mean = sum(data) / len(data)

variance = sum((x - mean) ** 2 for x in data) / len(data)

import math
std = math.sqrt(variance)

normalized = [(x - mean) / std for x in data]


print("mean:", mean)
print("variance:", variance)
print("std:", std)
print("normalized:", normalized)