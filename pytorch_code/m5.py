import numpy as np

# یه تصویر ساده 4×4
image = np.array([
    [255, 128, 64,  32],
    [16,  8,   4,   2],
    [1,   0,   128, 255],
    [64,  32,  16,  8]
])

shape = np.shape(image)
print(f"shape: {shape}")

flat = image.flatten()
print(f"flat: {flat}")

split = int(len(image) * 0.75)
train = image[ :split]
test  = image[split: ]

print(f"train: {train}")
print(f"test: {test}")