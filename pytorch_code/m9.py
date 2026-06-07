import torch
import torchvision
from torch import nn
import time

"""first it install MNIST dataset"""

train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=32,
    shuffle=True
)

model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(3136, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

start = time.time()
for epoch in range(41):
    for image, labels in train_loader:
        image = image.cuda()
        labels= labels.cuda()
        predicted = model(image).squeeze()
        loss      = loss_fn(predicted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for image, labels in test_loader:
                image = image.cuda()
                labels= labels.cuda()
                predicted = model(image)
                values, predicted_labels = torch.max(predicted, dim=1)
                total += labels.size(0)
                correct += (predicted_labels == labels).sum().item()
        print(f"epoch {epoch}: loss={loss.item():.4f},accuracy: {correct/total*100:.2f}%")
end = time.time()
print(f"time: {end - start:.2f} sec")