import torchvision
from torch import nn
import torch
import time

"""first it install fashionMNIST dataset"""

train_data = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.FashionMNIST(
    root = "./data",
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
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32,128,kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2304, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
loss_fn = nn.CrossEntropyLoss()
best_accuracy = 0
patience = 8
counter = 0
start = time.time()

for epoch in range(1000):
    for image, labels in train_loader:
        image = image.cuda()
        labels = labels.cuda()
        prediccted = model(image).squeeze()
        loss = loss_fn(prediccted, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for image_test, labels_test in test_loader:
            image_test = image_test.cuda()
            labels_test= labels_test.cuda()
            predicted = model(image_test).squeeze()
            values, predicted_labels = torch.max(predicted, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels_test).sum().item()  
    current_accuracy = correct / total * 100
    print(f"epoch {epoch}: loss={loss.item():.4f}, accuracy: {current_accuracy:.4f}") 
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
    if counter >= patience:
        print(f"Early Stopping! بهترین دقت: {best_accuracy:.2f}%")
        break
end = time.time()
print(f"time: {end - start:.4f} sec")