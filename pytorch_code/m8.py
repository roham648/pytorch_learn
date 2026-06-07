import torch
import numpy as np

# ---- داده ----
data = np.array([
    [80,  2, 10], [120, 3, 5],  [200, 5, 2],
    [60,  1, 15], [150, 4, 3],  [90,  2, 8],
    [175, 4, 4],  [65,  1, 12], [220, 5, 1],
    [110, 3, 7],  [95,  2, 6],  [135, 3, 4],
    [180, 4, 2],  [70,  2, 11], [250, 6, 1],
    [85,  2, 9],  [160, 4, 5],  [55,  1, 20],
    [190, 5, 3],  [125, 3, 6],
], dtype=np.float32)

# 1 = گرون (بالای 350)  0 = ارزون
labels = np.array([
    0, 0, 1, 0, 1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 1,
    0, 1, 0, 1, 1,
], dtype=np.float32)

# ---- نرمال‌سازی ----
data_mean = np.mean(data, axis=0)
data_std  = np.std(data, axis=0)
data_norm = (data - data_mean) / data_std

# ---- تبدیل به Tensor ----
X = torch.tensor(data_norm).cuda()
y = torch.tensor(labels).cuda()

# ---- مدل ----
model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
    torch.nn.Sigmoid()
).cuda()

# ---- optimizer و loss ----
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn   = torch.nn.BCELoss()

# ---- training ----
for epoch in range(35001):
    predicted = model(X).squeeze()
    loss      = loss_fn(predicted, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

# ---- پیش‌بینی ----
new_house = np.array([[210 , 5, 1]], dtype=np.float32)
new_norm   = (new_house - data_mean) / data_std
new_tensor = torch.tensor(new_norm).cuda()

with torch.no_grad():
    prob = model(new_tensor).item()
    label = "گرون 💰" if prob > 0.5 else "ارزون 🏠"
    print(f"احتمال گرون بودن: {prob:.2f}")
    print(f"نتیجه: {label}")