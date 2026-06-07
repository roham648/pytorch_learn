import torch
import numpy as np

# ---- داده ----
data = np.array([
    [80,  2, 10],
    [120, 3, 5],
    [200, 5, 2],
    [60,  1, 15],
    [150, 4, 3],
    [90,  2, 8],
    [175, 4, 4],
    [65,  1, 12],
    [220, 5, 1],
    [110, 3, 7],
    [95,  2, 6],
    [135, 3, 4],
    [180, 4, 2],
    [70,  2, 11],
    [250, 6, 1],
    [85,  2, 9],
    [160, 4, 5],
    [55,  1, 20],
    [190, 5, 3],
    [125, 3, 6],
], dtype=np.float32)

answers = np.array([
    200, 350, 600, 150, 450,
    220, 520, 160, 700, 320,
    240, 400, 560, 180, 800,
    210, 480, 130, 580, 370,
], dtype=np.float32)

data_mean = np.mean(data)
data_std  = np.std(data)
data_norm = (data - data_mean) / data_std

ans_mean = np.mean(answers)
ans_std  = np.std(answers)
ans_norm = (answers - ans_mean) / ans_std

x = torch.tensor(data_norm).cuda()
y = torch.tensor(ans_norm).cuda()

model = torch.nn.Sequential(
    torch.nn.Linear(3, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
).cuda()

optimazer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn   = torch.nn.MSELoss()

for epoch in range(100001):
    predicted = model(x).squeeze()
    loss      = loss_fn(predicted, y)
    optimazer.zero_grad()
    loss.backward()
    optimazer.step()

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")

# ---- پیش‌بینی ----
new_house = np.array([[100, 3, 7]], dtype=np.float32)
new_norm  = (new_house - data_mean) / data_std
new_tensor = torch.tensor(new_norm).cuda()

with torch.no_grad():
    pred_norm  = model(new_tensor).item()
    pred_price = pred_norm * ans_std + ans_mean
    print(f"قیمت پیش‌بینی شده: {pred_price:.0f} میلیون")