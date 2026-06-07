import numpy as np

data = np.array([
    [80,  2, 10],
    [120, 3, 5],
    [200, 5, 2],
    [60,  1, 15],
    [150, 4, 3],
])

answers = np.array([200, 350, 600, 150, 450])

data_mean = np.mean(data , axis=0)
data_std  = np.std(data, axis=0)
data_norm = (data - data_mean) / data_std

ans_mean = np.mean(answers)
ans_std  = np.std(answers)
ans_norm = (answers - ans_mean) / ans_std

w = np.zeros(3)
learning_rate = 0.1

for epoch in range(1000):
    predicted = data_norm.dot(w)
    loss = np.mean((predicted - ans_norm) ** 2)
    gradiant = data_norm.T.dot(predicted - ans_norm) / len(data)
    w = w - learning_rate * gradiant

    if epoch % 100 == 0:
        print(f"epoch {epoch}: loss={loss:.4f}")

print(f"w = {w}")

new_house = np.array([100, 3, 7])
new_norm = (new_house - data_mean) / data_std
pred_norm = new_norm.dot(w)
pred_price = pred_norm * ans_std + ans_mean
print(f"قیمت پیش‌بینی شده: {pred_price:.0f} میلیون")