data =    [3, 7, 2, 9, 5, 11, 4, 8]
answers = [9, 21, 6, 27, 15, 33, 12, 24]

w = 0.01
lr = 0.01

for epoch in range(101):
    for x, y in zip(data, answers):
        predicted = (x * w)
        loss = (predicted - y)** 2
        gradient = 2 * w * (predicted - y)
        w = w - lr * gradient
    
    if epoch % 10 == 0:
        print(f"epoch {epoch}: loss={loss:.4f}")
    
print(f"for number 100 answer: {100*w:.4f}")