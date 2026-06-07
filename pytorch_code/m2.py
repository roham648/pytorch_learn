students = [
    [90, 80, 70],
    [60, 75, 85],
    [95, 60, 80],
]

weights = [0.5, 0.3, 0.2]

for i, student in enumerate(students):
    score = sum(w * g for w,g in zip(weights, student))
    print(f"student {i+1}: {score}")