import csv

csv_path = './kadid10k/dmos.csv'

change = []
with open(csv_path, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    rows = [row for row in reader]
    idx = 0
    total = len(rows)
    for _, _, dmos, _ in rows:
        change.append(int(float(dmos) * 10))

    change_set = set(change)
    change = list(change_set)
    num_classes = len(change)

print(change)
print(num_classes)