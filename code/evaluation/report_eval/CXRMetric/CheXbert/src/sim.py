import torch
from tqdm import tqdm


d = torch.load('data.pt')
unique = []
same = []

prev = d[0]
for i in tqdm(range(1, len(d))):
    if torch.all(d[i] == prev).item():
        same.append(d[i].dot(prev).item())
        continue
    unique.append(i)
    prev = d[i]

torch.save([d[i] for i in unique], 'unique.pt')
print(sum(same) / len(same))

x = {}
copy = []
for i in range(1):
    for j in range(i+1, len(unique)):
        if (unique[i] @ unique[j]).item() > 370:
            if j in copy:
                continue
            if i not in x:
                x[i] = []
            x[i].append(j)
