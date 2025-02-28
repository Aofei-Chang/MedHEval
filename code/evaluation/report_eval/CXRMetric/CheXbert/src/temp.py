import torch
import matplotlib.pyplot as plt
import pandas
from tqdm import tqdm

data = torch.load("sentence_embed.pt")

device = "cuda:1" if torch.cuda.is_available() else "cpu"
norms = torch.stack([torch.norm(data[a]) for a in data]).to(device)
values = torch.stack([data[a] for a in sorted(data.keys())]).to(device)
cuda_data = [data[a].to(device) for a in data]
print(values.size(), norms.size(), len(cuda_data), device)

sizes = []
#cutoffs = [0.999, 0.99, 0.98, 0.95, 0.9, 0.85, 0.75, 0.6, 0.4, 0.2]
# cutoffs = [0.98, 0.95, 0.9, 0.85, 0.65, 0.4, 0.2]
# for cutoff in cutoffs[::-1]:
cutoff = 1.0
x = {}
copy = set()
pbar = tqdm(total=len(data))
idxs = set(range(len(data)))
values_copy = values.detach().clone().to(device)
while len(idxs) != 0:
    i = min(idxs)
    cos = values_copy @ cuda_data[i] / (norms * norms[i])
    same_t = torch.where(cos >= cutoff)[0]
    same = set(same_t.tolist())
    copy.update(same)
    x[i] = same
    idxs -= same
    idxs -= {i}
    values_copy[same_t, :] *= 0.
    pbar.update(1)
pbar.close()
print(cutoff, len(x))
sizes.append(len(x))
torch.save(x, "sentences_100.pt")
