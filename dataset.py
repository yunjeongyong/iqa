import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import csv
import os
import json

from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, csv_path='./kadid10k/dmos.csv'):
        self.data = []
        ref_dict = {}

        # csv_path = './kadid10k/dmos.csv'
        folder = './kadid10k/images/'
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            for dist, ref, dmos, var in reader:
                ref = os.path.join('./kadid10k/ref/', ref)
                if ref not in ref_dict:
                    ref_dict[ref] = Image.open(ref)
                self.data.append((os.path.join(folder, dist),
                                  ref_dict[ref],
                                  dmos,
                                  var))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        dist, ref, dmos, var = self.data[idx]
        dist_img = Image.open(dist)
        return dist_img, ref, dmos, var


if __name__ == '__main__':
    dataset = MyDataset('./kadid10k/dmos.csv')
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, drop_last=False)
    # for epoch in range(2):
    #     print(f"epoch : {epoch} ")
    # for batch in dataloader: print(batch)

    for data in dataloader:
        print(data)
        exit(0)



