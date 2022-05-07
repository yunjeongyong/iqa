from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import torch
import csv
from tqdm import tqdm
from dmos_normalize import minmax_normalize


class TidDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms

        self.dist = []
        self.ref = []
        self.dmos = []
        self.dmos_norm = []

        csv_path = './all_data_csv/TID2013.txt.csv'
        folder = './tid2013'

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            self.total = 10
            # self.total = len(rows)
            for dist, _, ref, dmos in tqdm(rows, total=self.total):
                dist_img = Image.open(folder + dist)
                dist_img = dist_img.convert("RGB")

                ref_img = Image.open(folder + ref)
                ref_img = ref_img.convert("RGB")

                if self.transforms is not None:
                    dist_img = self.transforms(dist_img)
                    ref_img = self.transforms(ref_img)

                self.dist.append(dist_img)
                self.ref.append(ref_img)
                self.dmos.append(float(dmos))
                if idx == self.total:
                    break
                else:
                    idx += 1
            self.normalize()

    def normalize(self):
        max_val = max(self.dmos)
        min_val = min(self.dmos)
        self.dmos_norm = [minmax_normalize(d, min_val, max_val) for d in self.dmos]

    def __getitem__(self, item):
        return self.ref[item], self.dist[item], self.dmos[item], self.dmos_norm[item]

    def __len__(self):
        return self.total

