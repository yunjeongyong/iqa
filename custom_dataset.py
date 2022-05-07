from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import csv
from tqdm import tqdm
from dmos_normalize import minmax_normalize
import json
from sklearn.model_selection import train_test_split
import numpy as np


# images = []
# dmos = []
# num_classes = []
# d = {}
# file_path = "./sample.json"


class Kadid10kDataset(Dataset):
    def __init__(self, transforms=None, is_train=True, return_type='all'):
        self.transforms = transforms
        self.is_train = is_train
        self.return_type = return_type  # all, dist, ref

        self.X = []

        self.dist = []
        self.ref = []
        self.dmos = []
        self.dmos_norm = []

        csv_path = './kadid10k/dmos.csv'
        folder = './kadid10k/images/'
        ref_folder = './kadid10k/ref/'

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            self.total = 100
            # self.total = len(rows)
            for dist, ref, dmos, _ in tqdm(rows, total=self.total):
                dist_img = Image.open(folder + dist)
                dist_img = dist_img.convert("RGB")
                # dist_img = np.array(dist_img)

                ref_img = Image.open(ref_folder + ref)
                ref_img = ref_img.convert("RGB")
                # ref_img = np.array(ref_img)

                if self.transforms is not None:
                    dist_img = self.transforms(dist_img)
                    ref_img = self.transforms(ref_img)

                self.X.append((dist_img, ref_img))

                # self.dist.append(dist_img)
                # self.ref.append(ref_img)
                self.dmos.append(float(dmos))
                if idx == self.total:
                    break
                else:
                    idx += 1

            self.X = torch.FloatTensor(self.X)
            self.dmos = torch.FloatTensor(self.dmos)

            x_train, x_test, y_train, y_test = self.train_test_split(self.X, self.dmos, 0.2, 2, False)
            if self.is_train:
                self.X = x_train
                self.dmos = y_train
            else:
                self.X = x_test
                self.dmos = y_test

            for dist_img, ref_img in self.X:
                self.dist.append(dist_img)
                self.ref.append(ref_img)

            if self.return_type == 'dist':
                self.X = [dist for dist, _ in self.X]
            elif self.return_type == 'ref':
                self.X = [ref for _, ref in self.X]

            self.normalize()

    def train_test_split(self, X, y, test_size, random_state, shuffle):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return x_train, x_test, y_train, y_test

    def normalize(self):
        max_val = max(self.dmos)
        min_val = min(self.dmos)
        self.dmos_norm = [minmax_normalize(d, min_val, max_val) for d in self.dmos]

    def __getitem__(self, item):
        return self.ref[item], self.dist[item], self.dmos[item], self.dmos_norm[item]

    def __len__(self):
        return self.total

