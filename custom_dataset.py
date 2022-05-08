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
import datacenter
from torchvision.transforms.functional import to_pil_image


class Kadid10kDataset(Dataset):
    def __init__(self, transforms=None, is_train=True, return_type='dist', test_data='live'):
        # train/test_data == 'kadid', 'live', 'tid'
        self.transforms = transforms
        self.is_train = is_train
        self.return_type = return_type  # all, dist, ref
        self.dmos_norm_train = []
        self.dmos_norm_test = []

        kadid10kData = datacenter.Kadid10kData(limit=20)
        if self.transforms is not None:
            for i in range(len(kadid10kData.X)):
                dist, ref = kadid10kData.X[i]
                dist = self.transforms(dist)
                ref = self.transforms(ref)
                kadid10kData.X[i] = (dist, ref)

        # x_train == [(dist, ref), (dist, ref), ... ] 80개
        # x_test == [(dist, ref), (dist, ref), ... ] 20개
        self.x_train, self.x_test, self.y_train, self.y_test =\
            self.train_test_split(kadid10kData.X, kadid10kData.dmos, test_size=0.2, random_state=2, shuffle=True)

        if self.return_type == 'dist':
            self.x_train = [np.array(dist) for dist, _ in self.x_train]
            self.x_test = [np.array(dist) for dist, _ in self.x_test]
        elif self.return_type == 'ref':
            self.x_train = [np.array(ref) for _, ref in self.x_train]
            self.x_test = [np.array(ref) for _, ref in self.x_test]
        elif self.return_type == 'all':
            self.x_train = [(np.array(dist), np.array(ref)) for dist, ref in self.x_train]
            self.x_test = [(np.array(dist), np.array(ref)) for dist, ref in self.x_test]

        # self.x_train = [(dist.numpy(), ref.numpy()) for dist, ref in self.x_train]
        # self.x_train = [(np.array(dist), np.array(ref)) for dist, ref in self.x_train]
        self.x_train = torch.FloatTensor(self.x_train)
        self.y_train = torch.FloatTensor(self.y_train)
        self.y_train = torch.unsqueeze(self.y_train, 1)

        # self.x_test = [(np.array(dist), np.array(ref)) for dist, ref in self.x_test]
        self.x_test = torch.FloatTensor(self.x_test)
        self.y_test = torch.FloatTensor(self.y_test)
        self.y_test = torch.unsqueeze(self.y_test, 1)

        # if self.return_type == 'all':
        #     self.X = [(dist, ref) for dist, ref in zip(self.dist, self.ref)]

        if test_data != 'kadid':
            data_test = self.get_data(test_data, limit=int(kadid10kData.limit * 0.2))
            self.x_test = data_test.X
            if self.transforms is not None:
                for i in range(len(data_test.X)):
                    dist, ref = data_test.X[i]
                    dist = self.transforms(dist)
                    ref = self.transforms(ref)
                    data_test.X[i] = (dist, ref)
            # self.x_test = [(dist.numpy(), ref.numpy()) for dist, ref in self.x_test]
            self.x_test = [(np.array(dist), np.array(ref)) for dist, ref in self.x_test]
            self.x_test = torch.FloatTensor(self.x_test)
            self.y_test = data_test.dmos
            self.y_test = torch.FloatTensor(self.y_test)

        self.normalize(self.y_train, True)
        self.normalize(self.y_test, False)

    def train_test_split(self, X, y, test_size, random_state, shuffle):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return x_train, x_test, y_train, y_test

    def get_data(self, type='kadid', limit=None):
        if type == 'kadid':
            return datacenter.Kadid10kData(limit)
        elif type == 'live':
            return datacenter.LiveData(limit)
        elif type == 'tid':
            return datacenter.TidData(limit)
        else:
            return None

    def normalize(self, dmos, is_train):
        max_val = max(dmos)
        min_val = min(dmos)
        if is_train:
            self.dmos_norm_train = [minmax_normalize(d, min_val, max_val) for d in dmos]
        else:
            self.dmos_norm_test = [minmax_normalize(d, min_val, max_val) for d in dmos]

    def __getitem__(self, item):
        if self.is_train:
            dist, ref = self.x_train[item]
            return ref, dist, self.y_train[item], self.dmos_norm_train[item]
        else:
            dist, ref = self.x_test[item]
            return ref, dist, self.y_test[item], self.dmos_norm_test[item]

    def __len__(self):
        return len(self.x_train) if self.is_train else len(self.x_test)

