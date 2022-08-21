from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import csv
from tqdm import tqdm
import json


images = []
targets = []
num_classes = []
d = {}
file_path = "./sample.json"



class Kadid10kDataset(Dataset):
    def __init__(self, transforms=None, is_train=True):
        # self.images = []
        # self.targets = []
        self.transforms = transforms
        self.is_train = is_train
        # tf_toTensor = ToTensor()

        self.num_classes = 40

        self.images_train = []
        self.images_test = []
        self.targets_train = []
        self.targets_test = []

        csv_path = './kadid10k/dmos.csv'
        folder = './kadid10k/images/'
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [row for row in reader]
            idx = 0
            # total = len(rows)
            total = 20
            if len(images) == 0:
                label_idx = 0
                for dist, _, dmos, _ in tqdm(rows, total=total):
                    img = Image.open(folder + dist)
                    img = img.convert("RGB")
                    # img = tf_toTensor(img)
                    if self.transforms is not None:
                        img = self.transforms(img)
                    images.append(img)

                    dmos = int(float(dmos)*10)
                    if dmos not in d:
                        d[dmos] = label_idx
                        label_idx += 1

                    targets.append(d[dmos])


                    # targets.append(idx)
                    if idx == total:
                        break
                    else:
                        idx += 1

        with open(file_path, 'w') as outfile:
            json.dump(d, outfile, indent=4)
        self.num_classes = 40

        entire_len = len(images)
        train_len = int(float(entire_len) / 100.0 * 80.0)
        for i, (img, t) in enumerate(zip(images, targets)):
            if i < train_len:
                self.images_train.append(img)
                self.targets_train.append(t)
            else:
                self.images_test.append(img)
                self.targets_test.append(t)

        # self.targets_train = [self.targets_test]
        # self.targets_test = [self.targets_test]

        print(self.targets_train)
        print(self.targets_test)

    def __getitem__(self, item: int): #용윤정 3.5
        # return self.images[item]
        if self.is_train:
            # return self.images_train[item]
            return {'image': self.images_train[item], 'label': self.targets_train[item]}
        else:
            # return self.images_test[item]
            return {'image': self.images_test[item], 'label': self.targets_test[item]}

    def __len__(self): # 전체 학생 수
        # return len(self.images)
        return len(self.images_train) if self.is_train else len(self.images_test)

    # def get_target(self):
    #     return self.targets

