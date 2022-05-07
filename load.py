from train import resnet50
from PIL import Image
from torchvision import transforms
from scipy.stats import spearmanr
from dmos_normalize import minmax_normalize
import torch
import os
import custom_dataset
import json
import csv
import scipy.stats
from custom_dataset import Kadid10kDataset


def spearman(preds):
    csv_path = './kadid10k/dmos.csv'
    with open(csv_path, 'r',  newline='') as f:
        reader = csv.reader(f)
        next(reader)

        dmos_array = [float(row[2]) for row in reader]
        dmos_min = min(dmos_array)
        dmos_max = max(dmos_array)
        dmos_norms = [minmax_normalize(dmos, dmos_min, dmos_max) for dmos in dmos_array]

        pred_min = min(preds)
        pred_max = max(preds)
        pred_norms = [minmax_normalize(pred, pred_min, pred_max) for pred in preds]

        return scipy.stats.pearsonr(dmos_norms, pred_norms)



if __name__ == '__main__':

    MODEL_PATH = './weights/model_state_dict.pt'
    IMAGE_PATH = '/Users/yunjeongyong/Desktop/hansung/2022intern/project_intern/kadid10k/images/'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = resnet50()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    file_list = os.listdir(IMAGE_PATH)
    preds = []

    for item in file_list:
        path = IMAGE_PATH + item
        image = Image.open(path)
        image = transforms_test(image)
        image = image.to(device)
        image = image.unsqueeze(0)
        print(image.shape)

        with torch.no_grad():
            prediction = model(image)
            preds.append(prediction)
            print('Predicted value: %f' % prediction)

    a = spearman(preds)
    print(a)



