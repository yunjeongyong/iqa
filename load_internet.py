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


if __name__ == '__main__':

    file_path = "./sample.json"
    json_data = {}
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
    swapped_data = {v: k for k, v in json_data.items()}

    with open(file_path, 'w') as outfile:
        json.dump(swapped_data, outfile, indent=4)

    MODEL_PATH = './weights/model_state_dict.pt'
    IMAGE_PATH = '/Users/yunjeongyong/Desktop/hansung/2022intern/project_intern/yunjeong/'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = resnet50()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    file_list = os.listdir(IMAGE_PATH)

    for item in file_list:
        path = IMAGE_PATH + item
        image = Image.open(path)
        image = transforms_test(image)
        image = image.to(device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            # outputs.append(outputs.detach())
            _, pred = torch.max(outputs.data, 1)
            pred = int(pred)
            # swapped_pred = (swapped_data.get(pred) / 10.0)
            swapped_pred = float(swapped_data[str(pred)]) / 10.0
            print('Predicted value: %f' % swapped_pred)


# 이거는 인터넷
