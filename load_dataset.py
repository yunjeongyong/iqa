from train import ShuffleNetV2
from torchvision import transforms
from scipy.stats import spearmanr, pearsonr, kendalltau
from dmos_normalize import minmax_normalize
from torch.utils.data import DataLoader, TensorDataset
import torch
from custom_dataset import Kadid10kDataset
from livedataset import LiveDataset
from tiddataset import TidDataset
from tqdm import tqdm


def norm(preds, dmos_array):
    dmos_min = min(dmos_array)
    dmos_max = max(dmos_array)
    dmos_norms = [minmax_normalize(dmos, dmos_min, dmos_max) for dmos in dmos_array]

    pred_min = min(preds)
    pred_max = max(preds)
    pred_norms = [minmax_normalize(pred, pred_min, pred_max) for pred in preds]

    return dmos_norms, pred_norms

def srcc(preds, dmos_array):
    dmos_norms, pred_norms = norm(preds, dmos_array)
    return spearmanr(dmos_norms, pred_norms)

def plcc(preds, dmos_array):
    dmos_norms, pred_norms = norm(preds, dmos_array)
    return pearsonr(dmos_norms, pred_norms)

def krcc(preds, dmos_array):
    dmos_norms, pred_norms = norm(preds, dmos_array)
    return kendalltau(dmos_norms, pred_norms)


if __name__ == '__main__':

    MODEL_PATH = './weights/model_state_dict.pt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ShuffleNetV2()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    # kadid_dataset = Kadid10kDataset(transforms=transforms_test)
    # X = kadid_dataset.dist
    # y = kadid_dataset.dmos

    live_dataset = Kadid10kDataset(transforms=transforms_test)
    X = live_dataset.dist
    y = live_dataset.dmos

    x_test = [t.numpy() for t in X]
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y)
    y_test = torch.unsqueeze(y_test, 1)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    preds = []
    dmos_array = []

    for idx, item in enumerate(tqdm(test_loader)):
        x_test, y_test = item
        with torch.no_grad():
            prediction = model(x_test)
            # print('Predicted value: %f' % prediction)

            preds.append(prediction)
            dmos_array.append(y_test)

    a = srcc(preds, dmos_array)
    b = plcc(preds, dmos_array)
    c = krcc(preds, dmos_array)
    print(a)
    print(b)
    print(c)



