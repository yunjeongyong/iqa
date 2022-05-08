import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
from torchvision import transforms
import torchvision.models.shufflenetv2 as ShuffleNetV2
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from custom_dataset import Kadid10kDataset
from tiddataset import TidDataset
from model import ShuffleNetV2


if __name__ == '__main__':
    hyper_param_epoch = 20
    hyper_param_batch = 4
    hyper_param_learning_rate = 0.001

    transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    dataset = Kadid10kDataset(transforms=transforms_train, is_train=True, return_type='dist')

    train_dataset = TensorDataset(dataset.x_train, dataset.y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # print(dataset.x_train)

    # train_dataset == (x_train, y_train)
    # train_dataset == ((dist, ref), y_train)

    test_dataset = TensorDataset(dataset.x_test, dataset.y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    custom_model = ShuffleNetV2().to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    custom_model.train()
    for e in range(hyper_param_epoch):
        # all이면 ((dist, ref), dmos)
        # ref면 (ref, dmos)
        for i_batch, (dist, dmos) in enumerate(train_loader):
            # print(item['label'])
            # images = item['image'].to(device)
            # print(images)
            # labels = item['label'].to(device)

            # labels = item['label']
            # labels = torch.LongTensor(labels)
            # print(labels)

            # x_train, y_train = item
            # print(torch.Size(y_train))
            # print(y_train)

            # Forward pass
            prediction = custom_model(dist)
            # print(images)
            # print(labels)
            loss = F.mse_loss(prediction, dmos)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i_batch + 1) % hyper_param_batch == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(e + 1, hyper_param_epoch, loss.item()))
                PATH = './weights/'
                torch.save(custom_model, PATH + 'model.pt')  # 전체 모델 저장
                torch.save(custom_model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
                torch.save({
                    'epoch': i_batch + 1,
                    'model': custom_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                }, PATH + 'all.tar')


    # Test the model
    custom_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for item in test_loader:
            x_test, y_test = item
            prediction = custom_model(x_test)
            # _, predicted = torch.max(prediction.data, 1)
            # total += len(y_test)
            # correct += (predicted == y_test).sum().item()
            print('dmos: {}, predicted: {}'.format(y_test, prediction))

