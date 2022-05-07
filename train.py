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
from model import ShuffleNetV2


if __name__ == '__main__':
    hyper_param_epoch = 20
    hyper_param_batch = 4
    hyper_param_learning_rate = 0.001

    transforms_train = transforms.Compose([transforms.Resize((224, 224)),
                                           transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])

    kadid_dataset = Kadid10kDataset(transforms=transforms_train)
    X = kadid_dataset.dist
    y = kadid_dataset.dmos

    # X = torch.FloatTensor(image)
    # y = torch.FloatTensor(dmos)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2, shuffle=True
    )

    x_train = [t.numpy() for t in x_train]
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    y_train = torch.unsqueeze(y_train, 1)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x_test = [t.numpy() for t in x_test]
    x_test = torch.FloatTensor(x_test)
    y_test = torch.FloatTensor(y_test)
    y_test = torch.unsqueeze(y_test, 1)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = Kadid10kDataset(transforms=transforms_train, is_train=True, return_type='dist')
    x = train_dataset.dist
    y = train_dataset.dmos
    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    test_dataset = Kadid10kDataset(transforms=transforms_test, is_train=False, return_type='dist')
    x = test_dataset.dist
    y = test_dataset.dmos
    test_dataset = TensorDataset(x, y)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    custom_model = ShuffleNetV2().to(device)

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(custom_model.parameters(), lr=hyper_param_learning_rate)
    custom_model.train()
    for e in range(hyper_param_epoch):
        for i_batch, (ref, dist, _, _) in enumerate(train_loader):
            # print(item['label'])
            # images = item['image'].to(device)
            # print(images)
            # labels = item['label'].to(device)

            # labels = item['label']
            # labels = torch.LongTensor(labels)
            # print(labels)
            # ref, dist, dmos, dmos_norm = item

            # x_train, y_train = item
            x_train = ref

            # Forward pass
            prediction = custom_model(x_train, dist)
            # print(images)
            # print(labels)
            loss = F.mse_loss(prediction, y_train)

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



