import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from custom_dataset import Kadid10kDataset2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 40
BATCH_SIZE = 64

# ## 데이터셋 불러오기

train_data_set = Kadid10kDataset2(transforms=None, is_train=True)
train_loader = DataLoader(train_data_set, batch_size=4, shuffle=True)

test_data_set = Kadid10kDataset2(transforms=None, is_train=False)
test_loader = DataLoader(test_data_set, batch_size=4, shuffle=True)

# x_train, x_test = train_test_split(dataset, test_size=0.2, random_state=123, shuffle=True)
# train_loader = DataLoader(x_train, batch_size=4, shuffle=True, num_workers=0)
# test_loader = DataLoader(x_test, batch_size=4, shuffle=True, num_workers=0)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# ## 하이퍼파라미터
# `to()` 함수는 모델의 파라미터들을 지정한 곳으로 보내는 역할을 합니다. 일반적으로 CPU 1개만 사용할 경우 필요는 없지만, GPU를 사용하고자 하는 경우 `to("cuda")`로 지정하여 GPU로 보내야 합니다. 지정하지 않을 경우 계속 CPU에 남아 있게 되며 빠른 훈련의 이점을 누리실 수 없습니다.
# 최적화 알고리즘으로 파이토치에 내장되어 있는 `optim.SGD`를 사용하겠습니다.

model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# ## 학습하기

def train(model, train_loader, optimizer, epoch):
    model.train()
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch_idx, (data, target) in enumerate(zip(train_loader, targets)):
        print(data)
        data, target = data.to(DEVICE), target
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# ## 테스트하기

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 배치 오차를 합산
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def save_model(model):
    PATH = './weights/'

    torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
    torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, PATH + 'all.tar')


# ## 코드 돌려보기
# 자, 이제 모든 준비가 끝났습니다. 코드를 돌려서 실제로 학습이 되는지 확인해봅시다!

prev_best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
        epoch, test_loss, test_accuracy))

    if prev_best_acc < test_accuracy:
        save_model(model)
        prev_best_acc = test_accuracy

