import torch
import torch.nn as nn
from torch import optim
from load import data_iter
from load import load_data

train_features, train_labels, test_features, test_labels = load_data()


class Act_Net(nn.Module):
    def __init__(self, epoch, batch_size, lr):
        super(Act_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(43, 256), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 1))
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sig(x)
        return x


class Bat_Net(nn.Module):
    def __init__(self, epoch, batch_size, lr):
        super(Bat_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(43, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 1))
        self.sig = nn.Sigmoid()

        self.loss = nn.BCELoss()
        self.lr = lr
        self.epoch = epoch

        self.batch_size = batch_size
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sig(x)
        return x


def train(train_features, train_labels, test_features, test_labels):
    model = Bat_Net(5000, 1024, 0.005)

    if torch.cuda.is_available():
        model = model.cuda()
        train_features = train_features.cuda()
        train_labels = train_labels.to(torch.float32).cuda()
        test_features = test_features.cuda()
        test_labels = test_labels.cuda()
    max_acc = 0
    now_epoch = 0
    for epoch in range(model.epoch):
        for X, y in data_iter(model.batch_size, train_features, train_labels):
            loss = model.loss(model(X).squeeze(), y)
            loss.sum().backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
        if epoch % 50 == 0:
            print(f'{epoch}/{model.epoch} loss:{loss.sum()}')

    y_pred = model(test_features)
    mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
    correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
    acc = correct.item() / test_labels.size(0)  # 计算分类准确率
    if acc > max_acc:
        max_acc = acc
        now_epoch = epoch
        torch.save(model, './best_dl.pkl')
    print(f'max_acc:{max_acc} now_epoch:{now_epoch}')


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = load_data()
    train(train_features, train_labels, test_features, test_labels)
