import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from load import load_data
import numpy as np
from numpy import log
from numpy import exp
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class LR(nn.Module):
    def __init__(self, epoch=100, lr=0.02):
        super(LR, self).__init__()
        self.features = nn.Linear(34, 1)
        self.sigmoid = nn.Sigmoid()

        self.lr = lr
        self.epoch = epoch

        self.loss = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

    def fit(self, x, y, weight=None):
        for epoch in range(self.epoch):
            loss = self.loss(self.forward(x).squeeze(), y)
            if weight is not None:
                loss = loss * torch.tensor(weight)
            loss.sum().backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return self

    def predict(self, x):
        y_pred = self.forward(x)
        y_pred = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        print('confusion_matrix', confusion_matrix(y, y_pred))
        print('precision_score', precision_score(y, y_pred))
        print('recall_score', recall_score(y, y_pred))
        print('f1_score', f1_score(y, y_pred))
        print('accuracy_score', accuracy_score(y, y_pred))


class adaboost_lr:
    def __init__(self, n=50):
        self.n = n
        self.final_coe_matrix = None
        self.final_int_matrix = None
        self.sigmoid = nn.Sigmoid()

    def fit(self, x, y):
        int_matrix = np.ones((1, self.n))
        coe_matrix = np.ones((34, self.n))
        alp_vector = np.ones((self.n, 1))
        for i in tqdm(range(self.n)):
            base_model = LR()
            base_model.fit(x, y, weight=weight.iloc[:, 1].values)
            ypred = base_model.predict(x)
            coe = base_model.features.weight

            inte = base_model.features.bias
            int_matrix[0, i] = inte
            coe_matrix[:, i] = np.mat(coe.detach().numpy())
            ypred = np.array(ypred, dtype=np.float32)
            weight['ypred'] = ypred
            em = weight[weight.iloc[:, 2] != weight.iloc[:, 0]].iloc[:, 1].sum()
            alp = 0.5 * log((1 - em) / em)
            alp_vector[i, 0] = alp
            W = weight.iloc[:, 1] * exp(-alp * (2 * weight.iloc[:, 0] - 1) * (2 * ypred - 1))
            W = W / W.sum()
            weight.iloc[:, 1] = W
        final_coe_matrix = coe_matrix @ alp_vector / sum(alp_vector)
        final_int_matrix = int_matrix @ alp_vector / sum(alp_vector)
        self.final_coe_matrix = final_coe_matrix
        self.final_int_matrix = final_int_matrix
        return self

    def predict(self, x):
        ypred = x @ self.final_coe_matrix + self.final_int_matrix
        ypred = self.sigmoid(ypred)
        ypred = ypred.ge(0.5).float().squeeze()
        return ypred

    def score(self, X, y):
        y_pred = self.predict(X)
        print('confusion_matrix', confusion_matrix(y, y_pred))
        print('precision_score', precision_score(y, y_pred))
        print('recall_score', recall_score(y, y_pred))
        print('f1_score', f1_score(y, y_pred))
        print('accuracy_score', accuracy_score(y, y_pred))


train_features, train_labels, test_features, test_labels = load_data()

weight = pd.DataFrame({'real': train_labels, 'weight': 1 / len(train_labels)})
print('LR 训练中')
a = LR()
a.fit(train_features, train_labels).score(test_features, test_labels)

print('adaboost_lr 训练中')
aa = adaboost_lr()
aa.fit(train_features, train_labels).score(test_features, test_labels)
