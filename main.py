import torch
from load import load_data
from dl import Bat_Net
from lr import LR
import numpy as np


def test():
    train_features, train_labels, test_features, test_labels = load_data()

    # 使用单一决策树建模
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=1)
    from sklearn.metrics import accuracy_score
    tree = tree.fit(train_features, train_labels)
    y_train_pred = tree.predict(train_features)
    y_test_pred = tree.predict(test_features)
    tree_train = accuracy_score(train_labels, y_train_pred)
    tree_test = accuracy_score(test_labels, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

    if torch.cuda.is_available():
        test_features = test_features.cuda()

    lr = torch.load('./best_lr.pkl')
    y_pred = lr(test_features)
    mask_lr = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类

    dl = torch.load('./best_dl.pkl')
    y_pred = dl(test_features)
    mask_dl = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
    mask = []
    for i in range(len(mask_dl)):
        if 0.3 * y_test_pred[i] + 0.3 * mask_dl[i] + 0.4 * mask_lr[i] >= 0.5:
            mask.append(1)
        else:
            mask.append(0)
    mask = np.array(mask)
    correct = 0
    for i in range(len(mask)):
        if mask[i] == test_labels[i]:
            correct += 1
    # correct = ((mask == test_labels).sum())  # 计算正确预测的样本个数
    acc = correct / test_labels.size(0)  # 计算分类准确率
    print(acc)


test()
