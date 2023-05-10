import numpy as np
from load import load_data
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class DecisionTreeClassifierWithWeight:
    def __init__(self):
        self.best_err = 1  # 最小的加权错误率
        self.best_fea_id = 0  # 最优特征id
        self.best_thres = 0  # 选定特征的最优阈值
        self.best_op = 1  # 阈值符号，其中 1: >, 0: <

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X)) / len(X)
        n = X.shape[1]
        for i in range(n):
            feature = X[:, i]  # 选定特征列
            fea_unique = np.sort(np.unique(feature))  # 将所有特征值从小到大排序
            for j in range(len(fea_unique) - 1):
                thres = (fea_unique[j] + fea_unique[j + 1]) / 2  # 逐一设定可能阈值
                for op in (0, 1):
                    y_ = 2 * (feature >= thres) - 1 if op == 1 else 2 * (feature < thres) - 1  # 判断何种符号为最优
                    err = np.sum((y_ != y) * sample_weight)
                    if err < self.best_err:  # 当前参数组合可以获得更低错误率，更新最优参数
                        self.best_err = err
                        self.best_op = op
                        self.best_fea_id = i
                        self.best_thres = thres
        return self

    def predict(self, X):
        feature = X[:, self.best_fea_id]
        return 2 * (feature >= self.best_thres) - 1 if self.best_op == 1 else 2 * (feature < self.best_thres) - 1

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        if sample_weight is not None:
            print(np.sum((y_pred == y) * sample_weight))
        print('confusion_matrix', confusion_matrix(y, y_pred))
        print('precision_score', precision_score(y, y_pred))
        print('recall_score', recall_score(y, y_pred))
        print('f1_score', f1_score(y, y_pred))
        print('accuracy_score', accuracy_score(y, y_pred))


class AdaBoostClassifier:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        sample_weight = np.ones(len(X)) / len(X)  # 初始化样本权重为 1/N
        for _ in range(self.n_estimators):
            dtc = DecisionTreeClassifierWithWeight().fit(X, y, sample_weight)  # 训练弱学习器
            alpha = 1 / 2 * np.log((1 - dtc.best_err) / dtc.best_err)  # 权重系数
            y_pred = dtc.predict(X)
            sample_weight *= np.exp(-alpha * y_pred * y)
            sample_weight /= np.sum(sample_weight)
            self.estimators.append(dtc)
            self.alphas.append(alpha)
        return self

    def predict(self, X):
        y_pred = np.empty((len(X), self.n_estimators))
        for i in range(self.n_estimators):
            y_pred[:, i] = self.estimators[i].predict(X)
        y_pred = y_pred * np.array(self.alphas)
        return 2 * (np.sum(y_pred, axis=1) > 0) - 1

    def score(self, X, y):
        y_pred = self.predict(X)
        print('confusion_matrix', confusion_matrix(y, y_pred))
        print('precision_score', precision_score(y, y_pred))
        print('recall_score', recall_score(y, y_pred))
        print('f1_score', f1_score(y, y_pred))
        print('accuracy_score', accuracy_score(y, y_pred))


train_features, train_labels, test_features, test_labels = load_data()

train_features = np.array(train_features)
train_labels = np.array(train_labels)
train_labels = 2 * train_labels - 1
test_features = np.array(test_features)
test_labels = np.array(test_labels)
test_labels = 2 * test_labels - 1
DecisionTreeClassifierWithWeight().fit(train_features, train_labels).score(test_features, test_labels)

AdaBoostClassifier().fit(train_features, train_labels).score(test_features, test_labels)
