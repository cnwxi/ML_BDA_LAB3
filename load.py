import pandas as pd
import torch


def load_data(arg='Bayes'):
    df_train = pd.read_csv('./adult_train.csv')
    df_test = pd.read_csv('./adult_test.csv')
    df_test.drop([0], inplace=True)

    print(df_train.head())
    print(df_test.head())

    # , 'Relationship', 'Race'
    drop_col = ['fnlwgt', 'Education', 'Country']
    for i in [df_train, df_test]:
        for j in drop_col:
            i.drop(columns=j, axis=1, inplace=True)

    print(df_train.info())
    print(df_train.isnull().sum())

    dd = pd.to_numeric(df_test['Age'])
    df_test = pd.concat([dd, df_test.drop(columns='Age', axis=1)], axis=1)

    # for i in df_train_X.columns:
    #     print(df_train[i].isna().value_counts())
    #
    # for i in df_train_X.columns:
    #     print(f'Name:{i}\n', df_train_X[i].unique())
    object_col = []
    int_col = []
    for i in df_train.columns:
        if df_train[i].dtype == 'object':
            object_col.append(i)
        elif df_train[i].dtype == 'int64':
            int_col.append(i)
    print(f"object:{object_col}\n int:{int_col}")

    df_train['AgeGroup'] = pd.cut(df_train.Age, range(0, 101, 10), right=False,
                                  labels=[i * 10 for i in range(10)]).astype(int)
    df_train.drop(columns='Age', axis=1, inplace=True)
    df_test['AgeGroup'] = pd.cut(df_test.Age, range(0, 101, 10), right=False,
                                 labels=[i * 10 for i in range(10)]).astype(int)
    df_test.drop(columns='Age', axis=1, inplace=True)
    df_train['EduGroup'] = pd.cut(df_train.Education_Num, range(0, 21, 5), right=False,
                                  labels=[i * 5 for i in range(4)]).astype(int)
    df_train.drop(columns='Education_Num', axis=1, inplace=True)
    df_test['EduGroup'] = pd.cut(df_test.Education_Num, range(0, 21, 5), right=False,
                                 labels=[i * 5 for i in range(4)]).astype(int)
    df_test.drop(columns='Education_Num', axis=1, inplace=True)

    # print(df_train['Capital_Gain'].unique().max())
    # print(df_test['Capital_Gain'].unique().max())
    # print(df_train['Capital_Loss'].unique().max())
    # print(df_test['Capital_Loss'].unique().max())

    df_train['Capital_Gain_Group'] = pd.cut(df_train.Capital_Gain, range(0, 100001, 5000), right=False,
                                            labels=[i * 5000 for i in range(20)]).astype(int)
    df_train.drop(columns='Capital_Gain', axis=1, inplace=True)

    df_train['Capital_Loss_Group'] = pd.cut(df_train.Capital_Loss, range(0, 5001, 1000), right=False,
                                            labels=[i * 1000 for i in range(5)]).astype(int)
    df_train.drop(columns='Capital_Loss', axis=1, inplace=True)

    df_test['Capital_Gain_Group'] = pd.cut(df_test.Capital_Gain, range(0, 100001, 5000), right=False,
                                           labels=[i * 5000 for i in range(20)]).astype(int)
    df_test.drop(columns='Capital_Gain', axis=1, inplace=True)

    df_test['Capital_Loss_Group'] = pd.cut(df_test.Capital_Loss, range(0, 5001, 1000), right=False,
                                           labels=[i * 1000 for i in range(5)]).astype(int)
    df_test.drop(columns='Capital_Loss', axis=1, inplace=True)

    for i in object_col:
        df_train[i] = df_train[i].astype(str).apply(lambda val: val.replace(" ", ""))
        df_test[i] = df_test[i].astype(str).apply(lambda val: val.replace(" ", ""))
        df_train[i] = df_train[i].astype(str).apply(lambda val: val.replace(".", ""))
        df_test[i] = df_test[i].astype(str).apply(lambda val: val.replace(".", ""))

    if arg != 'Bayes':
        for col_name in object_col:
            dd = pd.get_dummies(df_train[col_name], drop_first=True, prefix=col_name)
            df_train = pd.concat([df_train.drop(columns=col_name, axis=1), dd], axis=1)
        for col_name in object_col:
            dd = pd.get_dummies(df_test[col_name], drop_first=True, prefix=col_name)
            df_test = pd.concat([df_test.drop(columns=col_name, axis=1), dd], axis=1)

    if arg == 'Bayes':
        df_train_y = df_train['Target']
        df_train_X = df_train.drop('Target', axis=1)
        df_test_y = df_test['Target']
        df_test_X = df_test.drop('Target', axis=1)
        return df_train_X, df_train_y, df_test_X, df_test_y

    df_train_y = df_train['Target_>50K']
    df_train_X = df_train.drop('Target_>50K', axis=1)
    df_test_y = df_test['Target_>50K']
    df_test_X = df_test.drop('Target_>50K', axis=1)
    print(df_train_X.info())
    print(df_test_X.head())
    train_features = torch.tensor(df_train_X.values).to(torch.float32)
    train_labels = torch.tensor(df_train_y.values.reshape(-1)).to(torch.float32)

    test_features = torch.tensor(df_test_X.values).to(torch.float32)
    test_labels = torch.tensor(df_test_y.values.reshape(-1)).to(torch.float32)
    return train_features, train_labels, test_features, test_labels


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


if __name__ == '__main__':
    load_data()
