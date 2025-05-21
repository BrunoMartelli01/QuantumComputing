import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.utils import shuffle
import pandas as pd


def get_dataset(train_size=100, test_size=50, seed=62017, get_deviation=False):
    data = load_breast_cancer()
    xs, ys = data.data, data.target
    xs, ys = shuffle(xs, ys, random_state=seed)

    tr_cls = train_size // 2
    te_cls = test_size // 2

    train_features = np.concatenate([xs[ys == 1][:tr_cls], xs[ys == 0][:tr_cls]], axis=0)
    test_features = np.concatenate([xs[ys == 1][tr_cls:tr_cls + te_cls], xs[ys == 0][tr_cls:tr_cls + te_cls]], axis=0)

    train_labels = np.concatenate([np.ones(tr_cls), np.zeros(tr_cls)], axis=0)
    test_labels = np.concatenate([np.ones(te_cls), np.zeros(te_cls)], axis=0)

    if get_deviation:
        df_original = pd.DataFrame(xs, columns=data.feature_names).describe().drop(index='count')
        df_train = pd.DataFrame(train_features, columns=data.feature_names).describe().drop(index='count')
        df_test = pd.DataFrame(test_features, columns=data.feature_names).describe().drop(index='count')

        return np.round(np.abs(df_original.values - df_train.values).mean() +
                        np.abs(df_original.values - df_test.values).mean(), 3)

    train_features, train_labels = shuffle(train_features, train_labels, random_state=seed)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=seed)

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features, train_labels, test_labels
