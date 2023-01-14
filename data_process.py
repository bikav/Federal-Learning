"""
    资料处理
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('../')
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def dataSet(file_name, batch):
    df = pd.read_csv('Dataset/' + file_name + '.csv', encoding='gbk')

    Y = df[['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance']]
    X = df.drop(['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance'], 1)
    X = X.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # X_train, y_train = SMOTE().fit_resample(X_train, y_train)  # 上采样，提升样本数较少的资料
    # X_train, y_train = TomekLinks().fit_resample(X_train, y_train)  # 下采样，删除一些边界辨识度不高的样本

    # 数据集资料取整
    train_len = int(len(X_train) / batch) * batch
    test_len = int(len(X_test) / batch) * batch
    X_train, y_train, X_test, y_test = X_train[:train_len], y_train[:train_len], X_test[:test_len], y_test[:test_len]

    return X_train, X_test, y_train, y_test
