"""
    FedAvg 客户端
"""

from data_process import dataSet


def train(args, nn, file_name, num):
    print('Client', num + 1, 'training:')
    X_train, X_test, y_train, y_test = dataSet(file_name, args.B)
    nn.len = len(X_train)  # 设置模型大小
    batch_size = args.B  # 本地批量大小-200
    epochs = args.E  # 本地模型训练次数-10

    nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return nn


def test(args, nn):
    X_train, X_test, y_train, y_test = dataSet(nn.file_name, args.B)

    loss, acc = nn.evaluate(
        X_test,
        y_test,
        batch_size=args.B,
        verbose=0
    )
    print("\nTest accuracy: %.3f%%" % (100.0 * acc))

    return acc
