"""
    FedTrimmed-mean 构建模型
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


def DNN(args, file_name):
    model = Sequential()
    model.add(Dense(128, input_dim=args.input_dim))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(6))
    model.add(Activation('softmax'))  # 多分类

    # model.summary()  # 模型各层的参数状况
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
