"""
    FedAvg 服务器端
"""

import numpy as np
import sys

sys.path.append('../')
from client import train, test
from model import DNN

np.set_printoptions(threshold=np.inf)  # 打印完整list

clients_NB15 = ['NUSW-NB15-' + str(i) for i in range(1, 11)]


class FedAvg:
    def __init__(self, args):
        self.args = args
        self.clients = args.clients  # 资料集名称
        self.nn = DNN(args=args, file_name='server')
        self.nns = []
        self.exa_nn = DNN(args=args, file_name='server')  # 用作KS检验对比的权重
        self.secure_server = [i for i in range(0, self.args.K)]  # 安全客户端

        # args.K: 客户端数量-10; args.clients: 资料集名字
        for i in range(self.args.K):
            temp = DNN(args=args, file_name='server')
            temp.file_name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):  # t: 0~5，全局训练次数
            print('\nround', t + 1, ':')

            # 索引
            index = self.secure_server
            print('索引:', index)

            # 调度
            self.dispatch(index)

            # 客户端模型更新
            self.client_update(index)

            # 更新服务器权重
            m = np.max([len(self.secure_server), 1])  # m = 剩余客户端的数量
            self.aggregation(m)

        return self.nn  # 输出全局模型

    def dispatch(self, index):
        """调度: 把最新的nn的参数传给nns"""
        for i in index:
            weight = self.nn.get_weights()
            self.nns[i].set_weights(weight)

    def client_update(self, index):
        """本地更新: 获取每个客户端的模型更新"""
        for k in index:
            self.nns[k] = train(self.args, self.nns[k], self.nns[k].file_name, k)

    def aggregation(self, m):
        """更新权重"""
        weights = []
        for j in self.secure_server:
            weight = self.nns[j].get_weights()
            weight = weight / m
            weights.append(weight)

        update_weight = []
        for i in range(len(weights[0])):
            temp = weights[0][i]
            for j in range(1, len(self.secure_server)):
                temp = temp + weights[j][i]
            update_weight.append(temp)

        # 更新验证神经网络的参数
        self.nn.set_weights(update_weight)

    def global_test(self):
        model = self.nn
        c = clients_NB15
        Acc = 0
        for client in c:
            print('\n' + client + ':')
            model.file_name = client
            acc = test(self.args, model)
            Acc += acc
        print('Total accuracy:', Acc / 10)
