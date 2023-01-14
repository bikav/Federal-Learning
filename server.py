"""
    FedMedian 服务器端
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
            weights.append(weight)

        w_1 = []  # 52层
        for i in range(52):
            temp = []
            for j in range(128):
                nums = []
                for k in range(10):
                    nums.append(weights[k][0][i][j])
                temp.append(np.median(nums))
            w_1.append(temp)
        w_1 = np.array(w_1)

        b_1 = []  # 128层
        for i in range(128):
            nums = []
            for j in range(10):
                nums.append(weights[j][1][i])
            b_1.append(np.median(nums))
        b_1 = np.array(b_1)

        w_2 = []  # 128层
        for i in range(128):
            temp = []
            for j in range(128):
                nums = []
                for k in range(10):
                    nums.append(weights[k][2][i][j])
                temp.append(np.median(nums))
            w_2.append(temp)
        w_2 = np.array(w_2)

        b_2 = []  # 128层
        for i in range(128):
            nums = []
            for j in range(10):
                nums.append(weights[j][3][i])
            b_2.append(np.median(nums))
        b_2 = np.array(b_2)

        w_3 = []  # 128层
        for i in range(128):
            temp = []
            for j in range(64):
                nums = []
                for k in range(10):
                    nums.append(weights[k][4][i][j])
                temp.append(np.median(nums))
            w_3.append(temp)
        w_3 = np.array(w_3)

        b_3 = []  # 64层
        for i in range(64):
            nums = []
            for j in range(10):
                nums.append(weights[j][5][i])
            b_3.append(np.median(nums))
        b_3 = np.array(b_3)

        w_4 = []  # 64层
        for i in range(64):
            temp = []
            for j in range(32):
                nums = []
                for k in range(10):
                    nums.append(weights[k][6][i][j])
                temp.append(np.median(nums))
            w_4.append(temp)
        w_4 = np.array(w_4)

        b_4 = []  # 32层
        for i in range(32):
            nums = []
            for j in range(10):
                nums.append(weights[j][7][i])
            b_4.append(np.median(nums))
        b_4 = np.array(b_4)

        w_5 = []  # 32层
        for i in range(32):
            temp = []
            for j in range(16):
                nums = []
                for k in range(10):
                    nums.append(weights[k][8][i][j])
                temp.append(np.median(nums))
            w_5.append(temp)
        w_5 = np.array(w_5)

        b_5 = []  # 16层
        for i in range(16):
            nums = []
            for j in range(10):
                nums.append(weights[j][9][i])
            b_5.append(np.median(nums))
        b_5 = np.array(b_5)

        w_6 = []  # 16层
        for i in range(16):
            temp = []
            for j in range(6):
                nums = []
                for k in range(10):
                    nums.append(weights[k][10][i][j])
                temp.append(np.median(nums))
            w_6.append(temp)
        w_6 = np.array(w_6)

        b_6 = []  # 16层
        for i in range(6):
            nums = []
            for j in range(10):
                nums.append(weights[j][11][i])
            b_6.append(np.median(nums))
        b_6 = np.array(b_6)

        update_weight = []
        update_weight.append(w_1)
        update_weight.append(b_1)
        update_weight.append(w_2)
        update_weight.append(b_2)
        update_weight.append(w_3)
        update_weight.append(b_3)
        update_weight.append(w_4)
        update_weight.append(b_4)
        update_weight.append(w_5)
        update_weight.append(b_5)
        update_weight.append(w_6)
        update_weight.append(b_6)

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
