"""
    FedKS 服务器端
"""

import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.stats import ks_2samp  # KS检验

sys.path.append('../')
from client import train, test
from model import DNN

np.set_printoptions(threshold=np.inf)  # 打印完整list


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

            self.validation_set(self.args.B, index)

            self.distribution_difference()

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

    def validation_set(self, batch, index):
        """服务器验证集"""
        df = pd.read_csv('Dataset/Val.csv', encoding='gbk')

        y = df[['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance']]
        X = df.drop(['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance'], 1)
        X = X.astype(np.float32)

        # 数据集资料取整
        data_len = int(len(X) / batch) * batch
        X, y = X[:data_len], y[:data_len]

        all_pred = []
        for k in index:
            pred = self.nns[k].predict(X)
            all_pred.append(pred)

        y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5])

        KS_value = []
        for i in range(len(self.secure_server)):
            fpr, tpr, thresholds = roc_curve(y.ravel(), all_pred[i].ravel())
            ks_value = max(abs(fpr - tpr))
            KS_value.append(ks_value)

        print('KS_value:', KS_value)

        KS_value = np.array(KS_value)
        nonevil_index = np.where(KS_value > 0.5)
        nonevil_index = list(nonevil_index)  # 转为list

        secure_server = []
        for i in nonevil_index[0]:
            secure_server.append(self.secure_server[i])
        self.secure_server = secure_server

    def distribution_difference(self):
        """检测与其他客户端权重数据分布差异过大的客户端"""
        # KS检验得到"p-value"
        pvas = []
        for i in self.secure_server:
            pva = []
            for j in self.secure_server:
                pvalue = ks_2samp(np.array(self.nns[i].get_weights()[0]).flatten(),
                                  np.array(self.nns[j].get_weights()[0]).flatten()).pvalue
                pva.append(pvalue)
            pvas.append(pva)
        print(len(pvas))

        PVA_mean = []
        for i in range(len(pvas)):
            del pvas[i][i]
            pva_mean = np.mean(pvas[i])
            PVA_mean.append(pva_mean)
        print('PVA_mean:', PVA_mean)

        PVA_mean = np.array(PVA_mean)
        nonevil_index = np.where(PVA_mean > 0.05)
        nonevil_index = list(nonevil_index)  # 转为list

        secure_server = []
        for i in nonevil_index[0]:
            secure_server.append(self.secure_server[i])
        self.secure_server = secure_server

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
        c = ['NUSW-NB15-' + str(i + 1) for i in self.secure_server]
        Acc = 0
        for client in c:
            print('\n' + client + ':')
            model.file_name = client
            acc = test(self.args, model)
            Acc += acc
        print('Total accuracy:', Acc / len(self.secure_server))
