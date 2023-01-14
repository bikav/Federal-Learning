"""
    FedTrimmed-mean args
"""

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--E', type=int, default=40, help='本地模型训练次数')
    parser.add_argument('--r', type=int, default=12, help='全局训练次数')
    parser.add_argument('--K', type=int, default=10, help='客户端总数')
    parser.add_argument('--input_dim', type=int, default=52, help='输入维度')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--C', type=float, default=0.5, help='抽样率')
    parser.add_argument('--B', type=int, default=500, help='本地批量大小')
    parser.add_argument('--optimizer', type=str, default='adam', help='优化器')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减，每轮全球学习率下降')
    clients = ['NUSW-NB15-' + str(i) for i in range(1, 11)]
    parser.add_argument('--clients', default=clients)

    args = parser.parse_args()

    return args
