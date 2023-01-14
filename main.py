"""
    总控制台
"""

from args import args_parser
from server import FedAvg


def main():
    args = args_parser()
    fedAvg = FedAvg(args)

    print('\n训练更新模型: ')
    fedAvg.server()

    print('\n测试模型性能: ')
    fedAvg.global_test()


if __name__ == '__main__':
    main()
