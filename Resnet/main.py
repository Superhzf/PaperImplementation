# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
import argparse
from network import ResNet
from model import set_random, load_data, train, test
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-resnet_version', type=int, default=1, help='ResNet version')
    parser.add_argument('-resnet_size', type=int, default=18, help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument('-num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='dropout rate')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments are parsed!")
    print("args.seed",args.seed)
    print("args.resnet_version",args.resnet_version)
    print("args.resnet_size",args.resnet_size)
    print("args.num_classes",args.num_classes)
    print("args.batch",args.batch)
    print("args.lr",args.lr)
    print("args.drop",args.drop)
    set_random(args.seed)
    trainloader, testloader = load_data(args.batch)
    net = ResNet(args)
    # l = [module for module in net.modules() if not isinstance(module, nn.Sequential)]
    # print(l)
    # exit(0)
    if torch.cuda.is_available():
        net.cuda()
    train(args, net, trainloader)
    test(net, testloader)
