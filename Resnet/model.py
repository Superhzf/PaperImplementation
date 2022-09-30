import random
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import datetime


def load_data(batch_size):
    # mean = [0.49139968, 0.48215827 ,0.44653124]
    # std = [0.24703233,0.24348505,0.26158768]

    mean = [0.5, 0.5, 0.5]
    std = [0.5,0.5,0.5]

    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def train(args, net, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9,weight_decay=0.0001)
    training_start_time = datetime.datetime.now()
    for epoch in range(args.num_epochs):
        this_epoch_start_time = datetime.datetime.now()
        if epoch + 1 == 82 or epoch + 1 == 123:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        net.train()
        running_loss, total, correct = 0.0, 0, 0
        for images, labels in tqdm(trainloader, desc='Epoch ' + str(epoch), unit='b'):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        end_time = datetime.datetime.now()
        time_passed_epoch = (end_time - this_epoch_start_time).total_seconds()
        time_passed_all = (end_time - training_start_time).total_seconds()
        print('Epoch %d training loss: %.3f training accuracy: %.3f%%' % (
            epoch, loss, accuracy))
        print(f"Seconds passed in this epoch:{time_passed_epoch} | Seconds passed since the training starts:{time_passed_all}")


def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for images, labels in tqdm(testloader, desc='Test', unit='b'):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %.2f%%' % (
        100 * correct / total))


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
