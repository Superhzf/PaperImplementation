import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding

    Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L40
    The padding parameter is always 1 such that when stride = 1, the output size
    will be the same as the input size; when stride = 2, the output size is cut
    by half.

    Changes I make:
    1. I changed value of padding from dilation to 1.
    2. I simplified the code by removing the groups and dilation parameters
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution

    Copy and paste from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L54
    I don't change anything.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L59
    I simplified the code by removing the groups and dilation parameters

    Parameters:
    ----------------------------------------------------------------------------
    expansion: int
        This parameter is used for reducing the number of trainable parameters by
        the bottleneck block. The value is here to be consistent. And the value
        is fixed to 1.
    inplanes: int
        The number of input channels of the block.
    planes: int
        The number of output channels of the block.
    stride: int
        The stride size as in the vanilla convolution layer.
    downsample: nn.Module
        The Pytorch object to project the identity to different space. It
        is necessary when the dimension of the identity and the output of the
        residual block is different.
    norm_layer: nn.Module
        The Pytorch object to normalize the output of the convolution layer.
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L108
    Changes I make:
    1. I change the code to follow the original implementation by placing the stride
    at the first 1x1 convolution(self.conv1).
    2. I simplified the code by removing the groups and dilation parameters

    Ref:
    https://arxiv.org/pdf/1512.03385.pdf

    Parameters:
    ----------------------------------------------------------------------------
    expansion: int
        The times to increase the number of output channels.
    inplanes: int
        The number of input channels of the block.
    stride: int
        The stride size as in the vanilla convolution layer.
    downsample: nn.Module
        The Pytorch object to project the identity to different space. It
        is necessary when the dimension of the identity and the output of the
        residual block is different.
    norm_layer: nn.Module
        The Pytorch object to normalize the output of the convolution layer.
    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        version = args.resnet_version
        self.resnet_size = args.resnet_size
        # number of blocks for each part
        layers = [self.resnet_size,self.resnet_size,self.resnet_size]
        if version == 1:
            block = BasicBlock
        elif version == 2:
            block = Bottleneck
        else:
            assert 1 == 0, f"The version number {version} is not understood!"

        self.num_classes = args.num_classes
        self.drop_out = args.drop
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block:Type[Union[BasicBlock, Bottleneck]],planes:int,blocks:int,stride:int = 1)-> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        #Either changing the dimension of identity or subsampling will be
        #implemented in the first layer of a block
        layers.append(block(self.inplanes,planes,stride,downsample,norm_layer))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,planes,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x = self._forward_impl(x)
        ### END YOUR CODE
        return x
