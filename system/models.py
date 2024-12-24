import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.quantization import QuantStub, DeQuantStub, prepare, convert


def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same')
        self.fc1 = nn.Linear(12896, 128)

        self.parametrized_layers = [self.conv1, self.conv2, self.conv3, self.fc1]

        for layer in self.parametrized_layers:
            init_layer(layer)

        # 添加量化和去量化模块
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dequant(x)
        return x


def ConvNet():
    return SimCLRNet()