import torch
import torch.nn as nn
import numpy as np
import random


class LinearNet(nn.Module):
    def __init__(self, n_feature, n_class):
        super(LinearNet, self).__init__()
        self.linear1 = nn.Linear(n_feature, n_class, bias=False)
    # forward 定义前向传播

    def forward(self, x):
        y = self.linear1(x)
        return y


def linear_net(num_inputs, num_output=2):
    net = LinearNet(num_inputs, num_output)
    print(net)
    return net
