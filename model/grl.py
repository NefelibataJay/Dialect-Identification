from typing import Any, Optional, Tuple
from torch import nn
import torch
from torch.autograd import Function
import math

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, alpha: Optional[float] = 1.) -> torch.Tensor:
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.alpha, None

class GRLLayer(nn.Module):
    def __init__(self):
        super(GRLLayer, self).__init__()

    def forward(self, input, alpha=1.0):
        return GradientReverseFunction.apply(input,alpha)

class GRLClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        self.grl = GRLLayer()

    def forward(self, x):
        return self.linear(x)

    def grl_forward(self, x):
        x = self.linear(x)
        x = self.grl(x)
        return x

import torch.nn.functional as F

if __name__ == '__main__':
    import random
    import numpy
    random.seed(0)
    torch.manual_seed(0)
    numpy.random.seed(0)

    net1 = GRLClassifier(3, 6)
    net2 = GRLClassifier(6, 4)
    net3 = GRLClassifier(4, 2)

    data = torch.rand((4, 3))
    label = torch.ones((4), dtype=torch.long)
    out = net3(net2(net1(data)))
    loss = F.cross_entropy(out, label)
    loss.backward()
    print("第一次前向传播，没有GRL层")
    print('net1.linear.weight.grad', net1.linear.weight.grad)
    print('net2.linear.weight.grad', net2.linear.weight.grad)
    print('net3.linear.weight.grad', net3.linear.weight.grad)
    print("第二次前向传播，没有GRL层")
    net1.zero_grad()
    net2.zero_grad()
    net3.zero_grad()

    out = net3(net2(net1(data)))
    loss = F.cross_entropy(out, label)
    loss.backward()
    print('net1.linear.weight.grad', net1.linear.weight.grad)
    print('net2.linear.weight.grad', net2.linear.weight.grad)
    print('net3.linear.weight.grad', net3.linear.weight.grad)

    print("第一二次，梯度相同。 证明当X输入不变的时候，对于同一个loss值，计算的梯度数值总是相同。")


    print("第三次： -------- 验证GRL")
    net1.zero_grad()
    net2.zero_grad()
    net3.zero_grad()

    out = net3(net2(net1.grl_forward(data)))  ## 这里 net1先经过 linear，再经过GRL
    ## 网络前向： Net1--->  GRL ---> net2--->  net3
    ## 网络反向：  net3---->  net2 ----> GRL--->  net1
    loss = F.cross_entropy(out, label)
    loss.backward()
    print('net1.linear.weight.grad', net1.linear.weight.grad)
    print('net2.linear.weight.grad', net2.linear.weight.grad)
    print('net3.linear.weight.grad', net3.linear.weight.grad)


    print("可以看到 第三次打印的net1的梯度和第一次 互反")
