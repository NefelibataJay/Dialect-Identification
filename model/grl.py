from typing import Any, Optional, Tuple
from torch import nn
import torch
from torch.autograd import Function
import math

class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(nn.Module):
    def __init__(self, lambda_=1.):
        super(GRL, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        # p = 训练步数/总的训练步数
        # a = 10 超参数
        # 2 / (1 + math.exp(-a * p)) - 1
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)

# class GradientReverseFunction(Function):
#     """
#     重写自定义的梯度计算方式
#     """
#     @staticmethod
#     def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
#         ctx.coeff = coeff
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
#         return grad_output.neg() * ctx.coeff, None


# class GRL(nn.Module):
#     def __init__(self):
#         super(GRL, self).__init__()

#     def forward(self, *input):
#         return GradientReverseFunction.apply(*input)


class NormalClassifier(nn.Module):

    def __init__(self, num_features, num_classes, GRL=None):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
        if GRL:
            self.grl = GRL()

    def forward(self, x):
        if getattr(self, 'grl', None) is not None:
            x = self.grl(x)  # 注意这里
        return self.linear(x)

import torch.nn.functional as F

if __name__ == '__main__':
    import random
    import numpy
    random.seed(0)
    torch.manual_seed(0)
    numpy.random.seed(0)

    net1 = NormalClassifier(3, 2)
    net2 = NormalClassifier(2, 3)
    net3 = NormalClassifier(3, 2)
    net4 = NormalClassifier(3, 2, GRL=GRL)

    data = torch.rand((4, 3))
    label = torch.ones((4), dtype=torch.int64)
    label2 = torch.zeros((4), dtype=torch.int64)
    
    out = net3(net2(net1(data)))
    out2 = net4(net2(net1(data)))
    loss = F.cross_entropy(out, label) + F.cross_entropy(out2, label2)
    print(f"loss:{loss}")
    loss.backward()

    print('net1.linear.weight.grad', net1.linear.weight.grad)
    print('net2.linear.weight.grad', net2.linear.weight.grad)
    print('net3.linear.weight.grad', net3.linear.weight.grad)
    print('net4.linear.weight.grad', net4.linear.weight.grad)
     
