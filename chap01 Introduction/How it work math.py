#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: How it work math.py
@time: 7/24/20 11:08 PM
@desc:
'''

import torch

x = torch.randn(3,4)

def test1():
    # adding value to the existing tensor, scalar addition
    torch.add(x,20)
    # >>>tensor([[18.8255, 17.5330, 21.2946, 19.3442],
    #            [19.3348, 20.8189, 19.0248, 19.8465],
    #            [19.7656, 18.3401, 20.1747, 19.9280]])

    # scalar multiplication
    torch.mul(x, 2)

def test2():

    torch.manual_seed(1)
    a = torch.randn(2,3)
    # >>>tensor([[0.6614, 0.2669, 0.0617],
    # >>>        [0.6213, -0.4519, -0.1661]])

    torch.manual_seed(1)
    b = torch.ceil(torch.randn(2,3))
    # >>>tensor([[1., 1., 1.],
    #            [1., -0., -0.]])

    torch.manual_seed(1)
    c = torch.floor(torch.randn(2,3))
    # >>>tensor([[ 0.,  0.,  0.],
    #            [ 0., -1., -1.]])

    # Limiting the values of any tensor within a certain range
    torch.manual_seed(1)
    d = torch.clamp(torch.randn(2,3), min=0.3, max=0.4)
    # >>>tensor([[0.4000, 0.3000, 0.3000],
    #            [0.4000, 0.3000, 0.3000]])

def test3():
    # get the exponential of a tensor
    torch.exp(x)    # compute the exponential of a tensor

    torch.frac(x)   # get the fractional of each tensor. eg 9.25 -> 0.25

    torch.log(x)    # compute the log of the value in a tensor

    torch.pow(x, 2) # to rectify the negative values do a power tranforamtion\

def test4():

    torch.sigmoid(x)

    torch.sqrt(x) # finding the square root of the value

def test5():
    from torch.autograd import Variable

    x = Variable(torch.Tensor(2, 3).uniform_(-4, 5))
    # >>> tensor([[-0.9011, -0.8061, -2.8618],
    #            [-3.0827, -1.3738, 2.3349]])

    y = Variable(torch.Tensor(3, 2).uniform_(-3, -2))
    # >>> tensor([[-2.0059, -2.2399],
    #            [-2.4439, -2.8637],
    #            [-2.2635, -2.2959]])

    # matrix multiplication
    z = torch.mm(x, y)
    # >>> tensor([[10.2553, 10.8971],
    #             [4.2561, 5.4788]])
    return z



if __name__ == '__main__':
    test5()