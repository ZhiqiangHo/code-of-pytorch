#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: sampling.py
@time: 7/25/20 9:45 AM
@desc:
'''

import torch

torch.manual_seed(1)
torch.randn(2,3)

def test1():
    torch.Tensor(2,3).uniform_(0,1)    # continuous uniform distribution
    # >>>tensor([[0.6826, 0.3051, 0.4635],
    #            [0.4550, 0.5725, 0.4980]])

def test2():
    # calculate probability mass function instead of probability density function

    torch.bernoulli(torch.Tensor(2,3).uniform_(0,1))
    # >>> tensor([[0., 0., 1.],
    #             [1., 1., 1.]])

def test3():
    # the multinomial function picks up; returns the result as an index position for the tensors
    torch.multinomial(torch.tensor([10., 10., 13., 10.,
                                        34., 45., 65., 67.,
                                        87., 89., 87., 34.]), 3, replacement=False)
    # >>> tensor([7, 9, 8])

def test4():
    torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1))
    # >>> tensor([-0.5228,  2.3435,  2.1779,  3.6059,  4.4646,  5.9709,  6.9218,  7.7103,
    #         9.0845, 10.0267])

def test5():
    torch.mean(torch.randn(2,3))
    # >>> tensor(-0.6137)

    torch.mean(torch.randn(2,3), dim=0)
    # >>> tensor([ 0.0359, -0.6934, -0.0441])

    torch.mean(torch.randn(2,3), dim=1)
    # >>> tensor([ 0.6304, -0.5032])

def test6():
    """
    # Median, mode, and standard deviation computation can be written in the same way
    1. torch.median()
    2. torch.mode()
    3. torch.std()
    4. tor.var()
    :return:
    """

    torch.median(torch.randn((2,3)))
    # >>> tensor(-1.5228)

    torch.median(torch.randn(2, 3), dim=0)
    # >>> torch.return_types.median(
    #        values=tensor([-0.1955, -0.9656, -0.5107]),
    #        indices=tensor([0, 0, 1]))

    torch.median(torch.randn(2, 3), dim=1)
    # >>> torch.return_types.median(
    #        values=tensor([-0.1232, -1.2770]),
    #        indices=tensor([1, 2]))


if __name__ == '__main__':
    test6()