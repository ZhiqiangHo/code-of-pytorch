#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: How it work pro.py
@time: 7/24/20 10:25 PM
@desc:
'''

import torch

def test1():
    num = torch.randn(4, 4) # shape=(4,4)
    index = torch.LongTensor([0, 2])
    torch.index_select(num, dim=0, index=index) # shape=(2, 4); if dim=1 the shape=(4, 2)

def test2():
    """
    # identify non-zero elements in a large tensor
    :return:
    """

    # identify null input tensors using nonzero function
    torch.nonzero(torch.tensor([10, 0, 23, 0, 0.0])) # return: tensor([[0],[2]])

def test3():
    """
    The split function splits a long tensor into smaller tensors
    :return:
    """
    torch.split(torch.tensor([1,2,3,4,5,6]), split_size_or_sections=2) # Return: (tensor([1, 2]), tensor([3, 4]), tensor([5, 6]))

def test4():
    # reshape the tensor along a new dimension
    num = torch.randn(3, 4) # shape=(3,4)

    # transpose is one option to change the shape of the tensor
    num.t()         # shape=(4,3); same result will get num.transpose(1, 0)

def test5():
    # The unbind function removes a dimension from a tensor

    num = torch.randn(3, 4)  # shape=(3,4)
    # >>>tensor([[-0.6782, 0.4237, -0.0908, -0.1366],
    #            [0.6513, 2.7114, 0.9232, 0.7910],
    #            [-1.0527, -1.1931, -0.3357, 0.6662]])

    torch.unbind(num, dim=1) # dim=1 removing a columns, dim=0 removing a row
    # >>> (tensor([-0.6782,  0.6513, -1.0527]),
    #      tensor([ 0.4237,  2.7114, -1.1931]),
    #      tensor([-0.0908,  0.9232, -0.3357]),
    #      tensor([-0.1366,  0.7910,  0.6662]))





if __name__ == '__main__':
    test5()