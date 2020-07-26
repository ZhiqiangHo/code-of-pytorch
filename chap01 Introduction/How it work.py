#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: How it work.py
@time: 7/24/20 5:32 PM
@desc:
'''

import torch
import numpy as np

def test1():
    """
    # check whether an object in Python is a tensor
    :return: bool
    """
    x = [0, 1, 2]
    print("whether x is tensor :", torch.is_tensor(x))    # check whether is tensor -> False
    print("whether x is storage :",  torch.is_storage(x)) # check whether is stored -> False

    y = torch.randn(3,2) # shape=(3, 2) / torch.zeros(3, 2)
    print("whether y is tensor :", torch.is_tensor(y))    # check whether is tensor -> True
    print("whether y is storage :", torch.is_storage(y))  # check whether is stored -> False

    print("the total number of elements in the input Tensor is : {}".format(torch.numel(y)))

def test2():
    # the eye function creates a diagonal matrix, of witch the diagonal elements have ones
    print("the eye function creates diagonal matrix is :{}".format(torch.eye(2,3)))

def test3():
    x = [0, 1, 2]
    np_x = np.array(x)
    torch.from_numpy(np_x)  # convert array to tensor

def test4():
    x1 = torch.linspace(2, 10, steps=25) # creating 25 points in linear spacing
    x2 = torch.logspace(start=-10, end=10, steps=15) # logarithmic spacing

    print(x1, x2)

def test5():
    # random numbers from a uniform distribution between the values 0 and 1
    torch.rand(10)

    # random values between 0 and 1 fillied with a matrix of shape (4, 5)
    torch.rand(4, 5)

    # random numbers from a normal distribution with mean=0 and standard deviation=1
    torch.randn(10)

    # random values from a normal distribution fillied with a matrix of shape (4, 5)
    torch.randn(4, 5)

def test6():
    # select random values from a range of values
    torch.randperm(10) # selecting values from a range

    # usage of range function
    torch.arange(10, 40, 2) # step size 2

def test7():
    num = torch.randn(2, 3)
    torch.argmax(num, dim=1)

def test8():
    x = torch.randn(4,5) # shape(4, 5)
    torch.cat((x,x))     # shape(8, 5)

    # concatenate n times based on array size, over column
    torch.cat((x,x,x), 1) # shape(4,15)

    # concatenate n times based on array size, over rows
    torch.cat((x,x,x), 0) # shape(12,5)

    torch.stack([x, x], dim=0) # shape(2, 4, 5)

def test9():
    num = torch.randn(4, 4)

    # Split Tensor
    torch.chunk(num, 2, 0)  # 2 * shape=(2, 4)

    # Split Tensor
    torch.chunk(num, 2, 1)  # 2 * shape=(4, 2)

def test10():
    """
    # gather: collects elements from a tensor and places it in another tensor using an index argument
    :return:
    """
    num = torch.Tensor([[11, 12], [23, 24]])

    torch.gather(num, dim=1, index=torch.LongTensor([[0, 1], [1, 0]])) # row=1, selection along columns

if __name__ == '__main__':
    test10()
