#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: gradien computation.py
@time: 7/25/20 11:58 AM
@desc:
'''
import torch
from torch.autograd import Variable

class simple_model():
    def __init__(self, w):
        self.w = w
    def froward(self, x, w):
        """
        Using forward pass
        :param x:
        :return:
        """
        return x * w

    def loss_func(self, x, y):
        y_pre = self.froward(x, self.w)
        return (y-y_pre)*(y-y_pre)

def main():
    x_data = [1, 2, 3]
    y_data = [4, 5, 6]

    w = Variable(torch.Tensor([1.0]), requires_grad=True)
    model = simple_model(w=w)

    # Run the training loop
    for epoch in range(10):
        for x_val, y_val in zip(x_data, y_data):
            loss = model.loss_func(x_val, y_val)
            loss.backward()
            print("grad: x_val {}, y_val {}, w_grad {}".format(x_val, y_val, w.grad.data[0]))
            w.data = w.data - 0.01 * w.grad.data[0]
            w.grad.data.zero_()
        print("process: epoch {}, loss {}".format(epoch, loss.data[0]))

if __name__ == '__main__':
    main()