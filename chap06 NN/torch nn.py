#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: torch nn.py
@time: 7/27/20 1:33 PM
@desc:
'''
import torch
import numpy as np

batch = 64
input_dim = 1000
hidden_dim = 100
out_dim = 10

lr = 1e-6


x = torch.randn(batch, input_dim)
y = torch.randn(batch, out_dim)

w_1 = torch.randn(input_dim, hidden_dim)
w_2 = torch.randn(hidden_dim, out_dim)

for epoch in range(500):
    h = x.mm(w_1)
    h_relu = h.clamp(min=0)
    y_pre = h_relu.mm(w_2)

    # compute loss
    loss = (y_pre-y).pow(2).sum() / len(x)

    print("loss {}".format(loss))

    # compute gradient
    grad_y_pre = 2.0 * (y_pre-y)

    grad_w_2 = h_relu.t().mm(grad_y_pre)

    grad_h_relu = grad_y_pre.mm(w_2.t())

    grad_h_relu[h<0] = 0

    grad_w_1 = x.t().mm(grad_h_relu)

    # update weight of w1 and w2
    w_1 -= lr*grad_w_1
    w_2 -= lr*grad_w_2