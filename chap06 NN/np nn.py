#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: np nn.py
@time: 7/27/20 8:23 AM
@desc:
'''

import numpy as np

batch = 64
input_dim = 1000
hidden_dim = 100
out_dim = 10

lr = 1e-6


x = np.random.randn(batch, input_dim)
y = np.random.randn(batch, out_dim)

w_1 = np.random.randn(input_dim, hidden_dim)
w_2 = np.random.randn(hidden_dim, out_dim)

for epoch in range(500):
    h = x.dot(w_1)
    h_relu = np.maximum(h, 0)
    y_pre = h_relu.dot(w_2)

    # compute loss
    loss = sum(np.sqrt((y_pre-y)**2)) / len(y)

    print("loss {}".format(loss))

    # compute gradient
    grad_y_pre = 2.0 * (y_pre-y)

    grad_w_2 = h_relu.T.dot(grad_y_pre)

    grad_h_relu = grad_y_pre.dot(w_2.T)

    grad_h_relu[h<0] = 0

    grad_w_1 = x.T.dot(grad_h_relu)

    # update weight of w1 and w2
    w_1 -= lr*grad_w_1
    w_2 -= lr*grad_w_2