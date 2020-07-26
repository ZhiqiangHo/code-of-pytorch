#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: reloading model.py
@time: 7/25/20 10:20 PM
@desc:
'''

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

# sample data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + torch.randn(x.size()) # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)

def save():
    # save net1
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        pre = net1(x)
        loss = loss_func(pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("Net1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), "r-", lw=5)

    # 2 ways to save the net
    torch.save(net1, "net.pkl") # save entire net
    torch.save(net1.state_dict(), "net_params.pkl") # save only the parameters

def restore_net():
    # restore entire net1 to net2
    net2 = torch.load("net.pkl")
    pre = net2(x)

    # plot result
    plt.subplot(132)
    plt.title("Net2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), "r-", lw=5)

def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net's parameters into net3
    net3.load_state_dict(torch.load("net_params.pkl"))
    pre = net3(x)

    # plot result
    plt.subplot(133)
    plt.title("Net3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pre.data.numpy(), 'r-', lw=5)

if __name__ == '__main__':
    save()
    restore_net()
    restore_params()
    plt.show()