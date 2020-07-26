#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: rnn regression.py
@time: 7/26/20 8:43 AM
@desc:
'''

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(1)

def get_data(is_show=False):

    steps = np.linspace(0, np.pi *2, 100, dtype=np.float32)

    x = np.sin(steps)
    y = np.cos(steps)

    if is_show:
        plt.plot(steps, x, label="input (sin)")
        plt.plot(steps, y, label="output (cos)")
        plt.legend()
        plt.show()

    return steps, x, y

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
                input_size=1,
                hidden_size=32,
                num_layers=1,
                batch_first=True,
            )


        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state_init=None):
        # x_shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size(hidden_size))
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, h_state = self.rnn(x, h_state_init)

        outs = []

        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))

        return torch.stack(outs, dim=1), h_state

def main():

    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
    loss_func = nn.MSELoss()
    steps, x_np, y_np = get_data()

    h_state = None
    for step in range(60):
        x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis])) # shape(batchsize, time_step, input_size)
        y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis])) # shape(batchsize, time_step, output_size)
        pre, h_state = rnn(x, h_state) # pre shape (batch_size, time_step(100), output_size(1))

        # repack the hidden state, break the connection from last iteration
        h_state = Variable(h_state.data)

        loss = loss_func(pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        plt.plot(steps, pre.data.numpy().flatten(), "b-")
        plt.plot(steps, y_np, 'r-')
        plt.show()


if __name__ == '__main__':
    main()
