#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: base rnn.py
@time: 7/26/20 6:59 AM
@desc:
'''

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
torch.manual_seed(1)

def show_img(img, label):
    plt.imshow(X=img, cmap="gray")
    plt.title("the num is {}".format(label))
    plt.show()

def load_data(is_show_img=False, BATCH_SIZE=50):
    train_datas = torchvision.datasets.MNIST(root="./mnist",
                                      train=True, # This is training data
                                      transform=torchvision.transforms.ToTensor(),
                                      # torch.FloatTensor of shape (color x Height x width)
                                      download=True) # download it if you don't have it
    test_datas = torchvision.datasets.MNIST(root="./mnist",
                                            train=False)

    if is_show_img:
        show_img(img=train_datas.train_data[0].numpy(), label=train_datas.train_labels[0].numpy())

    train_loader = Data.DataLoader(dataset=train_datas, batch_size=BATCH_SIZE, shuffle=True)

    # increase the dimension in dim -> increase channel
    test_x = Variable(torch.unsqueeze(test_datas.test_data,dim=1)).type(torch.FloatTensor)[:2000]/255
    test_y = test_datas.test_labels[:2000]


    return train_loader, test_x, test_y

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,      # number of rnn layer
            batch_first=True,  # input & output will has batch size as ls dimension. e.g.
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x_shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None) # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

def main():
    rnn = RNN()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01) # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()
    train_loader, test_x, test_y = load_data()
    for epoch in range(1000):
        for step, (x, y) in enumerate(train_loader):
            batch_x = Variable(x.view(-1, 28, 28))
            batch_y = Variable(y)

            output = rnn(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_out = rnn(test_x.view(-1, 28, 28))
                pre_label = torch.max(test_out, dim=1)[1].data.squeeze() # torch.max()[1] for get index
                accuracy = (pre_label == test_y).sum().item() / float(test_y.size(0))
                print("Epoch: {}, | train loss: {}, | test accuracy: {}".format(epoch, loss.data, accuracy))


if __name__ == '__main__':
    main()