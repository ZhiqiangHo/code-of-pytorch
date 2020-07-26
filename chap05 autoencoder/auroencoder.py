#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: auroencoder.py
@time: 7/26/20 10:56 AM
@desc:
'''

import torchvision
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
    test_x = Variable(torch.unsqueeze(test_datas.data,dim=1)).type(torch.FloatTensor)[:2000]/255
    test_y = test_datas.targets[:2000]


    return train_loader, test_x, test_y

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Dropout(0.5),  # a typicel threshold for application dropout rate is 20% to 50%
            nn.Tanh(),

            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.Tanh(),

            nn.Linear(64, 12),
            nn.Dropout(0.5),
            nn.Tanh(),

            nn.Linear(12, 3), # cpmpress to 3 features which can be visualized in plt
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Dropout(0.5),
            nn.Tanh(),

            nn.Linear(12, 64),
            nn.Dropout(0.5),
            nn.Tanh(),

            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.Tanh(),

            nn.Linear(128, 28*28),
            nn.Sigmoid(), # compress to a range(0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoder = self.decoder(encoded)
        return encoded, decoder

def visual_orin_decoder(N_TEST_IMG=5, view_data=None, decoded_data=None):
    f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

    for i in range(N_TEST_IMG):
        a[0][i].imshow(np.reshape(view_data.data[i], (28, 28)), cmap="gray")
        a[1][i].imshow(np.reshape(decoded_data.data[i], (28, 28)), cmap="gray")

        a[0][i].set_xticks(())
        a[0][i].set_yticks(())

        a[1][i].set_xticks(())
        a[1][i].set_yticks(())

    plt.show()

def main(EPOCH=10, N_TEST_IMG=5):
    train_loader, test_x, test_y = load_data()
    autoencoder = AutoEncoder()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005)
    loss_func = nn.MSELoss()

    for epoch in range(EPOCH):
        for step ,(x, y) in enumerate(train_loader):
            batch_x = Variable(x.view(-1, 28*28))

            encoded, decoded = autoencoder(batch_x)

            loss = loss_func(decoded, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 500 == 0 and epoch in [0, 5, EPOCH-1]:

                autoencoder.eval() # parameters for dropout differ from train mode

                print("Epoch: {} | train loss: {}".format(epoch, loss.data.numpy()))
                view_data = test_x.view(-1, 28*28)

                encoded_data, decoded_data = autoencoder(view_data[:5,:])

                visual_orin_decoder(view_data=view_data, decoded_data=decoded_data)


if __name__ == '__main__':
    main()