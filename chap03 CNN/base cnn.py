#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: base cnn.py
@time: 7/25/20 4:38 PM
@desc:
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,     # n_filters
                kernel_size=5,       # filter size
                stride=1,
                # if want same width and length of this image after con2d,
                # padding =(kernel_size-1) / 2 if stride = 1.
                padding=2
            ),                       # output shape (16, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2) # output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32*7*7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x # return x for visualization


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

def show_img(img, label):
    plt.imshow(X=img, cmap="gray")
    plt.title("the num is {}".format(label))
    plt.show()

def visualization(lowDWeights, labels):
    from matplotlib import cm

    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize last layer")
    plt.show()



def main(visual_result=False):
    train_loader, test_x, test_y = load_data()

    cnn = CNN()
    print("cnn {}".format(cnn))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1000):
        for step, (x, y) in enumerate(train_loader):
            batch_x = Variable(x)
            batch_y = Variable(y)

            output = cnn(batch_x)[0]
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                test_output, last_layer = cnn(test_x)
                pre_label = torch.max(test_output.data, dim=1)[1].data.squeeze() # torch.max()[1] for get index
                accuracy = (pre_label == test_y).sum().item() / float(test_y.size(0))
                print("Epoch: {}, | train loss: {}, | test accuracy: {}".format(epoch, loss.data, accuracy))

                if visual_result:
                    # visulization of trained flatten layer (T-SNE)
                    from sklearn.manifold import TSNE
                    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:500, :])
                    labels = test_y.numpy()[:500]
                    visualization(lowDWeights=low_dim_embs, labels=labels)


if __name__ == '__main__':
    main()