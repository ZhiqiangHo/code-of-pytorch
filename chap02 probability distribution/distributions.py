#!/usr/bin/env python
# encoding: utf-8
'''
@author: Zhiqiang Ho
@contact: 18279406017@163.com
@file: distributions.py
@time: 7/25/20 2:15 PM
@desc:
'''

import torch

def test1():
    from torch.distributions.bernoulli import Bernoulli

    # Creates a Bernoulli distribution parameterized by probs
    dist = Bernoulli(torch.tensor([0.1, 0.5, 0.9]))

    # Samples are binary (0 or 1). They take the value 1 with probability p
    dist.sample()  # >>> tensor([0., 0., 1.])

def test2():
    """
    beta distribution is a family of continuous random variables defined in the range of 0 and 1.
    :return:
    """
    from torch.distributions.beta import Beta
    dist = Beta(torch.tensor([0.5]), torch.tensor(0.5))
    dist.sample()   # >>> tensor([0.0594])

def test3():
    """
    The binomial distribution
    :return:
    """
    from torch.distributions.binomial import Binomial
    # 100-count of trials: 0, 0.2, 0.8 and 1 are event probabilities
    dist = Binomial(100, torch.tensor([0, 0.2, 0.8, 1]))
    dist.sample()  # tensor([  0.,  19.,  72., 100.])

def test4():
    """
    categorical distribution can be defined as a generalized Bernoulli distribution
    :return:
    """
    from torch.distributions.categorical import Categorical
    dist = Categorical(torch.tensor([0.1, 0.2, 0.3, 0.4])) # 0.1, 0.2, 0.4, 0.4 event probabilities
    dist.sample() # >>> tensor(3)

def test5():
    """
    Laplacian distribution is a continuous probability distribution
    parameterized by loc and scale
    :return:
    """
    from torch.distributions.laplace import Laplace
    dist = Laplace(torch.tensor([10.0]), torch.tensor([0.99]))
    dist.sample() # >>> tensor([10.2167])

def test6():
    """
    Normal Gaussian distribution parameterized by loc and scale
    :return:
    """
    from torch.distributions.normal import Normal
    dist = Normal(torch.tensor([100.0]), torch.tensor([10.0]))
    dist.sample() # >>> tensor([95.2452])

if __name__ == '__main__':
    test4()