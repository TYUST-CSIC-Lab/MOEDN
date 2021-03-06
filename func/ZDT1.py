# -*- coding: utf-8 -*-
# !/usr/bin/env python

import numpy as np

Dimention = 30
Func_num = 2
Bound = [0, 1]


def ZDT1(X):
    # x=np.array([0,0,0,0,0,0])
    f1 = F1(X)
    gx = g(X)
    f2 = F2(gx, X)
    return (f1, f2)

def F1(X):
    return X[0]


def F2(gx, X):
    x = X[0]
    f2 = gx * (1 - np.sqrt(x / gx))
    return f2


def g(X):
    g = 1 + 9 * (np.sum(X[1:]) / (len(X)- 1))
    return g

if __name__=="__main__":
    np.random.seed(0)
    p=np.random.rand(20)
    zdt1=ZDT1(p)
    print("p",p)
    print(zdt1)
