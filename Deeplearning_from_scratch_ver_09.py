# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:45:05 2017

@author: KCH
"""
import sys, os
sys.path.append(os.pardir)
from dataset.mnist_data import load_mnist

(x_train, t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

import numpy as np
from PIL import Image
import _pickle as cPickle

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    
    return y



def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img=x_train[0]
label=t_train[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)

def get_data():
    (x_train, t_train), (x_test,t_test)=\
    load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    path=('E:/CH/1. 개인공부/★Software/Python/Deeplearning_from_scratch/dataset')
    with open(path + '/' + "sample_weight.pkl",'rb') as f:
        network=cPickle.load(f)
        
    return network

def predict(network, x):
    W1, W2, W3=network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    z3 = sigmoid(a3)
    y = softmax(a3)
    
    return y

x,t=get_data()
network = init_network()

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
