# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 11:13
# @Author  : 石鑫磊
# @Site    : 
# @File    : data_load.py
# @Software: PyCharm 
# @Comment :
import numpy as np
from PIL import Image
import pickle
import os

CHANNEL = 3
WIDTH = 32
HEIGHT = 32

data = []
labels=[]
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(5):
    with open("../detaset/cifar-10-batches-py.tar/data_batch_"+ str(i+1),mode='rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
        data+= list(data_dict[b'data'])
        labels+= list(data_dict[b'labels'])

img =  np.reshape(data,[-1,CHANNEL, WIDTH, HEIGHT])

for i in range(img.shape[0]):

    r = img[i][0]
    g = img[i][1]
    b = img[i][2]

    ir = Image.fromarray(r)
    ig = Image.fromarray(g)
    ib = Image.fromarray(b)
    rgb = Image.merge("RGB", (ir, ig, ib))

    name = "img-" + str(i) +"-"+ ".png"
    data_path=os.path.join("detaset/images/",classification[labels[i]])
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    rgb.save(data_path +'/'+ name, "PNG")


