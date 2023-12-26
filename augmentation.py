import torch
from random import random, randint
import random as r
import cv2 
import numpy as np

def random_value(a, b):
    w1 = b - a
    return random() * w1 + a

def change_color_channel(x, **kwargs):

    n = x.shape[0]
    shape = x.shape[1:]
    result = torch.zeros((2,n,)+shape)
    result[0] = x

    cha1 = x[:,0,:,:].view((-1,1,)+shape[1:])
    cha2 = x[:,1,:,:].view((-1,1,)+shape[1:])
    cha3 = x[:,2,:,:].view((-1,1,)+shape[1:])
    channels = [cha1,cha2,cha3]
    indexes = [0,1,2]
    choices = r.choices(indexes, k=2)
    while choices[0] == choices[1]:
        choices = r.choices(indexes, k=2)
    channels[choices[0]], channels[choices[1]] = channels[choices[1]], channels[choices[0]]
    
    result[1] = torch.concatenate(channels, dim=1)
    return result.permute((1,0,2,3,4)).reshape((-1,)+shape).to("cuda"), 0
            

def cutout_color(x, m, cut_size=1, aug_data = None, **kwargs):
    n = x.shape[0]
    shape = x.shape[1:]
    result = torch.zeros((m+1,n,)+shape)
    result[0] = x

    if aug_data == None:
        colors = []
    else:
        colors = aug_data
    for ind in range(1,m+1):

        new_x= x.clone()
        if aug_data == None:
            random_color = torch.rand((1,3,cut_size,cut_size))
            random_x = randint(0,shape[2] - cut_size)
            random_y = randint(0,shape[1] - cut_size)
            colors.append([random_color,random_x,random_y])
        new_x[:,:,colors[-1][2]:colors[-1][2]+cut_size,colors[-1][1]:colors[-1][1]+cut_size] = colors[-1][0]
        result[ind] = new_x

    return result.permute((1,0,2,3,4)).reshape((-1,)+shape), colors

def noise(x, m, var, aug_data=None, **kwargs):

    n = x.shape[0]
    shape = x.shape[1:]
    result = torch.zeros((m+1,n,)+shape, device="cuda")
    result[0] = x

    if aug_data==None:
        noises = []
    else:
        noises = aug_data
    for ind in range(1,m+1):

        if aug_data == None:
            noise = np.random.normal(loc=0, scale=np.sqrt(var), size=shape)
            noises.append(noise)
        new_x= x.clone()
        for i in range(n):
            new_x[i] += torch.from_numpy(noises[-1]).to("cuda")
        result[ind] = new_x

    return result.permute((1,0,2,3,4)).reshape((-1,)+shape), noises
