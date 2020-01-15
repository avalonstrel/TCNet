import torch
import random
import numpy as np

def one_hot(labels, dim):
    """Convert label indices to one-hot vector"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.squeeze().long()] = 1
    return out

def merge_list(lists):
    res = []
    for l in lists:
        res += l
    return set(res)

def randomflip(x, p=0.5):
    # x: (chn, h, w)
    if random.random() > p:
        return x

    w = x.size(2)

    dim = len(x.shape) - 1

    idxs = torch.arange(w-1,-1,-1).long()
    return x.index_select(index=idxs, dim=dim)

class crop_gen:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, x, mode='random'):
        size = self.crop_size
        w = x.size(2)
        if mode == 'center':
            start = (w-size) // 2
            x = x[:,:, start:start+size, start:start+size]
        if mode == 'upleft':
            startx = starty = 0
            x = x[:,:, startx:startx+size, starty:starty+size]
        if mode == 'upright':
            startx = w - size
            starty = 0
            x = x[:,:, startx:startx+size, starty:starty+size]
        if mode == 'downleft':
            startx = 0
            starty = w - size
            x = x[:,:, startx:startx+size, starty:starty+size]
        if mode == 'downright':
            startx = w - size
            starty = w - size
            x = x[:,:, startx:startx+size, starty:starty+size]
        if mode == 'random':
            startx = random.randint(0, (w-size))
            starty = random.randint(0, (w-size))
            x = x[:, startx:startx+size, starty:starty+size]
        return x