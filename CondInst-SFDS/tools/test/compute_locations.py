# coding=UTF-8
import torch
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
#from adet.utils.comm import compute_locations

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    plt.plot(shift_x,shift_y,color='red',marker='.',linestyle='')
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    plt.plot(locations[:,0],locations[:,1],color='blue',marker='.',linestyle='')
    tests_x=torch.arange(
        0, w, step=1,
        dtype=torch.float32, device=device
    )
    tests_y=torch.arange(
        0, h, step=1,
        dtype=torch.float32, device=device
    )
    test_x,test_y=torch.meshgrid(tests_x,tests_y)
    test_x=test_x.reshape(-1)
    test_y=test_y.reshape(-1)
    plt.plot(test_x,test_y,color='green',marker='.',linestyle='')
    plt.grid(True)
    plt.show()
    return locations

if __name__ == '__main__':
    h = 4
    w = 4
    stride = 2
    device = 'cpu'

    loc = compute_locations(h, w, stride, device)
    print("loc's shape: ", loc.shape)
    print(loc)