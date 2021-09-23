import torch
import torch.nn as nn
import numpy as np

import argparse

from train_by_torch_heat_tran import *

import matplotlib.pyplot as plt



def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Current device:", device)
    model = PINN()
    
    
    model_ckpt = "./models/model.data"
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device)

    # x_test = torch.from_numpy(np.arange(10001)/500).type(torch.FloatTensor)
    # y_test = torch.from_numpy(np.arange(10001)/1000).type(torch.FloatTensor)
    # # u_test = u(x_test)

    # x_fig = x_test.unsqueeze(0).T
    # y_fig = y_test.unsqueeze(0).T
    x, y = np.mgrid[0:1.01:0.01, 0:1.01:0.01]
    xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)
    t = torch.from_numpy(np.ones(xy[0].shape) * 1).type(torch.FloatTensor)

    pred = model(xy[0].unsqueeze(0).T, xy[1].unsqueeze(0).T, t.unsqueeze(0).T)

    pred = pred.detach().numpy()
    
    plt.scatter(x, y, c=pred, cmap='hot')
    plt.colorbar()
    plt.title("Temperature in PINN (degC), time=1s")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_heat_transfer_1s.png')

def draw():
    fname = "./data/FEM/heat_transfer_transient_01s.txt"
    data = np.loadtxt(fname=fname)

    x, y, temp = data.T

    plt.cla()
    plt.scatter(x, y, c=temp, cmap='hot')
    plt.title("Temperature in FEM (degC), time=0.1s")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_heat_transfer_transient_01s.png')

if __name__ == "__main__":
    evaluate()
    draw()