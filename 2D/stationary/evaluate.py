import torch
import torch.nn as nn
import numpy as np

from train_by_torch_heat import *

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


    x, y = np.mgrid[0:1.01:0.01, 0:1.01:0.01]
    xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)

    pred = model(xy[0].unsqueeze(0).T, xy[1].unsqueeze(0).T)

    pred = pred.detach().numpy()

    output = np.vstack((x.flatten(), y.flatten(), pred.flatten()))

    # np.save('./data/output.npy', output)

    # with open('./data/output.npy', 'rb') as f:
    #     a = np.load(f)
    #     print(a)

    plt.scatter(x, y, c=pred, cmap='hot')
    plt.colorbar()
    plt.title("Temperature in PINN (degC)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_heat_transfer_sta.png')
    
def draw():
    fname = "./data/FEM/heat_transfer_stationary.txt"
    data = np.loadtxt(fname=fname)

    x, y, temp = data.T

    plt.cla()
    plt.scatter(x, y, c=temp, cmap='hot')
    plt.title("Temperature in FEM (degC)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_heat_transfer_sta_fem.png')
    
if __name__ == "__main__":
    evaluate()
    # draw()
    draw()