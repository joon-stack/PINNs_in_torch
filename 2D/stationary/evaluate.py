import torch
import torch.nn as nn
import numpy as np

from train_by_torch_heat import *

import matplotlib.pyplot as plt

def evaluate():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Current device:", device)
    model = PINN()
    
    
    model_ckpt = "./models/model.data"
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device)

    x_test = torch.tensor((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)).unsqueeze(0).T
    y_test = torch.tensor((x_test / 24 * (1 - x_test) * (1 + x_test - x_test * x_test)))

    x_test.to(device)

    print(device)

    print(model(x_test))
    print(y_test)

def draw():
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
    x, y = np.mgrid[0:2.02:0.02, 0:1.01:0.01]
    xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)

    pred = model(xy[0].unsqueeze(0).T, xy[1].unsqueeze(0).T)

    pred = pred.detach().numpy()
    
    plt.scatter(x, y, c=pred, cmap='Spectral')
    plt.colorbar()
    plt.savefig('./figures/fig4.png')

if __name__ == "__main__":
    # evaluate()
    draw()