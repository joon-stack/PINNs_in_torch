import torch
import torch.nn as nn
import numpy as np

import os 

from forced_train_by_torch import *
from train_by_torch_beam import *

import matplotlib.pyplot as plt

def singlemodel():
    model = PINN()
    model_ckpt = "./models/model_single.data"
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)
    x_test = torch.from_numpy(np.arange(10001)/10000).type(torch.FloatTensor)

    pred = model(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()
    exact = u(x_test)
    plt.scatter(x_test, pred, marker='-')
    plt.scatter(x_test, exact, marker='-', c='red')
    plt.colorbar()
    plt.savefig('./figures/fig_single.png')




def draw():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Current device:", device)
    nn_1 = PINN()
    nn_2 = PINN()
    model = EnsembleModel((nn_1, nn_2))

    model_ckpt = "./models/nn_1.data"
    if os.path.isfile(model_ckpt):
        state_dict = torch.load(model_ckpt)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        nn_1.load_state_dict(state_dict)

    model_ckpt = "./models/nn_2.data"
    if os.path.isfile(model_ckpt):
        state_dict = torch.load(model_ckpt)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        nn_2.load_state_dict(state_dict)


    model_ckpt = "./models/model.data"
    if os.path.isfile(model_ckpt):
        state_dict = torch.load(model_ckpt)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        model.load_state_dict(state_dict)

    model.to(device)

    x_test = torch.from_numpy(np.arange(10001)/10000).type(torch.FloatTensor)

    pred = model(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()
    
    plt.scatter(x_test, pred, marker='.')
    plt.colorbar()
    plt.savefig('./figures/fig.png')

    plt.cla()
    pred = nn_1(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()

    plt.scatter(x_test, pred, marker='.')
    plt.savefig('./figures/forced_nn_1.png')

    plt.cla()
    pred = nn_2(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()

    plt.scatter(x_test, pred, marker='.')
    plt.savefig('./figures/forced_nn_2.png')
    plt.cla()

    pred = distance(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()

    plt.scatter(x_test, pred, marker='.')
    plt.savefig('./figures/forced_dist.png')
    plt.cla()


if __name__ == "__main__":
    # evaluate()
    # draw()
    singlemodel()