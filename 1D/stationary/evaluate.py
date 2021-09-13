import torch
import torch.nn as nn
import numpy as np

from guess_beam import *

import matplotlib.pyplot as plt

def draw():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print("Current device:", device)
    model = Inference()
    
    
    model_ckpt = "./models/model_infer.data"
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)

    model.to(device)

    x_test = torch.from_numpy(np.arange(10001)/10000).type(torch.FloatTensor)
    # y_test = torch.from_numpy(np.arange(10001)/1000).type(torch.FloatTensor)
    # # u_test = u(x_test)

    # x_fig = x_test.unsqueeze(0).T
    # y_fig = y_test.unsqueeze(0).T
    # x, y = np.mgrid[0:1.01:0.01, 0:1.01:0.01]
    # xy = torch.from_numpy(np.vstack((x.flatten(), y.flatten()))).type(torch.FloatTensor)

    pred = model(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()
    
    plt.scatter(x_test, pred )
    plt.colorbar()
    plt.savefig('./figures/fig.png')

if __name__ == "__main__":
    # evaluate()
    draw()