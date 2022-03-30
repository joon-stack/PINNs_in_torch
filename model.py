import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
from copy import copy
from torchsummary import summary as summary_

class PINN(nn.Module):
    def __init__(self, neuron_num, layer_num):
        super(PINN, self).__init__()

        layers = []

        for i in range(layer_num):
            if i == 0:
                layer = nn.Linear(2, neuron_num)
            elif i == layer_num - 1:
                layer = nn.Linear(neuron_num, 1)
            else:
                layer = nn.Linear(neuron_num, neuron_num)

            layers.append(layer)

        self.module1 = nn.Sequential(*layers)
    
    def forward(self, input):
        # input      = torch.cat([x, t], axis=1)
        act_func        = nn.Sigmoid()
        
        tmp = input
        for layer in self.module1:
            tmp = act_func(layer(tmp))
        
        return tmp

    def __call__(self, input):
        return self.forward(input)


    def calc_loss_f(self, input, target):
        u_hat = self(input)
        x, t = input

        u_hat_x     = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
        u_hat_t     = autograd.grad(u_hat.sum(), t, create_graph=True)[0]
        u_hat_x_x   = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]

        f = u_hat_t + u_hat * u_hat_x - (0.01/np.pi) * u_hat_x_x

        func = nn.MSELoss()
        return func(f, target)

    def calc_loss_by_tag(self, input, target, tag):
        loss = []
        loss_func = nn.MSELoss()
        for inp, tar, t in zip(list(input, target, tag)):
            if t == 1:
                loss.append(self.calc_loss_f(inp, tar))
            else:
                loss.append(loss_func(inp, tar))
        return torch.tensor(loss).float()
    
if __name__ == "__main__":
    a = PINN(5, 5)
    