import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time


from copy import copy

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
        act_func        = nn.Tanh()
        
        tmp = input
        for layer in self.module1:
            tmp = act_func(layer(tmp))
        
        return tmp

    def __call__(self, input):
        return self.forward(input)

    
    def calc_loss_f(self, input, target):
        print(input)
        u_hat = self(input)
        
        deriv_1 = autograd.grad(u_hat.sum(), input, create_graph=True)
        u_hat_x = deriv_1[0][:, 0].reshape(-1, 1)
        u_hat_t = deriv_1[0][:, 1].reshape(-1, 1)
        deriv_2 = autograd.grad(u_hat_x.sum(), input, create_graph=True)
        u_hat_x_x = deriv_2[0][:, 0].reshape(-1, 1)

        # to modify governing equation, modify here
        f = u_hat_t + u_hat * u_hat_x - (0.005/np.pi) * u_hat_x_x
        # f = u_hat_t - u_hat_x_x
        func = nn.MSELoss()
        return func(f, target)

    def calc_loss_by_tag(self, input, target, tag):
        loss_i = 0
        loss_b = 0
        loss_f = 0

        loss_func = nn.MSELoss()
        for inp, tar, t in zip(input, target, tag):
            if t == 1:
                loss_f += self.calc_loss_f(inp, tar)
            elif t == 0:
                loss_b += loss_func(self(inp), tar)
            else:
                loss_i += loss_func(self(inp), tar)

        if isinstance(loss_f, int):
            loss_f = torch.zeros((1)).requires_grad_()
        if isinstance(loss_b, int):
            loss_b = torch.zeros((1)).requires_grad_()
        if isinstance(loss_i, int):
            loss_i = torch.zeros((1)).requires_grad_()
        
        return loss_i, loss_b, loss_f
    
if __name__ == "__main__":
    a = PINN(5, 5)
    