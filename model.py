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

    # to modify governing equation, modify here
    
    
if __name__ == "__main__":
    a = PINN(5, 5)
    