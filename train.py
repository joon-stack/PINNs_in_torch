import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time

from torch.utils.data import DataLoader, Dataset


from model import *
from generate_data import *
from utils import *

from copy import copy

class CustomDataset(Dataset):
    def __init__(self, input, target, tag):
        self.input = input
        self.target = target
        self.tag = tag
    
    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        input = self.input[idx]
        target = self.target[idx]
        tag = self.tag[idx]
        return input, target, tag

def calc_loss_f(model, input, target):
    u_hat = model(input)
    
    deriv_1 = autograd.grad(u_hat.sum(), input, create_graph=True)
    u_hat_x, u_hat_t = deriv_1[0]
    deriv_2 = autograd.grad(u_hat_x.sum(), input, create_graph=True)
    u_hat_x_x, _ = deriv_2[0]

    # f = u_hat_t + u_hat * u_hat_x - (0.01/np.pi) * u_hat_x_x
    f = u_hat_t - u_hat_x_x

    func = nn.MSELoss()
    return func(f, target)

def calc_loss_by_tag(model, input, target, tag):
    loss_i = 0
    loss_b = 0
    loss_f = 0

    loss_func = nn.MSELoss()
    for inp, tar, t in zip(input, target, tag):
        if t == 1:
            # print(calc_loss_f(model, inp, tar))
            loss_f += calc_loss_f(model, inp, tar)
        elif t == 0:
            loss_b += loss_func(model(inp), tar)
        else:
            loss_i += loss_func(model(inp), tar)

    if isinstance(loss_f, int):
        loss_f = torch.zeros((1)).requires_grad_()
    if isinstance(loss_b, int):
        loss_b = torch.zeros((1)).requires_grad_()
    if isinstance(loss_i, int):
        loss_i = torch.zeros((1)).requires_grad_()
    
    return loss_i, loss_b, loss_f

def train(epochs=1000):
    i_size = 500
    b_size = 500
    f_size = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    model = PINN(5, 5)
    model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=0.1)

    i_set = make_training_initial_data(i_size)
    b_set = make_training_boundary_data(b_size)
    f_set = make_training_domain_data(f_size)

    inputs      = torch.cat((i_set[0], b_set[0], f_set[0]), axis=0).to(device)
    targets     = torch.cat((i_set[1], b_set[1], f_set[1]), axis=0).to(device)
    tags        = torch.cat((i_set[2], b_set[2], f_set[2]), axis=0).to(device)

    loss_save = np.inf

    dataset_train = CustomDataset(inputs, targets, tags)
    loader_train = DataLoader(dataset_train, batch_size=i_size + b_size + f_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        train_loss_i = []
        train_loss_b = []
        train_loss_f = []
        train_loss = []

        for batch, data in enumerate(loader_train, 1):
            input, target, tag = data

            optim.zero_grad()

            loss_i, loss_b, loss_f = calc_loss_by_tag(model, input, target, tag)
            
            loss_i = loss_i.to(device)
            loss_b = loss_b.to(device)
            loss_f = loss_f.to(device)

            loss = loss_i + loss_b + loss_f
            loss.backward()

            optim.step()

            train_loss_i += [loss_i.item()]
            train_loss_b += [loss_b.item()]
            train_loss_f += [loss_f.item()]
            train_loss   += [loss.item()]
        
        with torch.no_grad():
            model.eval()
            print("Epoch {0} | Loss_I: {1:.4f} | Loss_B: {2:.4f} | Loss_F: {3:.4f}".format(epoch, np.mean(train_loss_i), np.mean(train_loss_b), np.mean(train_loss_f)))
            


if __name__ == "__main__":
    train()
    # a = torch.tensor([[1, 2, 3], [4, 5, 6]]).float().requires_grad_(True)
    # b = torch.tensor([[2], [4]]).float().requires_grad_(True)
    # c = a * b
    # d = autograd.grad(c.sum(), a)
    # print(d)