import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time

from torch.utils.data import DataLoader, Dataset

from model import *
from generate_data import *

from copy import copy

class CustomDataset(Dataset):
    def __init__(self, input, u, tag):
        self.input = input
        self.u = u
        self.tag = tag
    
    def __len__(self):
        return self.input.shape[0]
    
    def __getitem__(self, idx):
        input = self.input[idx]
        u = self.u[idx]
        tag = self.tag[idx]
        return input, u, tag


def train(epochs=1000):
    i_size = 5
    b_size = 5
    f_size = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    model = PINN(5, 5)
    model.to(device)
    
    optim = torch.optim.Adam(model.parameters())

    i_set = make_training_initial_data(i_size)
    b_set = make_training_boundary_data(b_size)
    f_set = make_training_domain_data(f_size)

    inputs      = torch.cat((i_set[0], b_set[0], f_set[0]), axis=0).to(device)
    us          = torch.cat((i_set[1], b_set[1], f_set[1]), axis=0).to(device)
    tags        = torch.cat((i_set[2], b_set[2], f_set[2]), axis=0).to(device)

    loss_save = np.inf
    
    loss_func = nn.MSELoss()

    dataset_train = CustomDataset(inputs, us, tags)
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
    for epoch in range(epochs):
        model.train()
        train_loss = []

        for batch, data in enumerate(loader_train, 1):
            input, u, tag = data
            output = model(input)

            optim.zero_grad()

            loss = model.calc_loss_by_tag(input, u, tag).to(device)
            loss.backward()

            optim.step()

            train_loss += [loss.item()]

        with torch.no_grad():
            model.eval()
            
    

        
        



    for epoch in range(epochs):
        pass


if __name__ == "__main__":
    train()