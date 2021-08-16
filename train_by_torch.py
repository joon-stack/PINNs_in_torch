import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import warnings


warnings.filterwarnings('ignore')

class PhysicsInformedNeuralNetwork(nn.Module):
    def __init__(self, lr):
        super().__init__()

        self.loss_function = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        print("Current device:", self.device)
        
    def u(self, x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi * np.pi * t)

    def make_training_initial_data(self, x_i_size, seed=2021):
        torch.manual_seed(seed)

        x_i = [torch.tensor((0.0, float(torch.rand(1))), requires_grad=True) for _ in range(x_i_size)]
        u_i = [torch.tensor((0.0), requires_grad=True) for _ in range(x_i_size)]

        x = torch.stack(x_i)
        u = torch.stack(u_i)

        return x, u
    
    def make_training_boundary_data(self, x_b_size, seed=2021):
        torch.manual_seed(seed)
        x_x = [torch.rand(1) for _ in range(x_b_size)]
        x_b = [torch.tensor((x, 0.0), requires_grad=True) for x in x_x]
        u_b = [torch.tensor((np.sin(np.pi * x)), requires_grad=True) for x in x_x]

        x = torch.stack(x_b)
        u = torch.stack(u_b)

        return x, u
    
    def make_training_functional_data(self, x_f_size, seed=2021):
        torch.manual_seed(seed)
        x_f = [torch.tensor((torch.rand(1), torch.rand(1)), requires_grad=True) for _ in range(x_f_size)]
        x = torch.stack(x_f)

        return x

    def make_test_data(self, x_test_size, seed=2021):
        torch.manual_seed(seed)
        
        x_test = [torch.tensor((torch.rand(1), torch.rand(1))) for _ in range(x_test_size)]
        u_test = [torch.tensor(self.u(x[0], x[1])) for x in x_test]

        x = torch.stack(x_test)
        u = torch.stack(u_test)

        return x, u
    

    def calc_loss(self, dataloader, loss_func, model):
        for batch, data in enumerate(dataloader, 1):
            loss = 0.0
            input = data[0].to(self.device)
            y = data[1].to(self.device)
            output = model(input)
            loss += loss_func(output, y)
        
        return loss
    
    def calc_loss_f(self, dataloader, model):
        for batch, data in enumerate(dataloader, 1):
            input = data[0].to(self.device)
            label = data[1].to(self.device)
            x = input.clone()
            loss = self.loss_f(x, label, model)
        
        return loss

    def loss_i(self, output, y):
        loss = 0.0
        cnt = 0
        for pred, label in zip(output, y):
            loss += self.loss_function(pred, label)
            cnt += 1
        return (loss / cnt)
    
    def loss_b(self, output, y):
        loss = 0.0
        cnt = 0
        for pred, label in zip(output, y):
            loss += self.loss_function(pred, label)
            cnt += 1
        return (loss / cnt)
    
    def loss_f(self, input, y, model):
        loss = 0.0

        u_hat = model(input)
        input.retain_grad()
        s = u_hat.sum()
        # s.backward()
        # print(input.grad)

        u_hat_first     = torch.autograd.grad(s, input, create_graph=True, allow_unused=True)[0]
        u_hat_second    = torch.autograd.grad(u_hat_first.sum(), input, create_graph=True, allow_unused=True)[0]

        u_hat_x     = u_hat_first[:, [0]]
        u_hat_t     = u_hat_first[:, [1]]
        u_hat_xx    = u_hat_second[:, [0]]

        # print(u_hat_x)
        # print(u_hat_t)
        # print(u_hat_xx)

        
        loss = self.loss_function(u_hat_xx, u_hat_t)

        return loss
    
    def train(self, epochs=100, lr=0.01):
        
        model = nn.Sequential(nn.Linear(2, 10),
                                    nn.Sigmoid(),
                                    nn.Linear(10, 10),
                                    nn.Sigmoid(),
                                    nn.Linear(10, 10),
                                    nn.Sigmoid(),
                                    nn.Linear(10, 1),
                                    nn.Sigmoid())

        optim = torch.optim.Adam(model.parameters(), lr=lr)

        model.to(self.device)

        model = nn.DataParallel(model)
        

        in_train_i, u_train_i    = self.make_training_initial_data(500)
        in_train_b, u_train_b    = self.make_training_boundary_data(500)
        in_train_f               = self.make_training_functional_data(100000)
        in_test, u_test          = self.make_test_data(1000)
        u_train_f                = torch.zeros(in_train_f.shape)

        

        if u_train_i.ndim < 2:
            u_train_i = u_train_i.unsqueeze(1)
        
        in_train    = torch.cat([in_train_i, in_train_b])
        u_train     = torch.cat([u_train_i, u_train_b])


        training_data_i     = TensorDataset(in_train_i, u_train_i)
        training_data_b     = TensorDataset(in_train_b, u_train_b)
        training_data_f     = TensorDataset(in_train_f, u_train_f)
        test_data           = TensorDataset(in_test, u_test)

        loader_train_i      = DataLoader(training_data_i, batch_size=len(in_train_i))
        loader_train_b      = DataLoader(training_data_b, batch_size=len(in_train_b))
        loader_train_f      = DataLoader(training_data_f, batch_size=len(in_train_f))
        loader_test         = DataLoader(test_data, batch_size=len(in_test))

        # for batch, data in enumerate(loader_train_i, 1):
        #     input = data[0]
        #     print(input)
        #     input.retain_grad()
        #     output = model(input)
        #     print(input.grad)
        #     print(output)

        #     ot = output.sum()
        #     ot.retain_grad()
        #     ot.backward()
        #     print(input.grad)
            

        for n, epoch in enumerate(range(epochs)):
            
            loss_i = self.calc_loss(loader_train_i, self.loss_i, model)
            loss_b = self.calc_loss(loader_train_b, self.loss_b, model)
            loss_f = self.calc_loss_f(loader_train_f, model)

            loss = loss_i + loss_b + loss_f
            loss.backward(retain_graph=True)

            optim.step()

            with torch.no_grad():
                model.eval()
                
                loss_test = self.calc_loss(loader_test, self.loss_i, model)

                print("Epoch: {0} | LOSS_I: {1:.4f} | LOSS_B: {2:.4f} | LOSS_F: {3:.8f} | LOSS_TEST: {4:.4f}".format(epoch + 1, loss_i, loss_b, loss_f, loss_test))

        print("Done!") 
        

          
        
        
u_hat = PhysicsInformedNeuralNetwork(lr=0.2)
u_hat.train()

