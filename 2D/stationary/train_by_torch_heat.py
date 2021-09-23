from numpy.random import normal
import torch
from torch.functional import norm 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
import matplotlib.pyplot as plt


from copy import copy

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.hidden_layer1    = nn.Linear(2, 10)
        self.hidden_layer2    = nn.Linear(10, 10)
        self.hidden_layer3    = nn.Linear(10, 10)
        self.hidden_layer4    = nn.Linear(10, 10)
        self.hidden_layer5    = nn.Linear(10, 10)
        self.output_layer     = nn.Linear(10, 1)

    def forward(self, x, y):
        input_data     = torch.cat([x, y], axis=1)
        act_func       = nn.Sigmoid()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        a_layer4       = act_func(self.hidden_layer4(a_layer3))
        a_layer5       = act_func(self.hidden_layer5(a_layer4))
        out            = self.output_layer(a_layer5)

        return out


def make_training_boundary_data_x(b_size, x_lb=0.0, x_hb=1.0, y=0.0, u=0.0, seed=1004):
    np.random.seed(seed)

    x_b = np.random.uniform(low=x_lb, high=x_hb, size=(b_size, 1))
    y_b = np.ones((b_size, 1)) * y
    u_b = np.ones((b_size, 1)) * u

    return x_b, y_b, u_b

def make_training_boundary_data_y(b_size, y_lb=0.0, y_hb=1.0, x=0.0, u=0.0, seed=1004):
    np.random.seed(seed)

    y_b = np.random.uniform(low=y_lb, high=y_hb, size=(b_size, 1))
    x_b = np.ones((b_size, 1)) * x
    u_b = np.ones((b_size, 1)) * u


    return x_b, y_b, u_b


def make_training_collocation_data(f_size, x_lb=0.0, x_hb=1.0, y_lb=0.0, y_hb=1.0, seed=1004):
    np.random.seed(seed)

    x_f = np.random.uniform(low=x_lb, high=x_hb, size=(f_size, 1))
    y_f = np.random.uniform(low=y_lb, high=y_hb, size=(f_size, 1))
    u_f = np.zeros((f_size, 1))

    return x_f, y_f, u_f


def make_test_data(t_size, seed=1004):
    np.random.seed(seed)

    x_t = np.random.uniform(low=0.0, high=1.0, size=(t_size, 1))
    u_t = np.array([u(x) for x in x_t])

    return x_t, u_t
    

def calc_loss_f(x, y, u, model, func):
    u_hat = model(x, y)

    u_hat_x = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
    u_hat_y = autograd.grad(u_hat.sum(), y, create_graph=True)[0]
    u_hat_x_x = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]
    u_hat_y_y = autograd.grad(u_hat_y.sum(), y, create_graph=True)[0]


    f = u_hat_x_x + u_hat_y_y
    # f = u_hat_t_t - 4 * u_hat_x_x

    return func(f, u) 

def calc_deriv(x, input, times):
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res

def make_tensor(x, requires_grad=True):
    t = torch.from_numpy(x)
    t = t.float()

    t.requires_grad=requires_grad

    return t

def make_dataset(x, y, u):
    xyu = torch.cat((x, y, u), axis=1)
    return xyu

def train(epochs=0):
    b_size = 50
    f_size = 300
    t_size = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Current device:", device)
    
    model = PINN()
    model.to(device)
    model = nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    x_b, y_b, u_b = make_training_boundary_data_y(b_size, x=0, u=0)
    x_b_2, y_b_2, u_b_2 = make_training_boundary_data_y(b_size, x=1, u=1)
    x_b_3, y_b_3, u_b_3 = make_training_boundary_data_x(b_size, y=0, u=0)
    x_b_4, y_b_4, u_b_4 = make_training_boundary_data_x(b_size, y=1, u=0)

    x_f, y_f, u_f = make_training_collocation_data(f_size)

    x_b_plt = np.concatenate((x_b, x_b_2, x_b_3, x_b_4))
    y_b_plt = np.concatenate((y_b, y_b_2, y_b_3, y_b_4))

    plt.figure(figsize=(6, 6))
    plt.scatter(x_b_plt, y_b_plt, marker='x', label='Boundary conditions')
    plt.scatter(x_f, y_f, marker='x', label='Computational domain')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower right')

    plt.savefig('./figures/data.png')

    x_b = make_tensor(x_b).to(device)
    y_b = make_tensor(y_b).to(device)
    u_b = make_tensor(u_b, requires_grad=False).to(device)

    xyu = torch.cat([x_b, y_b, u_b], axis=1)
    print(xyu.shape)
    
    x_f = make_tensor(x_f).to(device)
    y_f = make_tensor(y_f).to(device)
    u_f = make_tensor(u_f, requires_grad=False).to(device)

    x_b_2 = make_tensor(x_b_2).to(device)
    y_b_2 = make_tensor(y_b_2).to(device)
    u_b_2 = make_tensor(u_b_2, requires_grad=True).to(device)

    x_b_3 = make_tensor(x_b_3).to(device)
    y_b_3 = make_tensor(y_b_3).to(device)
    u_b_3 = make_tensor(u_b_3, requires_grad=True).to(device)

    x_b_4 = make_tensor(x_b_4).to(device)
    y_b_4 = make_tensor(y_b_4).to(device)
    u_b_4 = make_tensor(u_b_4, requires_grad=True).to(device)

    # to do: SGD
    # loader = DataLoader(x_b, batch_size=16, shuffle=True)
    # print(loader)

    loss_save = np.inf

    for epoch in range(epochs):
        optim.zero_grad()
        

        loss_func = nn.MSELoss()

        loss_b = loss_func(model(x_b, y_b), u_b)
        loss_b += loss_func(model(x_b_2, y_b_2), u_b_2)

        loss_b += loss_func(calc_deriv(y_b_3, model(x_b_3, y_b_3), times=1), u_b_3)
        loss_b += loss_func(calc_deriv(y_b_4, model(x_b_4, y_b_4), times=1), u_b_4)


        loss_f = calc_loss_f(x_f, y_f, u_f, model, loss_func)


        # print(deriv_i, u_i_2)       

        # loss_i_2 = loss_func(deriv_i, u_i_2)

        # loss_i += loss_i_2

        loss = loss_b + loss_f
        # loss = loss_i + loss_f

        loss.backward()

        optim.step()

        with torch.no_grad():
            model.eval()
                
            # loss_t = loss_func(pred_t, u_t)

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(model.state_dict(), './models/model.data')
                print(".......model updated (epoch = ", epoch+1, ")")
            print("Epoch: {0} | LOSS_B: {1:.8f} | LOSS_F: {2:.8f} | LOSS_TOTAL: {3:.8f} | LOSS_TEST: {4:.8f}".format(epoch + 1,  loss_b, loss_f, loss, loss))

            if loss < 0.00001:
                break

    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)
    print("Done!") 

        



def main():
    train()

if __name__ == '__main__':
    main()


    