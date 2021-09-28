import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt

from copy import copy

pi = torch.acos(torch.zeros(1)).item() * 2

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        self.hidden_layer1    = nn.Linear(2, 30)
        self.hidden_layer2    = nn.Linear(30, 30)
        self.hidden_layer3    = nn.Linear(30, 30)
        self.hidden_layer4    = nn.Linear(30, 30)
        self.hidden_layer5    = nn.Linear(30, 30)
        self.hidden_layer6    = nn.Linear(30, 30)
        self.hidden_layer7    = nn.Linear(30, 30)
        self.hidden_layer8    = nn.Linear(30, 30)
        self.output_layer     = nn.Linear(30, 1)
    
    def forward(self, x, t):
        input_data      = torch.cat([x, t], axis=1)
        act_func        = nn.Sigmoid()
        a_layer1        = act_func(self.hidden_layer1(input_data))
        a_layer2        = act_func(self.hidden_layer2(a_layer1))
        a_layer3        = act_func(self.hidden_layer3(a_layer2))
        a_layer4        = act_func(self.hidden_layer4(a_layer3))
        a_layer5        = act_func(self.hidden_layer2(a_layer4))
        a_layer6        = act_func(self.hidden_layer3(a_layer5))
        a_layer7        = act_func(self.hidden_layer4(a_layer6))
        a_layer8        = act_func(self.hidden_layer4(a_layer7))
        out             = self.output_layer(a_layer8)

        return out


def make_training_boundary_data_x(b_size, x_lb=0.0, x_hb=1.0, t=0.0, seed=1004):
    np.random.seed(seed)

    x_b = np.random.uniform(low=x_lb, high=x_hb, size=(b_size, 1))
    t_b = np.ones((b_size, 1)) * t
    u_b = -1 * np.sin(np.pi * x_b)

    return x_b, t_b, u_b  

def make_training_boundary_data_t(b_size, t_lb=0.0, t_hb=1.0, x=0.0, u=0.0, seed=1004):
    np.random.seed(seed)

    t_b = np.random.uniform(low=t_lb, high=t_hb, size=(b_size, 1))
    x_b = np.ones((b_size, 1)) * x
    u_b = np.ones((b_size, 1)) * u

    return x_b, t_b, u_b  

def make_training_collocation_data(f_size, x_lb=0.0, x_hb=1.0, t_lb=0.0, t_hb=1.0, seed=1004):
    np.random.seed(seed)

    x_f = np.random.uniform(low=x_lb, high=x_hb, size=(f_size, 1))
    t_f = np.random.uniform(low=t_lb, high=t_hb, size=(f_size, 1))
    u_f = np.zeros((f_size, 1))

    return x_f, t_f, u_f

def calc_loss_f(x, t, u, model, func):
    u_hat = model(x, t)

    u_hat_x = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
    u_hat_t = autograd.grad(u_hat.sum(), t, create_graph=True)[0]
    u_hat_x_x = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]

    

    f = u_hat_t + u_hat * u_hat_x - 0.01 / pi * u_hat_x_x

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

def train(epochs=50000):
    b_size = 100
    f_size = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    model = PINN()
    model.to(device)
    model = nn.DataParallel(model)

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    x_b, t_b, u_b = make_training_boundary_data_x(b_size // 2, x_lb=-1.0, x_hb=1.0, t=0.0)
    x_b_2, t_b_2, u_b_2 = make_training_boundary_data_t(b_size // 4, t_lb=0.0, t_hb=1.0, x=-1.0, u=0.0)
    x_b_3, t_b_3, u_b_3 = make_training_boundary_data_t(b_size // 4, t_lb=0.0, t_hb=1.0, x=1.0, u=0.0)

    x_f, t_f, u_f = make_training_collocation_data(f_size, x_lb=-1.0, x_hb=1.0, t_lb=0.0, t_hb=1.0)

    x_b_plt = np.concatenate((x_b, x_b_2, x_b_3))
    t_b_plt = np.concatenate((t_b, t_b_2, t_b_3))
    u_b_plt = np.concatenate((u_b, u_b_2, u_b_3))


    plt.figure(figsize=(6, 6))
    plt.scatter(x_b_plt, t_b_plt, marker='x', label='Boundary conditions')
    plt.scatter(x_f, t_f, marker='x', label='Computational domain')
    plt.xlabel('X')
    plt.ylabel('T')
    plt.legend(loc='lower right')
    plt.colorbar()
    plt.savefig('./figures/sampled_data.png')

    plt.cla()
    plt.scatter(x_b, u_b)
    plt.savefig('./figures/initial_condition.png')

    x_b = make_tensor(x_b_plt).to(device)
    t_b = make_tensor(t_b_plt).to(device)
    u_b = make_tensor(u_b_plt).to(device)

    x_f = make_tensor(x_f).to(device)
    t_f = make_tensor(t_f).to(device)
    u_f = make_tensor(u_f).to(device)

    loss_save = np.inf

    for epoch in range(epochs):
        optim.zero_grad()
        loss_func = nn.MSELoss()
        loss_b = loss_func(model(x_b, t_b), u_b)
        loss_f = calc_loss_f(x_f, t_f, u_f, model, loss_func)

        loss = loss_b + loss_f

        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(model.state_dict(), './models/model.data')
                print(".......model updated (epoch = ", epoch+1, ")")
            print("Epoch: {0} | LOSS_B: {1:.8f} | LOSS_F: {2:.8f} | LOSS_TOTAL: {3:.8f} ".format(epoch + 1,  loss_b, loss_f, loss))

            if loss < 0.000001:
                break

    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)
    print("Done!") 

def draw(time):
    model = PINN()

    model_ckpt = "./models/model.data"
    state_dict = torch.load(model_ckpt)
    state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    model.load_state_dict(state_dict)
    x, t = np.mgrid[-1.00:1.01:0.01, 0.00:1.01:0.01]
    xt = torch.from_numpy(np.vstack((x.flatten(), t.flatten()))).type(torch.FloatTensor)

    pred = model(xt[0].unsqueeze(0).T, xt[1].unsqueeze(0).T)
    pred = pred.detach().numpy()

    plt.cla()
    plt.scatter(t, x, c=pred, cmap='Spectral')
    plt.colorbar()
    plt.title("Burgers")
    plt.xlabel("X")
    plt.ylabel("T")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_burgers.png')

    x = xt[0].unsqueeze(0).T
    t = torch.ones(x.shape) * 0.25
    u = model(x, t).detach().numpy()

    plt.cla()
    plt.scatter(x, u, cmap='Spectral')
    plt.title("Burgers")
    plt.xlabel("X")
    plt.ylabel("U")
    plt.xticks()
    plt.yticks()
    plt.savefig('./figures/fig_burgers_{}s.png'.format(time))



def main():
    train()
    draw(0.75)

if __name__ == '__main__':
    main()