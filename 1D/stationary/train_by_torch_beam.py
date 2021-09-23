from numpy.random import normal
import torch
from torch.functional import norm 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

import time


from copy import copy

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # self.hidden_layer1    = nn.Linear(1, 50)
        # # self.dropout_layer    = nn.Dropout(0.01)
        # # self.hidden_layer2    = nn.Linear(50, 50)
        # # self.hidden_layer3    = nn.Linear(50, 50)
        # # self.hidden_layer4    = nn.Linear(50, 50)
        # self.output_layer     = nn.Linear(50, 1)
        self.hidden_layer1      = nn.Linear(1, 40)
        self.hidden_layer2      = nn.Linear(40, 40)
        self.hidden_layer3      = nn.Linear(40, 40)
        self.output_layer       = nn.Linear(40, 1)

    def forward(self, x):
        input_data     = x
        act_func       = nn.Sigmoid()
        a_layer1       = act_func(self.hidden_layer1(input_data))
        a_layer2       = act_func(self.hidden_layer2(a_layer1))
        a_layer3       = act_func(self.hidden_layer3(a_layer2))
        # a_layer4       = act_func(self.hidden_layer4(a_layer3))
        out            = self.output_layer(a_layer3)

        return out

# def u(x, t):
#     value = 100 * np.exp(-np.pi * np.pi * t) * np.sin(np.pi * x)
#     return value

def u(x):
    l = 1
    w = 1
    # value = x * x * (x * x - 4 * x + 6) / 24
    value = x * (x - l) * (x * x - l * x - l * l) / 24 * w
    # value = 0.5 * x * (x - 1)
    return value

# def make_training_initial_data(i_size, seed=1004):
#     np.random.seed(seed)

#     x_i = np.random.uniform(low=0.0, high=1.0, size=(i_size, 1))
#     t_i = np.zeros((i_size, 1))
#     u_i = 0.25 * x_i * (1 - x_i)
#     # u_i = 2 * np.sin(3 * np.pi * x_i)

#     return x_i, t_i, u_i

def make_training_boundary_data(b_size, x=0.0, u=0.0, seed=1004):
    np.random.seed(seed)

    x_b = np.ones((b_size, 1)) * x
    u_b = np.ones((b_size, 1)) * u

    return x_b, u_b

# def make_training_boundary_data(b_size, seed=1004, zero=True):
#     np.random.seed(seed)
#     if zero:
#         x_b = np.zeros((b_size, 1))
#     else:
#         x_b = np.ones((b_size, 1))
#     t_b = np.random.uniform(low=0.0, high=1.0, size=(b_size, 1))
#     u_b = np.zeros((b_size, 1))

#     return x_b, t_b, u_b

def make_training_collocation_data(f_size, x_lb=0.0, x_hb=1.0, seed=1004):
    np.random.seed(seed)

    x_f = np.random.uniform(low=x_lb, high=x_hb, size=(f_size, 1))
    u_f = np.zeros((f_size, 1))

    return x_f, u_f

# def make_training_collocation_data(f_size, seed=1004):
#     np.random.seed(seed)

#     x_f = np.random.uniform(low=0.0, high=1.0, size=(f_size, 1))
#     t_f = np.random.uniform(low=0.0, high=1.0, size=(f_size, 1))
#     u_f = np.zeros((f_size, 1))

#     return x_f, t_f, u_f

def make_test_data(t_size, seed=1004):
    np.random.seed(seed)

    x_t = np.random.uniform(low=0.0, high=1.0, size=(t_size, 1))
    u_t = np.array([u(x) for x in x_t])

    return x_t, u_t
    
# def make_test_data(t_size, seed=1004):
#     np.random.seed(seed)

#     x_t = np.random.uniform(low=0.0, high=1.0, size=(t_size, 1))
#     t_t = np.random.uniform(low=0.0, high=1.0, size=(t_size, 1))
#     u_t = np.array([u(x, t) for x, t in zip(x_t, t_t)])

#     return x_t, t_t, u_t

def calc_loss_f(x, y, model, func):
    u_hat = model(x)


    u_hat_x     = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
    # u_hat_t     = autograd.grad(u_hat.sum(), t, create_graph=True)[0]
    u_hat_x_x   = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]
    u_hat_x_x_x = autograd.grad(u_hat_x_x.sum(), x, create_graph=True)[0]
    u_hat_x_x_x_x = autograd.grad(u_hat_x_x_x.sum(), x, create_graph=True)[0]
    # u_hat_t_t   = autograd.grad(u_hat_t.sum(), t, create_graph=True)[0]

    f = u_hat_x_x_x_x - 1
    # f = u_hat_t_t - 4 * u_hat_x_x

    return func(f, y) 

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

def train(epochs=100000):
    since = time.time()
    b_size = 5000
    f_size = 5000
    t_size = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Current device:", device)
    
    model = PINN()
    model.to(device)
    model = nn.DataParallel(model)
    
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    x_b, u_b = make_training_boundary_data(b_size, x=0.0, u=0.0)
    x_b_2, u_b_2 = make_training_boundary_data(b_size, x=0.5, u=0.0)
    x_b_3, u_b_3 = make_training_boundary_data(b_size, x=1.0, u=0.0)
    x_f, u_f = make_training_collocation_data(f_size)
    # x_t, u_t = make_test_data(t_size)

    # x_b = np.concatenate((x_b, x_b_2))
    # u_b = np.concatenate((u_b, u_b_2))

    # scaler_x = MinMaxScaler()
    # scaler_u = MinMaxScaler()

    # scaler_x.fit(x_f)
    # scaler_u.fit(u_b)

    # x_i = scaler_x.transform(x_i)
    # x_b = scaler_x.transform(x_b)
    # x_f = scaler_x.transform(x_f)
    # x_t = scaler_x.transform(x_t)
    # t_i = scaler_t.transform(t_i)
    # t_b = scaler_t.transform(t_b)
    # t_f = scaler_t.transform(t_f)
    # t_t = scaler_t.transform(t_t)
    # u_i = scaler_u.transform(u_i)
    # u_b = scaler_u.transform(u_b)
    # u_f = scaler_u.transform(u_f)
    # u_t = scaler_u.transform(u_t)

    # print(u_b)
    x_b = make_tensor(x_b).to(device)
    u_b = make_tensor(u_b, requires_grad=False).to(device)
    x_f = make_tensor(x_f).to(device)
    u_f = make_tensor(u_f, requires_grad=False).to(device)
    # x_t = make_tensor(x_t, requires_grad=False).to(device)
    # u_t = make_tensor(u_t, requires_grad=False).to(device)

    x_b_2 = make_tensor(x_b_2).to(device)
    u_b_2 = make_tensor(u_b_2, requires_grad=True).to(device)
    x_b_3 = make_tensor(x_b_3).to(device)
    u_b_3 = make_tensor(u_b_3, requires_grad=True).to(device)
    



    # x = torch.cat([x_i, x_b, x_f, x_t])
    # t = torch.cat([t_i, t_b, t_f, t_t])
    # u = torch.cat([u_i, u_b, u_f, u_t])

    # max_x = x.max()
    # min_x = x.min()
    # max_t = t.max()
    # min_t = t.min()
    # max_u = u.max()
    # min_u = u.min()

    # u_

    # u_i_2 = make_tensor(u_i_2).to(device)


    loss_save = np.inf

    for epoch in range(epochs):
        optim.zero_grad()
        loss_b = 0.0

        # pred_i, max_i, min_i = normalize(pred_i)
        # pred_b, max_b, min_b = normalize(pred_b)
        # pred_t, max_t, min_t = normalize(pred_t)
  
        # deriv_i  = autograd.grad(pred_i.sum(), t_i, create_graph=True)[0]

        loss_func = nn.MSELoss()

        loss_b += loss_func(model(x_b), u_b)
        # loss_b += loss_func(model(x_b_2), u_b_2)
        loss_b += loss_func(model(x_b_3), u_b_3)
        loss_b += loss_func(calc_deriv(x_b, model(x_b), 2), u_b)
        # loss_b += loss_func(calc_deriv(x_b_2, model(x_b_2), 2), u_b_2)
        loss_b += loss_func(calc_deriv(x_b_3, model(x_b_3), 2), u_b_3)

        # loss_b *= 1000

        loss_f = calc_loss_f(x_f, u_f, model, loss_func)


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
                torch.save(model.state_dict(), './models/model_single.data')
                print(".......model updated (epoch = ", epoch+1, ")")
                
            # print("Epoch: {} | LOSS_TOTAL: {:.8f} | LOSS_TEST: {:.4f}".format(epoch + 1, loss, loss))
            print("Epoch: {0} | LOSS_B: {1:.8f} | LOSS_F: {2:.8f} | LOSS_TOTAL: {3:.8f} | LOSS_TEST: {4:.8f}".format(epoch + 1,  loss_b, loss_f, loss, loss))

            if loss < 0.00001:
                
                break
            # print("Epoch: {0} | LOSS: {1:.4f} | LOSS_F: {2:.8f} | LOSS_TEST: {3:.4f}".format(epoch + 1, loss_i + loss_b, loss_f, loss_test))
        
    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)
    print("Elapsed time: {:.3f} s".format(time.time() - since))
    print("Done!") 

        



def main():
    train()

if __name__ == '__main__':
    main()


    