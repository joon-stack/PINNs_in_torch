import torch
import torch.autograd as autograd
import torch.nn as nn

import matplotlib.pyplot as plt

import time
import os

from train_by_torch_beam import *

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
    
        self.nn_1, self.nn_2 = models
    
    def forward(self, x):
        out = self.nn_1(x) + torch.mul(self.nn_2(x), distance(x))
        return out

def distance(x):

    dist = torch.cat((torch.abs(x - 0), torch.abs(x - 0.5), torch.abs(x - 1.0)), 1)
    # dist = torch.cat((torch.abs(x - 0), torch.abs(x - 1.0)), 1)
    dist = 1 / (1 + torch.exp(torch.min(dist, 1).values))

    if dist.ndim < 2:
        dist = torch.unsqueeze(dist, 1)
    

    return dist

def calc_deriv(x, input, times):
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res


def forced_train():
    # save start time
    since = time.time()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    # initialize PINNs structures
    nn_1 = PINN()
    nn_2 = PINN()

    # set PINNs on the device
    nn_1.to(device)
    nn_2.to(device)

    optim_1 = torch.optim.Adam(nn_1.parameters(), lr=0.001)
    optim_2 = torch.optim.Adam(nn_2.parameters(), lr=0.001)

    # set hyperparameters
    b_size = 500
    f_size = 5000
    epochs = 10000

    # generate training data
    x_b, u_b = make_training_boundary_data(b_size, x=0.0, u=0.0)
    x_b_2, u_b_2 = make_training_boundary_data(b_size, x=0.5, u=0.0)
    x_b_3, u_b_3 = make_training_boundary_data(b_size, x=1.0, u=0.0)

    x_f, u_f = make_training_collocation_data(f_size, x_lb=0.0, x_hb=1.0)

    plt.scatter(x_b, u_b, marker='x')
    plt.scatter(x_b_2, u_b_2, marker='x')
    plt.scatter(x_b_3, u_b_3, marker='x')
    plt.scatter(x_f, u_f, marker='.')

    plt.savefig('./figures/data.png')
    
    x_b = make_tensor(x_b).to(device)
    u_b = make_tensor(u_b, requires_grad=False).to(device)
    x_f = make_tensor(x_f).to(device)
    u_f = make_tensor(u_f, requires_grad=False).to(device)
    x_b_2 = make_tensor(x_b_2).to(device)
    u_b_2 = make_tensor(u_b_2, requires_grad=True).to(device)
    x_b_3 = make_tensor(x_b_3).to(device)
    u_b_3 = make_tensor(u_b_3, requires_grad=True).to(device)

    # initialize save loss
    loss_save = np.inf

    for epoch in range(epochs):
        optim_1.zero_grad()
        loss = 0.0
        loss_func = nn.MSELoss()

        loss += loss_func(nn_1(x_b), u_b)
        loss += loss_func(nn_1(x_b_2), u_b_2)
        loss += loss_func(nn_1(x_b_3), u_b_3)

        loss += loss_func(calc_deriv(x_b, nn_1(x_b), 2), u_b)
        loss += loss_func(calc_deriv(x_b_3, nn_1(x_b_3), 3), u_b_3)

        loss.backward()
        optim_1.step()

        with torch.no_grad():
            nn_1.eval()

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(nn_1.state_dict(), './models/nn_1.data')
                print(".......model updated (epoch = ", epoch+1, ")")
        print("Epoch: {0} | LOSS: {1:.8f} ".format(epoch + 1,  loss))


    # loss_save = np.inf
    # for epoch in range(epochs):
    #     optim_2.zero_grad()
    #     loss = 0.0

    #     loss += calc_loss_f(x_f, u_f, nn_2, loss_func)

    #     loss.backward()
    #     optim_2.step()

    #     with torch.no_grad():
    #         nn_2.eval()

    #         if loss < loss_save:
    #             best_epoch = epoch
    #             loss_save = copy(loss)
    #             torch.save(nn_2.state_dict(), './models/nn_2.data')
    #             print(".......model updated (epoch = ", epoch+1, ")")
    #     # print("Epoch: {} | LOSS_TOTAL: {:.8f} | LOSS_TEST: {:.4f}".format(epoch + 1, loss, loss))
    #     print("Epoch: {0} | LOSS: {1:.8f} ".format(epoch + 1,  loss))
    #         # print("Epoch: {0} | LOSS: {1:.4f} | LOSS_F: {2:.8f} | LOSS_TEST: {3:.4f}".format(epoch + 1, loss_i + loss_b, loss_f, loss_test))
    

    # load pretrained weights of PINNs
    nn_1_path = "./models/nn_1.data"
    # nn_2_path = "./models/nn_2.data"

    if os.path.isfile(nn_1_path):
        state_dict = torch.load(nn_1_path)
        state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
        nn_1.load_state_dict(state_dict)
    
    # if os.path.isfile(nn_2_path):
    #     state_dict = torch.load(nn_2_path)
    #     state_dict = {m.replace('module.', '') : i for m, i in state_dict.items()}
    #     nn_2.load_state_dict(state_dict)

    # initialize ensemble model
    model = EnsembleModel((nn_1, nn_2))

    for child in model.nn_1.children():
        for param in child.parameters():
            param.requires_grad = False

    # initialize Adam optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_save = np.inf
    # train
    for epoch in range(epochs):
        optim.zero_grad()
        loss = 0.0
        loss_func = nn.MSELoss()

        # loss += loss_func(model(x_b), u_b)
        # loss += loss_func(model(x_b_2), u_b_2)
        # loss += loss_func(model(x_b_3), u_b_3)
        # loss += loss_func(calc_deriv(x_b, model(x_b), 2), u_b)
        # loss += loss_func(calc_deriv(x_b_3, model(x_b_3), 3), u_b_3)
        loss += calc_loss_f(x_f, u_f, model, loss_func)

        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(model.state_dict(), './models/model.data')
                print(".......model updated (epoch = ", epoch+1, ")")

            print("Epoch: {0} | LOSS: {1:.8f} ".format(epoch + 1,  loss))


        



    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)
    print("Elapsed time: {:.3f} s".format(time.time() - since))
    print("Done!") 
        
def combine():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    nn_1 = PINN()
    nn_2 = PINN()

    nn_1_path = "./models/nn_1.data"
    state_dict = torch.load(nn_1_path)
    nn_1.load_state_dict(state_dict)

    nn_2_path = "./models/nn_2.data"
    state_dict = torch.load(nn_2_path)
    nn_2.load_state_dict(state_dict)

    nn_1.to(device)
    nn_2.to(device)

    model = EnsembleModel((nn_1, nn_2))

    model.to(device)

    x_test = torch.from_numpy(np.arange(10001)/10000).type(torch.FloatTensor).unsqueeze(0).T



    pred = nn_1(x_test) + torch.mul(nn_2(x_test), distance(x_test))
    # pred = distance(x_test)

    pred = pred.detach().numpy()

    plt.scatter(x_test, pred)
    plt.colorbar()

    plt.savefig('./figures/forced_1.png')


if __name__ == "__main__":
    forced_train()
    # combine()
