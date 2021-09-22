import torch
import torch.autograd as autograd
import torch.nn as nn

import matplotlib.pyplot as plt

import time

from train_by_torch_beam import *

def distance(x):

    dist = torch.cat((torch.abs(x - 0), torch.abs(x - 0.5), torch.abs(x - 1.0)), 1)
    dist = torch.min(dist, 1).values
    

    return dist

def calc_deriv(x, input, times):
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res

def forced_train():
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    nn_1 = PINN()
    nn_2 = PINN()

    nn_1.to(device)
    nn_2.to(device)

    optim_1 = torch.optim.Adam(nn_1.parameters(), lr=0.01)
    optim_2 = torch.optim.Adam(nn_2.parameters(), lr=0.01)

    b_size = 500
    f_size = 500
    epochs = 10000

    x_b, u_b = make_training_boundary_data(b_size, x=0.0, u=0.0)
    x_b_2, u_b_2 = make_training_boundary_data(b_size, x=0.5, u=0.0)
    x_b_3, u_b_3 = make_training_boundary_data(b_size, x=1.0, u=0.0)

    x_f, u_f = make_training_collocation_data(f_size, x_lb=0.0, x_hb=1.0)
    
    x_b = make_tensor(x_b).to(device)
    u_b = make_tensor(u_b, requires_grad=False).to(device)
    x_f = make_tensor(x_f).to(device)
    u_f = make_tensor(u_f, requires_grad=False).to(device)
    x_b_2 = make_tensor(x_b_2).to(device)
    u_b_2 = make_tensor(u_b_2, requires_grad=True).to(device)
    x_b_3 = make_tensor(x_b_3).to(device)
    u_b_3 = make_tensor(u_b_3, requires_grad=True).to(device)

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
        # print("Epoch: {} | LOSS_TOTAL: {:.8f} | LOSS_TEST: {:.4f}".format(epoch + 1, loss, loss))
        print("Epoch: {0} | LOSS: {1:.8f} ".format(epoch + 1,  loss))
            # print("Epoch: {0} | LOSS: {1:.4f} | LOSS_F: {2:.8f} | LOSS_TEST: {3:.4f}".format(epoch + 1, loss_i + loss_b, loss_f, loss_test))
    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)

    loss_save = np.inf
    for epoch in range(epochs):
        optim_2.zero_grad()
        loss = 0.0

        loss += calc_loss_f(x_f, u_f, nn_2, loss_func)

        loss.backward()
        optim_2.step()

        with torch.no_grad():
            nn_2.eval()

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(nn_2.state_dict(), './models/nn_2.data')
                print(".......model updated (epoch = ", epoch+1, ")")
        # print("Epoch: {} | LOSS_TOTAL: {:.8f} | LOSS_TEST: {:.4f}".format(epoch + 1, loss, loss))
        print("Epoch: {0} | LOSS: {1:.8f} ".format(epoch + 1,  loss))
            # print("Epoch: {0} | LOSS: {1:.4f} | LOSS_F: {2:.8f} | LOSS_TEST: {3:.4f}".format(epoch + 1, loss_i + loss_b, loss_f, loss_test))
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

    x_test = torch.from_numpy(np.arange(10001)/10000).type(torch.FloatTensor)

    pred = nn_1(x_test.unsqueeze(0).T) + nn_2(x_test.unsqueeze(0).T) * distance(x_test.unsqueeze(0).T)
    pred = distance(x_test.unsqueeze(0).T)

    pred = pred.detach().numpy()

    plt.scatter(x_test, pred)
    plt.colorbar()
    
    plt.savefig('./figures/forced_1.png')











if __name__ == "__main__":
    # forced_train()
    combine()
