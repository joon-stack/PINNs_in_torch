import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
from copy import copy

from train_by_torch_heat import *

def load_training_data(file):
    with open(file, 'rb') as f:
        data = np.load(f)
    return data


def train(epochs=50000):
    since = time.time()
    size = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Current device:", device)
    
    model = PINN()
    model = model.to(device)

    lambda_1 = torch.tensor([0.0]).to(device).detach().requires_grad_(True)
    lambda_2 = torch.tensor([0.0]).to(device).detach().requires_grad_(True)

    lambda_1 = lambda_1.cuda()
    lambda_2 = lambda_2.cuda()

 
    # model = nn.DataParallel(model)
    
    optim = torch.optim.Adam([{'params': model.parameters()},
                             {'params': [lambda_1, lambda_2], 'lr': 0.01}
                            ],
                            lr=0.01)

    # optim_lambda = torch.optim.Adam([lambda_1], lr=0.1)

    x_train, y_train, 
    u_zeros = torch.zeros(x_train.shape)
    
    x_train = make_tensor(x_train)
    u_train = make_tensor(u_train)


    x_train = x_train.to(device)
    u_train = u_train.to(device) 
    u_zeros = u_zeros.to(device)


    # data_train = torch.cat((x_train, u_train), axis=1)

    loss_func = nn.MSELoss()

    loss_save = np.inf

    for epoch in range(epochs):
        optim.zero_grad()
        # optim_lambda.zero_grad()

        u_hat = model(x_train)

        loss = 0.0

        loss_b = loss_func(u_hat, u_train)

        loss_f = calc_loss_f(x_train, u_zeros, u_hat, func=loss_func, lambda_1=lambda_1) * 100

        loss += loss_b
        loss += loss_f
        
        
        loss.backward()
        # loss_f.backward()
        

        optim.step()
        # optim_lambda.step()

        with torch.no_grad():
            model.eval()
                
            # loss_t = loss_func(pred_t, u_t)

            if loss < loss_save:
                best_epoch = epoch
                loss_save = copy(loss)
                torch.save(model.state_dict(), './models/model_infer.data')
                # print(".......model updated (epoch = ", epoch+1, ")")

            if epoch % 10 == 0:    
                print("Epoch: {0} | LOSS: {1:.8f} | LOSS_F: {2:.8f} | Lambda: {3:.4f}".format(epoch + 1, loss_b, loss_f, lambda_1.cpu().data[0]))    
            
            if loss < 1e-6:
                break



    print("Best epoch: ", best_epoch)
    print("Best loss: ", loss_save)
    print("Elapsed time: {:.3f} s".format(time.time() - since))
    print("Done!") 
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)


def main():
    train()

if __name__ == '__main__':
    main()


    