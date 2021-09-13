import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import time
from copy import copy

class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()

        self.hidden_layer_1 = nn.Linear(1, 20)
        self.hidden_layer_2 = nn.Linear(20, 20)
        self.hidden_layer_3 = nn.Linear(20, 20)
        self.hidden_layer_4 = nn.Linear(20, 20)
        self.hidden_layer_5 = nn.Linear(20, 20)
        self.output_layer   = nn.Linear(20, 1)


    def forward(self, x):
        input_data      = x
        act_func        = nn.Tanh()
        act_layer_1     = act_func(self.hidden_layer_1(input_data))
        act_layer_2     = act_func(self.hidden_layer_2(act_layer_1))
        act_layer_3     = act_func(self.hidden_layer_3(act_layer_2))
        act_layer_4     = act_func(self.hidden_layer_3(act_layer_3))
        act_layer_5     = act_func(self.hidden_layer_3(act_layer_4))
        u_hat           = self.output_layer(act_layer_5)

        return u_hat
    

def governing_equation(x, w=1.0, l=1.0):
    value = x * (x - l) * (x * x - x * l - l * l) * w / 24
    return value

def calc_loss_f(x, y, u_hat, func, lambda_1):
    u_hat_x     = autograd.grad(u_hat.sum(), x, create_graph=True)[0]
    # u_hat_t     = autograd.grad(u_hat.sum(), t, create_graph=True)[0]
    u_hat_x_x   = autograd.grad(u_hat_x.sum(), x, create_graph=True)[0]
    u_hat_x_x_x = autograd.grad(u_hat_x_x.sum(), x, create_graph=True)[0]
    u_hat_x_x_x_x = autograd.grad(u_hat_x_x_x.sum(), x, create_graph=True)[0]
    # u_hat_t_t   = autograd.grad(u_hat_t.sum(), t, create_graph=True)[0]

    # print(lambda_1.mean().data, u_hat_x_x_x_x.mean().data)
    f = lambda_1 * u_hat_x_x_x_x - 1

    # f = u_hat_t_t - 4 * u_hat_x_x

    
    return func(f, y)

def make_training_data(size, lb=0.0, ub=1.0):
    np.random.seed(0)
    x = np.random.uniform(low=lb, high=ub, size=(size, 1))
    value = governing_equation(x)
    return x, value

def make_tensor(x, requires_grad=True):
    t = torch.from_numpy(x)
    t = t.float()
    t.requires_grad=requires_grad
    return t

def train(epochs=50000):
    since = time.time()
    size = 5000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Current device:", device)
    
    model = Inference()
    model = model.to(device)

    lambda_1 = torch.tensor([0.0]).to(device).detach().requires_grad_(True)

    lambda_1 = lambda_1.cuda()

    print(lambda_1.grad)
 
    # model = nn.DataParallel(model)
    
    optim = torch.optim.Adam([{'params': model.parameters()},
                             {'params': lambda_1, 'lr': 0.01}
                            ],
                            lr=0.01)

    # optim_lambda = torch.optim.Adam([lambda_1], lr=0.1)

    x_train, u_train = make_training_data(size)
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

        loss_b = loss_func(u_hat, u_train) / torch.abs(u_hat.mean())

        loss_f = calc_loss_f(x_train, u_zeros, u_hat, func=loss_func, lambda_1=lambda_1)

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


    