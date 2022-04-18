import torch 
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import time

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.module1 = nn.Sequential(nn.Linear(1, 10), 
                                      nn.Linear(10, 10),
                                      nn.Linear(10, 10),
                                      nn.Linear(10, 1))
    def forward(self, input):
        act_func = nn.Tanh()

        tmp = input
        for layer in self.module1:
            tmp = act_func(layer(tmp))
        
        return tmp

    def __call__(self, input):
        return self.forward(input)

def train(load=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    epochs = 5000
    model = MLP()

    if load:
        model.load_state_dict(torch.load('saved.data'))

    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_func = nn.MSELoss()

    x = (torch.linspace(0, 1, 10000) * 10).reshape(-1, 1)
    y = (torch.sin(3 * x)).reshape(-1, 1)
    x = x.to(device)
    y = y.to(device)

    loss_save = np.inf
    train_loss = []

    for epoch in range(epochs):
        optim.zero_grad()
        model.train()
        pred = model(x)
        loss = loss_func(pred, y)
        loss.backward()
        optim.step()

        train_loss += [loss.item()]
        

        with torch.no_grad():
            model.eval()
            if loss < loss_save:
                torch.save(model.state_dict(), 'model.data')
                print(".......model updated (epoch = ", epoch+1, ")")
            loss_save = loss.item()

            print("Epoch {} | Loss: {:.3f}".format(epoch+1, loss.item()))
    
    return train_loss



if __name__ == "__main__":
    rand = train(load=False)

    model = MLP()
    model.load_state_dict(torch.load('model.data'))
    x = (torch.linspace(0, 1, 10000) * 10).reshape(-1, 1)
    y1 = model(x).detach().numpy()
    label = torch.sin(3 * x).detach().numpy()

    meta = train(load=True)
    model = MLP()
    model.load_state_dict(torch.load('model.data'))
    y2 = model(x).detach().numpy()

    plt.plot(x, y1, label='pred_rand')
    plt.plot(x, y2, label='pred_meta')
    plt.plot(x, label, label='truth')
    # # plt.show()
    plt.legend()
    plt.savefig('test.png')

    plt.cla()
    plt.plot(np.arange(5000), rand, label='rand')
    plt.plot(np.arange(5000), meta, label='meta')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('training.png')