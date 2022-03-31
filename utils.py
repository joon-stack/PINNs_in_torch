import torch.autograd as autograd

def calc_deriv(x, input, times):
    if times == 0:
        return input
    res = input
    for _ in range(times):
        res = autograd.grad(res.sum(), x, create_graph=True)[0]
    return res