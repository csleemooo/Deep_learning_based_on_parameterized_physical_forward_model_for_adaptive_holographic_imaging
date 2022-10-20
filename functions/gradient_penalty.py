import torch
from torch import autograd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE):

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1)  # alpha is random number sampled from uniform distribution
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device=device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # outputs : f(x)
    # inputs : x  --> calculate gradient w.r.t inputs x
    # changed 22.10.20
    # _, nc, N11, N12 = interpolates.size()
    # _, _, N21, N22 = disc_interpolates.size()

    gradients = autograd.grad(outputs=disc_interpolates[:, 0, 0, 0].reshape(BATCH_SIZE, 1, 1, 1), inputs=interpolates,
                              grad_outputs=torch.ones((BATCH_SIZE, 1, 1, 1)).to(device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0].reshape(BATCH_SIZE, -1)

    # gradient penalty for patch gan
    grad_norm2 = gradients.norm(2, dim=(-2, -1))  # matrix norm 2
    gradient_penalty = ((grad_norm2 - 1) ** 2).mean(dim=0)  # mean for batch

    return gradient_penalty
