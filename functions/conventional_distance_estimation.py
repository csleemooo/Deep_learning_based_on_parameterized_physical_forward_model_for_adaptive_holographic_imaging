import torch
from torch import nn

class TC_sparsity(nn.Module):

    def __init__(self):
        super(TC_sparsity, self).__init__()

    def forward(self, x):
        _, _, Nx, Ny = x.shape()
        gradient = torch.sqrt(torch.pow(x[:, :, 0:Nx-1, 0:Ny-1] - x[:, :, 1:Nx, 0:Ny-1], 2) +
                              torch.pow(x[:, :, 0:Nx-1, 0:Ny-1] - x[:, :, 0:Nx-1, 1:Ny], 2))

        std_grad = gradient.std(dim=[-2, -1])
        mean_grad = gradient.mean(dim=[-2, -1])

        Tamura_gradient = torch.sqrt(std_grad/(mean_grad+1e-4))

        return Tamura_gradient.mean()

class Edge_gradient(nn.Module):

    def __init__(self):
        super(Edge_gradient, self).__init__()

    def forward(self, x):
        Nx = x.shape[-2]
        Ny = x.shape[-1]

        gradient = torch.sqrt(torch.pow(x[:, :, 0:Nx-1, 0:Ny-1] - x[:, :, 1:Nx, 0:Ny-1], 2) +
                              torch.pow(x[:, :, 0:Nx-1, 0:Ny-1] - x[:, :, 0:Nx-1, 1:Ny], 2))

        gradient = torch.squeeze(gradient.sum(dim=[-2, -1]))

        return gradient.mean(dim=0) + 1e-4

class LAP(nn.Module):

    def __init__(self):
        super(LAP, self).__init__()

    def forward(self, x):
        Nx = x.shape[-2]
        Ny = x.shape[-1]

        gradient = x[:, :, 0:Nx-2, 1:Ny-1] + x[:, :, 2:Nx, 1:Ny-1] + \
                   x[:, :, 1:Nx-1, 0:Ny-2] + x[:, :, 1:Nx-1, 2:Ny]\
                   -4*x[:, :, 1:Nx-1, 1:Ny-1]

        gradient = torch.pow(gradient, 2)

        gradient = torch.squeeze(gradient.sum(dim=[-2, -1]))

        return gradient.mean(dim=0)

import torch.nn.functional as F

class Laplace_op(nn.Module):
    def __init__(self, args):
        super(Laplace_op, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=1)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)    # filter
        down = filtered[:,:,::2,::2]               # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x):
        out = self.laplacian_kernel(x)
        return out

class Gauss_filt(nn.Module):
    def __init__(self):
        super(Gauss_filt, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(1, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=1)

    def forward(self, x):
        out = self.conv_gauss(x)
        return out