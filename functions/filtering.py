import torch
from torch import nn
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