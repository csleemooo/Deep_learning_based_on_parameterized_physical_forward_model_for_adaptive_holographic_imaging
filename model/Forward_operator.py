from functions.Angular_Spectrum_Method import ASM
from torch import nn
import torch

class Holo_Generator(nn.Module):
    '''
    Implementation for distance parameterized physical forward model.
    We use Angular Spectrum method as forward model and object-to-sensor distance is parameterized.
    '''

    def __init__(self, args):
        super(Holo_Generator, self).__init__()
        self.wavelength = args.wavelength
        self.pixel_size = args.pixel_size
        self.distance_normalize = args.distance_normalize
        self.distance_normalize_constant = args.distance_normalize_constant
        self.zero_padding = args.zero_padding
        self.phase_normalize = args.phase_normalize


    def forward(self, amplitude, phase, d):

        d = ((d+self.distance_normalize_constant)*self.distance_normalize)*0.001

        phase = phase*self.phase_normalize
        O_low = amplitude*torch.exp(1j*phase)

        O_holo = ASM(O_low, self.wavelength, d, self.pixel_size, zero_padding=self.zero_padding)

        I = torch.pow(torch.abs(O_holo), 2)

        return I

