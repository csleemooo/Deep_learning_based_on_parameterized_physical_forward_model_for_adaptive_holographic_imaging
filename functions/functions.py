import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi, sqrt

def center_crop(H, size):
    batch, channel, Nh, Nw = H.size()

    return H[:, :, (Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def random_crop(H, size):

    batch, channel, Nh, Nw = H.size()

    x_off = int(np.floor(np.random.rand() * (Nh-size)) + size/2)
    y_off = int(np.floor(np.random.rand() * (Nw-size)) + size/2)

    return H[:, :, (x_off - size//2) : (x_off+size//2), (y_off - size//2) : (y_off+size//2)]

def center_crop_numpy(H, size):
    Nh = H.shape[0]
    Nw = H.shape[1]

    return H[(Nh - size)//2 : (Nh+size)//2, (Nw - size)//2 : (Nw+size)//2]

def amp_pha_generate(real, imag):
    field = real + 1j*imag
    amplitude = np.abs(field)
    phase = np.angle(field)

    return amplitude, phase

def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)


def save_fig_(save_path, result_data, args):

    holo, fake_holo, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance = result_data

    fig2 = plt.figure(2, figsize=[12, 8])

    plt.subplot(2, 3, 1)
    plt.title('input holography')
    plt.imshow(holo, cmap='gray', vmax=0.5, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.title('ground truth %dmm'%real_distance)
    plt.imshow(real_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.title('output ' + str(np.round(fake_distance, 2)) + 'mm')
    plt.imshow(fake_amplitude, cmap='gray', vmax=1, vmin=0)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title('generated_holography')
    plt.imshow(fake_holo, cmap='gray', vmax=0.5, vmin=0)
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.title('ground truth phase')
    plt.imshow(real_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.title('output phase')
    plt.imshow(fake_phase, cmap='hot', vmax=2.5, vmin=-0.1)
    plt.axis('off')
    plt.colorbar()

    # fig_save_name = os.path.join(p, 'test' + str(b + 1) + '.png')
    fig2.savefig(save_path)
    plt.close(fig2)
