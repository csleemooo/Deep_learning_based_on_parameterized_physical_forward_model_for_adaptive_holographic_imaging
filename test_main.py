import os, sys
import random
import matplotlib
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from model.Forward_operator import Holo_Generator
from model.Inverse_operator import Distance_Generator, Field_Generator
from model.Initialization_test import parse_args
from functions.functions import *
from functions.Data_Loader_custom import Holo_Recon_Dataloader

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..', 'model'))
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(77)
random.seed(77)

if __name__ == '__main__':

    args = parse_args()
    params = torch.load(os.path.join(args.model_root, args.experiment, 'model.pth'))
    model_args = params['args']

    model_args.distance_normalize=model_args.distance_max - model_args.distance_min
    model_args.distance_normalize_constant=model_args.distance_min/model_args.distance_normalize

    data_path_test = os.path.join(args.data_root, args.data_name_test)
    saving_path = os.path.join(args.model_root, args.experiment)
    make_path(saving_path)

    # load data
    transform_img = transforms.Compose([transforms.ToTensor()])
    ## load test data
    test_diffraction_list = [int(i) for i in args.test_diffraction_list.split(',')]

    try:
        test_gt_loader = Holo_Recon_Dataloader(root=data_path_test, data_type=['gt_amplitude', 'gt_phase'],
                                           image_set='test', transform=transform_img)
    except:
        test_gt_loader = None
    test_diffraction_loader = Holo_Recon_Dataloader(root=data_path_test, data_type=['holography'], image_set='test',
                                             transform=transform_img, holo_list=test_diffraction_list)
    N_test = test_diffraction_loader.__len__()
    ##############################################################

    # define model
    model_args.output_channel=2
    diffraction_G = Holo_Generator(model_args).to(device=device)
    distance_G = Distance_Generator(model_args).to(device=device)
    Field_G = Field_Generator(model_args).to(device=device)

    distance_G.load_state_dict(params['distance_G_state_dict'])
    Field_G.load_state_dict(params['Field_G_state_dict'])

    Field_G.eval()
    distance_G.eval()

    test_gt = iter(DataLoader(test_gt_loader, batch_size=1, shuffle=False))
    test_diffraction = iter(DataLoader(test_diffraction_loader, batch_size=1, shuffle=False))

    p = os.path.join(saving_path, 'test_image_result')
    make_path(p)

    for b in range(N_test):

        if test_gt_loader:
            real_amplitude, real_phase = next(test_gt)
            real_amplitude = real_amplitude.to(device=device).float()
            real_phase = real_phase.to(device=device).float()

        diffraction = next(test_diffraction).to(device).float()
        real_distance = test_diffraction_list[b]

        ## generate test amplitude and distance
        fake_amplitude, fake_phase = Field_G(diffraction)
        fake_distance = distance_G(diffraction)

        ## crop for comparison
        fake_diffraction = diffraction_G(fake_amplitude, fake_phase, fake_distance)
        fake_diffraction = fake_diffraction.cpu().detach().numpy()[0][0]
        diffraction = diffraction.cpu().detach().numpy()[0][0]
        fake_distance = (fake_distance.item() + model_args.distance_normalize_constant) * model_args.distance_normalize

        fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0]
        fake_phase = fake_phase.cpu().detach().numpy()[0][0] * model_args.phase_normalize

        if test_gt_loader:
            real_amplitude = real_amplitude.cpu().detach().numpy()[0][0]
            real_phase = real_phase.cpu().detach().numpy()[0][0]
        else:
            real_amplitude = np.zeros_like(fake_amplitude)
            real_phase = np.zeros_like(fake_phase)

        save_fig_(save_path=os.path.join(p, f'test{b+1}.png'), result_data=
                 [diffraction, fake_diffraction, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance], args=model_args)
