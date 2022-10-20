import os, sys
from itertools import chain
import random
import matplotlib
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn

from model.Forward_operator import Holo_Generator
from model.Inverse_operator import Distance_Generator, Field_Generator, Discriminator
from model.Initialization_experiment import parse_args
from functions.functions import *
from functions.gradient_penalty import calc_gradient_penalty
from functions.SSIM import SSIM
from functions.Data_Loader_custom import Holo_Recon_Dataloader

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), '..', 'model'))
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(777)
random.seed(777)

if __name__ == '__main__':

    args=parse_args()

    args.distance_normalize=args.distance_max - args.distance_min
    args.distance_normalize_constant=args.distance_min/args.distance_normalize

    data_path_gt_train = os.path.join(args.data_root, args.data_name_gt)
    data_path_diffraction_train = os.path.join(args.data_root, args.data_name_diffraction)
    data_path_test = os.path.join(args.data_root, args.data_name_test)

    saving_path = os.path.join(args.result_root, args.experiment)

    # load data
    transform_img = transforms.Compose([transforms.ToTensor()])
    ## load train data
    train_diffraction_list = [int(i) for i in args.train_diffraction_list.split(',')]
    train_gt_loader = Holo_Recon_Dataloader(root=data_path_gt_train, data_type=['gt_amplitude', 'gt_phase'],
                                            image_set='train', transform=transform_img, train_type="train",
                                            ratio=args.train_gt_ratio)
    args.train_gt_data_set = train_gt_loader.data_list
    train_diffraction_loader = Holo_Recon_Dataloader(root=data_path_diffraction_train, data_type=['holography'],
                                              image_set='train', transform=transform_img, train_type="train",
                                              holo_list=train_diffraction_list, ratio=args.train_diffraction_ratio)
    args.train_diffraction_dataset = train_diffraction_loader.data_list

    train_gt_loader = DataLoader(train_gt_loader, batch_size=args.batch_size, shuffle=True)
    train_diffraction_loader = DataLoader(train_diffraction_loader, batch_size=args.batch_size, shuffle=True)

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
    diffraction_G = Holo_Generator(args).to(device=device)
    distance_G = Distance_Generator(args).to(device=device)
    Field_G = Field_Generator(args).to(device=device)
    Field_D = Discriminator(args, input_channel=3).to(device=device)

    # optimizer
    op_G = torch.optim.Adam(chain(Field_G.parameters(), distance_G.parameters()), lr=args.lr_gen, betas=(0.5, 0.9))
    op_D = torch.optim.Adam(Field_D.parameters(), lr=args.lr_disc, betas=(0.5, 0.9))

    # scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.StepLR(op_G, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
    lr_scheduler_D = torch.optim.lr_scheduler.StepLR(op_D, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    # loss
    criterion_cycle = nn.L1Loss()
    criterion_MSE = nn.MSELoss()
    criterion_wgan = torch.mean
    criterion_ssim = SSIM()

    loss_sum_G, loss_sum_cycle_diffraction, loss_sum_cycle_field, loss_sum_cycle_distance = 0, 0, 0, 0
    loss_G_list, loss_cycle_diffraction_list, loss_cycle_field_list, loss_cycle_distance_list = [], [], [], []

    loss_sum_Field_D, loss_sum_Field_D_penalty = 0, 0
    loss_field_D_list, loss_field_D_penalty_list = [], []


    for it in range(args.iterations):

        Field_G.train()
        Field_D.train()
        distance_G.train()

        real_amplitude, real_phase = next(iter(train_gt_loader))
        diffraction = next(iter(train_diffraction_loader)).to(device).float()

        real_amplitude = center_crop(real_amplitude, args.crop_size).to(device).float()
        real_phase = center_crop(real_phase, args.crop_size).to(device).float()/ (args.phase_normalize)
        diffraction = center_crop(diffraction, args.crop_size)

        real_distance = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()

        # top cycle
        fake_amplitude, fake_phase = Field_G(diffraction)
        fake_distance = distance_G(diffraction)
        consistency_diffraction = diffraction_G(fake_amplitude, fake_phase, fake_distance)

        # bottom cycle
        fake_diffraction = diffraction_G(real_amplitude, real_phase, real_distance).float()
        consistency_amplitude, consistency_phase = Field_G(fake_diffraction)
        consistency_distance = distance_G(fake_diffraction)

        real_field = torch.cat([real_amplitude, real_phase], dim=1)
        fake_field = torch.cat([fake_amplitude, fake_phase], dim=1)
        consistency_field = torch.cat([consistency_amplitude, consistency_phase], dim=1)

        # train discriminator
        if args.gan_regularizer:
            op_D.zero_grad()

            fake_D = Field_D(fake_field.detach())
            real_D = Field_D(real_field)

            D_penalty_loss = args.penalty_regularizer*calc_gradient_penalty(Field_D, real_field, fake_field, real_amplitude.shape[0])
            D_adversarial_loss = criterion_wgan(fake_D.mean(dim=(-2,-1))) - criterion_wgan(real_D.mean(dim=(-2,-1)))

            loss_field_D = D_penalty_loss + D_adversarial_loss

            loss_sum_Field_D_penalty += D_penalty_loss.item()
            loss_sum_Field_D += D_adversarial_loss.item()

            loss_field_D.backward()  # maximize cost for discriminator
            op_D.step()  # step

        ## train complex amplitude and distance generator
        op_G.zero_grad()

        G_loss = args.gan_regularizer*-1*criterion_wgan(Field_D(fake_field).mean(dim=(-2, -1)))
        consistency_ssim = args.ssim_regularizer*((1-criterion_ssim(real_amplitude, consistency_amplitude))
                                                  + (1-criterion_ssim(real_phase, consistency_phase)))

        consistency_loss_diffraction = args.diffraction_regularizer*criterion_cycle(consistency_diffraction, diffraction)
        consistency_loss_field = args.field_regularizer*criterion_cycle(consistency_field, real_field)

        consistency_loss_distance = args.distance_regularizer*criterion_cycle(consistency_distance, real_distance)

        loss_4_gan_x = G_loss + consistency_loss_diffraction + consistency_loss_field + consistency_loss_distance + consistency_ssim

        loss_sum_G += G_loss.item()
        loss_sum_cycle_diffraction += consistency_loss_diffraction.item()
        loss_sum_cycle_field += consistency_loss_field.item() + consistency_ssim.item()
        loss_sum_cycle_distance += consistency_loss_distance.item()

        loss_4_gan_x.backward()
        op_G.step()

        if (it + 1) % args.chk_iter == 0:

            lr_scheduler_G.step()
            lr_scheduler_D.step()

            loss_sum_G=round(loss_sum_G/args.chk_iter, 4)
            loss_sum_cycle_diffraction=round(loss_sum_cycle_diffraction/args.chk_iter, 4)
            loss_sum_cycle_distance=round(loss_sum_cycle_distance/args.chk_iter, 4)
            loss_sum_Field_D=round(loss_sum_Field_D/args.chk_iter, 4)
            loss_sum_Field_D_penalty=round(loss_sum_Field_D_penalty/args.chk_iter, 4)
            loss_sum_cycle_field=round(loss_sum_cycle_field/args.chk_iter, 4)


            print(f"[Iterations : {it+1}/{args.iterations}] : L_wgan_generator: {loss_sum_G}, L_cycle_diffraction: {loss_sum_cycle_diffraction}, L_cycle_field:{loss_sum_cycle_field}, L_cycle_distance: {loss_sum_cycle_distance}, L_wgan_discriminator: {loss_sum_Field_D}, L_wgan_penalty: {loss_sum_Field_D_penalty}")

            make_path(saving_path)
            make_path(os.path.join(saving_path, 'generated'))

            # path for saving result
            p = os.path.join(saving_path, 'generated', 'iterations_' + str(it + 1))
            make_path(p)

            loss_G_list.append(loss_sum_G)
            loss_cycle_diffraction_list.append(loss_sum_cycle_diffraction)
            loss_cycle_field_list.append(loss_sum_cycle_field)
            loss_cycle_distance_list.append(loss_sum_cycle_distance)
            loss_field_D_list.append(loss_sum_Field_D)
            loss_field_D_penalty_list.append(loss_sum_Field_D_penalty)

            loss_sum_G, loss_sum_cycle_diffraction, loss_sum_cycle_field, loss_sum_cycle_distance = 0, 0, 0, 0
            loss_sum_Field_D, loss_sum_Field_D_penalty = 0, 0

            Field_G.eval()
            Field_D.eval()
            distance_G.eval()

            if (it + 1) % (args.model_chk_iter) == 0:
                loss = {}
                loss['G_adversarial_loss'] = loss_G_list
                loss['D_adversarial_loss'] = loss_field_D_list
                loss['D_penalty_loss'] = loss_field_D_penalty_list
                loss['G_cycle_diffraction_loss'] = loss_cycle_diffraction_list
                loss['G_cycle_field_loss'] = loss_cycle_field_list
                loss['G_cycle_distance_loss'] = loss_cycle_distance_list

                save_data = {'iteration': it+1,
                             'Field_G_state_dict': Field_G.state_dict(),
                             'Field_D_state_dict': Field_D.state_dict(),
                             'distance_G_state_dict': distance_G.state_dict(),
                             'loss': loss,
                             'args': args}

                torch.save(save_data, os.path.join(p, "model.pth"))

            test_gt = iter(DataLoader(test_gt_loader, batch_size=1, shuffle=False))
            test_diffraction = iter(DataLoader(test_diffraction_loader, batch_size=1, shuffle=False))

            for b in range(N_test):
                if test_gt_loader:
                    real_amplitude, real_phase = next(test_gt)
                    real_amplitude = real_amplitude.to(device=device).float()
                    real_phase = real_phase.to(device=device).float()
                    real_distance = test_diffraction_list[b]
                else:
                    real_distance = 0
                    
                diffraction = next(test_diffraction).to(device).float()
                

                ## generate test amplitude and distance
                fake_amplitude, fake_phase = Field_G(diffraction)
                fake_distance = distance_G(diffraction)

                ## crop for comparison
                fake_diffraction = diffraction_G(fake_amplitude, fake_phase, fake_distance)
                fake_diffraction = fake_diffraction.cpu().detach().numpy()[0][0]
                diffraction = diffraction.cpu().detach().numpy()[0][0]
                fake_distance = (fake_distance.item() + args.distance_normalize_constant) * args.distance_normalize

                fake_amplitude = fake_amplitude.cpu().detach().numpy()[0][0]
                fake_phase = fake_phase.cpu().detach().numpy()[0][0] * args.phase_normalize

                if test_gt_loader:
                    real_amplitude = real_amplitude.cpu().detach().numpy()[0][0]
                    real_phase = real_phase.cpu().detach().numpy()[0][0]
                else:
                    real_amplitude = np.zeros_like(fake_amplitude)
                    real_phase = np.zeros_like(fake_phase)

                save_fig_(save_path=os.path.join(p, f'test{b+1}.png'), result_data=
                         [diffraction, fake_diffraction, real_amplitude, fake_amplitude, real_phase, fake_phase, real_distance, fake_distance], args=args)

    loss = {}
    loss['G_adversarial_loss'] = loss_G_list
    loss['D_adversarial_loss'] = loss_field_D_list
    loss['D_penalty_loss'] = loss_field_D_penalty_list
    loss['G_cycle_diffraction_loss'] = loss_cycle_diffraction_list
    loss['G_cycle_field_loss'] = loss_cycle_field_list
    loss['G_cycle_distance_loss'] = loss_cycle_distance_list

    save_data = {'iteration': args.iterations,
                 'Field_G_state_dict': Field_G.state_dict(),
                 'Field_D_state_dict': Field_D.state_dict(),
                 'distance_G_state_dict': distance_G.state_dict(),
                 'loss': loss,
                 'args': args}

    torch.save(save_data, os.path.join(saving_path, "model.pth"))
