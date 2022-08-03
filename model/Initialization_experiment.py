import argparse
from math import pi

def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", type=str, help="Path to data folder")
    parser.add_argument("--data_name_gt", default='polystyrene_bead', type=str, help="complex amplitude")
    parser.add_argument("--data_name_diffraction", default='polystyrene_bead', type=str, help="diffraction pattern")
    parser.add_argument("--data_name_test", default='polystyrene_bead_test', type=str)
    parser.add_argument("--result_root", type=str, help="Path to save folder")
    parser.add_argument("--experiment", type=str, help="experiment name")
    parser.add_argument("--train_gt_ratio", default=1, type=float, help="the percentage of complex amplitude used for train")
    parser.add_argument("--train_diffraction_ratio", default=1, type=float, help="the percentage of diffracted intensity used for train")
    parser.add_argument("--train_diffraction_list", default='13', type=str, help="depth of measured diffraction pattern for train")
    parser.add_argument("--test_diffraction_list", default='7,8,9,10,11,12,13,14,15,16,17', type=str, help="depth of measured diffraction pattern for test")


    # network type
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--batch_mode", default='G', type=str)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument("--initial_channel", default=64, type=int)

    # hyper-parameter
    parser.add_argument("--output_channel", default=2, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--iterations", default=10000, type=int)
    parser.add_argument("--chk_iter", default=100, type=int)
    parser.add_argument("--model_chk_iter", default=2000, type=int)
    parser.add_argument("--lr_disc", default=1e-4, type=float)
    parser.add_argument("--lr_gen", default=1e-3, type=float)
    parser.add_argument("--lr_decay_epoch", default=5, type=int)
    parser.add_argument("--lr_decay_rate", default=0.95, type=float)
    parser.add_argument("--distance_regularizer", default=100, type=float)
    parser.add_argument("--penalty_regularizer", default=20, type=int)
    parser.add_argument("--ssim_regularizer", default=0, type=int)
    parser.add_argument("--gan_regularizer", default=1, type=int)
    parser.add_argument("--diffraction_regularizer", default=100, type=float)
    parser.add_argument("--field_regularizer", default=100, type=float)
    parser.add_argument("--phase_normalize", default=2*pi, type=float)

    # experiment parameter
    parser.add_argument("--wavelength", default=532e-9, type=float)
    parser.add_argument("--pixel_size", default=6.5e-6, type=float)
    parser.add_argument("--distance_min", default=7, type=int)
    parser.add_argument("--distance_max", default=17, type=int)

    return parser.parse_args()
