import argparse
from math import pi

def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", default='', type=str, help="Path to data folder")
    parser.add_argument("--model_root", default='', type=str, help="Path to model parameter folder")
    parser.add_argument("--data_name_test", default='polystyrene_bead_test', type=str)
    parser.add_argument("--experiment", default='', type=str, help="experiment name")
    parser.add_argument("--test_diffraction_list", default='', type=str, help="depth of measured diffraction pattern for test")

    return parser.parse_args()
