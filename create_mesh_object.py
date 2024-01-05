from utils.visualize import *

import argparse

def main(args):
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='results.csv')
    parser.add_argument('--n_classes', type=int, default=4)

    parser.add_argument('--model', default='',required=True, help="path to model (to continue training)")
    parser.add_argument('--outdir', type=str, default='.', help='output dir to save generated meshes')

    args = parser.parse_args()

    main(args)