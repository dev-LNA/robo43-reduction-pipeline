#!/bin/python3

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate thumbnail images from FITS files.')
    parser.add_argument('input_dir', type=str,
                        help='Directory containing FITS files.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save thumbnail images.')
    return parser.parse_args()


def create_thumbnail(args):
    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fits_files = glob.glob(os.path.join(input_dir, '*.fits'))

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            title = os.path.basename(fits_file)
            xlabel = f' {hdul[0].header.get("FILTER", "")}'
            xlabel += f' {hdul[0].header.get("OBJECT", "")}'
            data = hdul[0].data

            plt.figure(figsize=(8, 6))
            plt.imshow(data, cmap='gray', origin='lower', vmin=np.percentile(data, 5),
                       vmax=np.percentile(data, 95))
            plt.colorbar()
            plt.title(os.path.basename(fits_file))
            plt.xlabel(xlabel)
            output_file = os.path.join(output_dir, os.path.basename(
                fits_file).replace('.fits', '.png'))
            print(f'Saving thumbnail to {output_file}')
            plt.savefig(output_file)
            plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    create_thumbnail(args)
