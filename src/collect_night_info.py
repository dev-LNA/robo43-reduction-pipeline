#!/bin/python3

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Collect night information from FITS files.')
    parser.add_argument('directory', type=str,
                        help='Directory containing FITS files')
    return parser.parse_args()


def collect_night_info(workdir):
    fits_files = sorted(glob.glob(os.path.join(workdir, '*.fits')))
    fits_files = [f for f in fits_files if '_proc' not in f]

    filenames = np.array([os.path.basename(f) for f in fits_files])
    objects = np.zeros(len(fits_files), dtype='U20')
    exptimes = np.zeros(len(fits_files), dtype=float)
    ras = np.zeros(len(fits_files), dtype=float)
    decs = np.zeros(len(fits_files), dtype=float)

    for i, f in enumerate(fits_files):
        hdr = fits.getheader(f)
        objects[i] = hdr.get('OBJECT', 'Unknown')
        exptimes[i] = hdr.get('EXPTIME', np.nan)
        ras[i] = hdr.get('RA', np.nan)
        decs[i] = hdr.get('DEC', np.nan)

    df = pd.DataFrame({
        'filename': filenames,
        'object': objects,
        'exptime': exptimes,
        'ra': ras,
        'dec': decs
    })
    output_file = os.path.join(workdir, 'night_info.csv')
    df.to_csv(output_file, index=False)
    print(f"Night information saved to {output_file}")


if __name__ == "__main__":
    args = parse_arguments()
    collect_night_info(args.directory)
