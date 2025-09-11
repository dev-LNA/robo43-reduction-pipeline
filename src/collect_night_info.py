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
    filters = np.zeros(len(fits_files), dtype='U10')
    ras = np.zeros(len(fits_files), dtype='U20')
    decs = np.zeros(len(fits_files), dtype='U20')
    date_obs = np.zeros(len(fits_files), dtype='U23')
    is_valid = np.ones(len(fits_files), dtype=bool)

    for i, f in enumerate(fits_files):
        hdr = fits.getheader(f)
        objects[i] = hdr.get('OBJECT', 'Unknown')
        exptimes[i] = hdr.get('EXPTIME', 0.0)
        filters[i] = hdr.get('FILTER', 'None')
        ras[i] = hdr.get('RA', 'None')
        decs[i] = hdr.get('DEC', 'None')
        date_obs[i] = hdr.get('DATE-OBS', 'Unknown')
        if 'VALID' in hdr and not hdr['VALID']:
            is_valid[i] = False
        else:
            is_valid[i] = True

    df = pd.DataFrame({
        'filename': filenames,
        'object': objects,
        'exptime': exptimes,
        'filter': filters,
        'ra': ras,
        'dec': decs,
        'date_obs': date_obs,
        'is_valid': is_valid
    })
    output_file = os.path.join(workdir, 'night_info.csv')
    df.to_csv(output_file, index=False)
    print(f"Night information saved to {output_file}")


if __name__ == "__main__":
    args = parse_arguments()
    collect_night_info(args.directory)
