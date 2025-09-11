#!/bin/python3

import os
from astropy.io import fits
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Invalidate frames in FITS files by setting the 'VALID' keyword to False.")
    parser.add_argument(
        'workdir', help='Working directory containing FITS files')
    parser.add_argument('--list_files', type=str, default='list_files.txt',
                        help='Text file containing list of FITS files to process (one per line)')
    parser.add_argument('--files', nargs='+',
                        help='List of FITS files to process')
    return parser.parse_args()


def invalidate_frames(workdir, files_list):
    for filename in files_list:
        file_path = os.path.join(workdir, filename)
        if not os.path.isfile(file_path):
            print(f"File '{file_path}' not found. Skipping.")
            continue

        try:
            with fits.open(file_path, mode='update') as hdul:
                if 'VALID' in hdul[0].header:
                    hdul[0].header['VALID'] = (
                        False, 'Frame invalidated by script')
                else:
                    hdul[0].header.set(
                        'VALID', False, 'Frame invalidated by script')
                hdul[0].header['OBJECT'] = 'trash'
                hdul.flush()
            print(f"Processed file: {file_path}")
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")


if __name__ == "__main__":
    args = parse_arguments()

    workdir = args.workdir

    if args.files:
        files_to_process = args.files
    else:
        list_file_path = os.path.join(args.workdir, args.list_files)
        if not os.path.isfile(list_file_path):
            raise FileNotFoundError(f"List file '{list_file_path}' not found.")
        else:
            with open(list_file_path, 'r') as f:
                files_to_process = [line.strip() for line in f if line.strip()]

    invalidate_frames(workdir, files_to_process)
