#!/bin/python3

import os
from sewpy import SEW
import glob
import argparse
import logging


def setup_logging(verbose=False, logfile=None, loglevel=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel if verbose else logging.WARNING)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] @%(module)s.%(funcName)s() %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(loglevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process FITS images to extract photometric data.')
    parser.add_argument('input_dir', type=str,
                        help='Directory containing FITS images.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save output files.')
    parser.add_argument('--files_to_process', type=str,
                        default='files_to_process.txt',
                        help='File listing FITS files to process.')
    parser.add_argument('--logfile', type=str, default=None,
                        help='Path to log file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    return parser.parse_args()


def get_fits_files(input_dir, files_to_process):
    if os.path.isfile(os.path.join(input_dir, files_to_process)):
        files_to_process = os.path.join(input_dir, files_to_process)
        with open(files_to_process, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
    else:
        files = glob.glob(os.path.join(input_dir, '*.fits'))

    files = [f.replace('.fits', '_proc.fits') for f in files]

    return files


def run_sewpy(image_path, output_dir, sigma_clip=2.0):
    files_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    proc_path = image_path
    if not os.path.exists(proc_path):
        logger.error(f'File not found: {proc_path}')
        return
    out_params = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'ALPHA_J2000', 'DELTA_J2000',
                  'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO',
                  'FWHM_IMAGE', 'FLAGS', 'CLASS_STAR']
    sex_config = {
        "DETECT_TYPE": "CCD",
        "DETECT_MINAREA": 8,
        "DETECT_THRESH": sigma_clip,
        "ANALYSIS_THRESH": 3.0,
        "FILTER": "Y",
        "FILTER_NAME": os.path.join(files_path, 'data', 'tophat_3.0_3x3.conv'),
        "DEBLEND_NTHRESH": 32,
        "DEBLEND_MINCONT": 0.005,
        "CLEAN": "Y",
        "CLEAN_PARAM": 1.0,
        "MASK_TYPE": "CORRECT",
        "PHOT_APERTURES": 10,
        "PHOT_AUTOPARAMS": '2.5,5',
        # "PHOT_PETROPARAMS": '2.0,2.73',
        # "PHOT_FLUXFRAC": '0.2,0.5,0.7,0.9',
        "SATUR_LEVEL": 50000,
        "MAG_ZEROPOINT": 20,
        "MAG_GAMMA": 4.0,
        "GAIN": 10,
        "PIXEL_SCALE": 0.53,
        "SEEING_FWHM": 3.0,
        "STARNNW_NAME": os.path.join(files_path, 'data', 'default.nnw'),
        "BACK_SIZE": 64,
        "BACK_FILTERSIZE": 3,
        "BACKPHOTO_TYPE": "GLOBAL",
        "BACKPHOTO_THICK": 24,
        # "CHECKIMAGE_TYPE": "SEGMENTATION",
        # "CHECKIMAGE_NAME": pathtoseg
    }
    sew = SEW(workdir=output_dir, config=sex_config,
              sexpath='source-extractor', params=out_params)
    sources = sew(proc_path)['table']


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    files_to_process = args.files_to_process

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fits_files = get_fits_files(input_dir, files_to_process)
    logger.info(f'Found {len(fits_files)} files to process.')

    for fits_file in fits_files:
        file_to_process = os.path.join(input_dir, fits_file)
        logger.info(f'Processing file: {fits_file}')
        try:
            run_sewpy(file_to_process, output_dir)
            logger.info(f'Successfully processed {fits_file}')
        except Exception as e:
            logger.error(f'Error processing {fits_file}: {e}')


if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logging(args.verbose, args.logfile)

    main(args)
