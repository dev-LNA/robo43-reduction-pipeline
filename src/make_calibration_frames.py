#!/bin/python3

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create master calibration frames (bias, dark, flat)."
    )
    parser.add_argument("input_dir", type=str,
                        help="Input directory containing calibration FITS files.")
    parser.add_argument("cal_type", type=str, choices=['bias', 'dark', 'flat'],
                        help="Type of calibration frame to create.")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory to save master calibration frames.")
    parser.add_argument("--save_frame", action="store_true",
                        help="Save the master calibration frame to disk.")
    parser.add_argument("--clobber", action="store_true",
                        help="Overwrite existing master calibration frames.")

    return parser.parse_args()


def logger_setup():
    logger = logging.getLogger('calibration_logger')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class CalibrationMaker:
    def __init__(self, args, logger):
        self.workdir = args.input_dir
        self.cal_type = args.cal_type
        self.output_dir = args.output_dir if args.output_dir else self.workdir
        self.clobber = args.clobber
        self.logger = logger
        self.master_filename = os.path.join(
            self.output_dir, f'master_{self.cal_type}.fits')
        self.save_frame = args.save_frame

    def get_calibration_files(self):
        pattern = os.path.join(self.workdir, '*.fits')
        filelist = sorted(glob.glob(pattern))
        selected_files = []
        filters_list = []
        for filename in filelist:
            header = fits.getheader(filename)
            if self.cal_type == 'bias' and header.get('OBJECT', '').lower() == 'bias':
                if 'master' in filename.lower():
                    continue
                else:
                    selected_files.append(filename)
            elif self.cal_type == 'dark' and header.get('OBJECT', '').lower() == 'dark':
                selected_files.append(filename)
            elif self.cal_type == 'flat' and 'flat' in header.get('OBJECT', '').lower():
                if 'master' in filename.lower():
                    continue
                else:
                    selected_files.append(filename)
                    filters_list.append(header.get('FILTER', 'Unknown'))
        if self.cal_type == 'flat':
            filters_list = set(filters_list)
            self.logger.info(
                f'Found flat frames with filters: {", ".join(filters_list)}')

        if len(selected_files) == 0:
            self.logger.error(
                f'No {self.cal_type} files found in {self.workdir}')
            raise FileNotFoundError(
                f'No {self.cal_type} files found in {self.workdir}')

        return selected_files, filters_list

    def create_master_bias(self, filelist):
        self.logger.info('Creating master bias frame...')
        bias_frames = []
        header = fits.getheader(filelist[0])
        for filename in filelist:
            self.logger.debug(f'Reading {filename}')
            with fits.open(filename) as hdul:
                bias_frames.append(hdul[0].data)
        master_bias = np.median(bias_frames, axis=0)
        header['NCOMBINE'] = (len(filelist), 'Number of combined frames')
        hdul = fits.PrimaryHDU(master_bias, header=header)

        if self.save_frame:
            self.logger.info(
                f'Saving master bias frame to {self.master_filename}')
            hdul.writeto(self.master_filename, overwrite=self.clobber)

        return hdul

    def create_master_dark(self, filelist):
        self.logger.info('Creating master dark frame...')
        # TODO: implement dark frame creation
        dark_frames = []
        for filename in filelist:
            self.logger.debug(f'Reading {filename}')
            with fits.open(filename) as hdul:
                dark_frames.append(hdul[0].data)
        master_dark = np.median(dark_frames, axis=0)
        return master_dark

    def create_master_flat(self, filelist, filters_list):
        self.logger.info('Creating master flat frame...')
        flat_frames = []
        master_flats = {}
        # remove master_bias from the images before combining
        master_bias_path = os.path.join(self.output_dir, 'master_bias.fits')
        if os.path.exists(master_bias_path):
            self.logger.info(f'Using master bias from {master_bias_path}')
            with fits.open(master_bias_path) as hdul:
                master_bias = hdul[0].data
        else:
            self.logger.error(
                f'Master bias frame not found at {master_bias_path}. Cannot proceed with flat creation.')
            raise FileNotFoundError(
                f'Master bias frame not found at {master_bias_path}. Cannot proceed with flat creation.')

        for filter_name in filters_list:
            self.logger.info(f'Processing filter: {filter_name}')
            filter_files = [f for f in filelist if fits.getheader(
                f).get('FILTER', 'Unknown') == filter_name]
            if len(filter_files) == 0:
                self.logger.warning(
                    f'No flat frames found for filter {filter_name}')
                continue
            self.logger.info(
                f'Found {len(filter_files)} frames for filter {filter_name}')
            flat_frames = []
            for filename in filter_files:
                self.logger.debug(f'Reading {filename}')
                with fits.open(filename) as hdul:
                    try:
                        subtracted_frame = hdul[0].data - master_bias
                        flat_frames.append(
                            subtracted_frame / np.median(subtracted_frame))
                    except ValueError as e:
                        self.logger.error(
                            f'Error processing {filename}: {e} for filter {filter_name}')
            if len(flat_frames) == 0:
                self.logger.warning(
                    f'No valid flat frames left for filter {filter_name} after exclusion')
                continue
            master_flat = np.median(flat_frames, axis=0)
            variance = np.var(flat_frames, axis=0)
            master_flat = sigma_clip(master_flat, sigma=5, maxiters=5,
                                     cenfunc='median', stdfunc='mad_std',
                                     masked=True, copy=True)
            # create weight map where the masked values are 0 and other are the inverse of the variance
            weight_map = np.where(master_flat.mask, 0,
                                  1.0 / variance)
            # replace masked values in master_flat with 1.0
            master_flat = np.where(master_flat.mask, 1.0, master_flat)

            if self.save_frame:
                header = fits.getheader(filter_files[0])
                header['NCOMBINE'] = (
                    len(filter_files), 'Number of combined frames')
                header['FILTER'] = (filter_name, 'Filter name')
                hdul = fits.PrimaryHDU(master_flat, header=header)
                master_flats[filter_name] = hdul
                flat_filename = os.path.join(
                    self.output_dir, f'master_flat_{filter_name}.fits')
                self.logger.info(
                    f'Saving master flat frame to {flat_filename}')
                hdul.writeto(flat_filename, overwrite=self.clobber)
                self.logger.info(
                    f'Saving weight map to {flat_filename.replace(".fits", "_weight.fits")}')
                weight_hdul = fits.PrimaryHDU(weight_map, header=header)
                weight_hdul.writeto(flat_filename.replace(
                    ".fits", "_weight.fits"), overwrite=self.clobber)
            else:
                master_flats[filter_name] = master_flat
        return master_flats

    def plot_frame(self, hdu):
        self.logger.info('Plotting master frame...')
        if isinstance(hdu, dict):
            for filter_name, frame in hdu.items():
                title = f'MASTER_flat_{filter_name}'
                title += ' NCOMBINE=' + \
                    str(hdu[filter_name].header.get('NCOMBINE', 'N/A'))
                plt.figure(figsize=(10, 8))
                plt.imshow(frame.data, cmap='gray', vmin=np.percentile(frame.data, 5),
                           vmax=np.percentile(frame.data, 95))
                plt.colorbar()
                plt.title(title)
                plt.xlabel('Pixels')
                plt.ylabel('Pixels')
                img_name = os.path.join(self.output_dir,
                                        f'master_flat_{filter_name}.png')
                plt.savefig(img_name)
                self.logger.info(f'Saved image to {img_name}')
                plt.close()
        else:
            title = 'master_' + self.cal_type.capitalize()
            title += ' NCOMBINE=' + str(hdu.header.get('NCOMBINE', 'N/A'))
            plt.figure(figsize=(10, 8))
            plt.imshow(hdu.data, cmap='gray', vmin=np.percentile(hdu.data, 5),
                       vmax=np.percentile(hdu.data, 95))
            plt.colorbar()
            plt.title(title)
            plt.xlabel('Pixels')
            plt.ylabel('Pixels')
            img_name = os.path.join(self.output_dir,
                                    f'master_{self.cal_type}.png')
            plt.savefig(img_name)
            self.logger.info(f'Saved image to {img_name}')
            plt.close()

    def main(self):
        if os.path.exists(self.master_filename) and not self.clobber:
            self.logger.error(
                f'Master {self.cal_type} frame already exists at {self.master_filename}. Use --clobber to overwrite.')
            return
        if self.cal_type == 'bias':
            filelist, _ = self.get_calibration_files()
            master_frame = self.create_master_bias(filelist)
            self.plot_frame(master_frame)
        elif self.cal_type == 'dark':
            self.logger.warning("Not implemented. Use bias mode instead.")
            return
            filelist, _ = self.get_calibration_files()
            master_frame, _ = self.create_master_dark(filelist)
            self.plot_frame(master_frame)
        elif self.cal_type == 'flat':
            filelist, filters_list = self.get_calibration_files()
            master_frame = self.create_master_flat(filelist, filters_list)
            self.plot_frame(master_frame)

        self.logger.info('Done!')


if __name__ == "__main__":
    args = parse_args()
    logger = logger_setup()
    maker = CalibrationMaker(args, logger)
    maker.main()
