#!/bin/python3

import os
import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from reproject import reproject_interp
from photutils.psf import fit_fwhm
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Create coloured images from FITS files.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input FITS files.')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save output coloured FITS files.')
    parser.add_argument('--b_channel', type=str, nargs='+',
                        required=True, help='List of FITS files for blue channel.')
    parser.add_argument('--g_channel', type=str, nargs='+',
                        required=True, help='List of FITS files for green channel.')
    parser.add_argument('--r_channel', type=str, nargs='+',
                        required=True, help='List of FITS files for red channel.')
    parser.add_argument('--object_name', type=str, default='myobject',
                        help='Name of the OBJECT to select frames.')
    parser.add_argument('--stretch', type=float, default=1.0,
                        help='Stretch factor for contrast enhancement.')
    parser.add_argument('--clip', type=float, default=0.01,
                        help='Clipping factor for contrast enhancement.')
    parser.add_argument('--low_perc', type=float, default=5.0,
                        help='Low percentile for scaling the image.')
    parser.add_argument('--up_perc', type=float, default=99.0,
                        help='High percentile for scaling the image.')
    parser.add_argument('--smooth_factor', type=float, default=20.0,
                        help='Smoothing factor for the coloured image.')
    parser.add_argument('--save_output', action='store_true',
                        help='Save the output coloured image to file.')
    parser.add_argument('--degrade_psf', action='store_true',
                        help='Degrade images to the worst PSF before combining.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output.')
    return parser.parse_args()


class ColouredImageMaker:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir if args.output_dir else args.input_dir
        self.b_channel_files = args.b_channel
        self.g_channel_files = args.g_channel
        self.r_channel_files = args.r_channel
        self.object_name = args.object_name
        self.stretch = args.stretch
        self.clip = args.clip
        self.low_perc = args.low_perc
        self.up_perc = args.up_perc
        self.smooth_factor = args.smooth_factor
        self.save_output = args.save_output
        self.degrade_psf = args.degrade_psf
        self.verbose = args.verbose

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def select_images_per_channel(self, channel_files):
        selected_files = []
        for file in channel_files:
            full_path = os.path.join(self.input_dir, file)
            if os.path.isfile(full_path):
                selected_files.append(full_path)
            else:
                if self.verbose:
                    print(
                        f"File {full_path} does not exist and will be skipped.")
        return selected_files

    def reproject_into_reference_frame(self, files_per_channel, ref_band='R'):
        all_files = [f for files in files_per_channel.values() for f in files]
        if not all_files:
            raise ValueError("No valid files provided for reprojection.")
        reference_img = [
            f for f in all_files if ref_band in os.path.basename(f)]
        ref_header = fits.getheader(reference_img[0])
        for files in all_files:
            if files not in reference_img:
                output_path = os.path.join(self.output_dir, os.path.basename(
                    files).replace('.fits', '_reproj.fits'))
                if not os.path.isfile(output_path):
                    with fits.open(files) as hdu:
                        data, _ = reproject_interp(hdu, ref_header)
                        # chance nans to zeros
                        data = np.nan_to_num(data, nan=0.0)
                        hdu[0].data = data
                        # update header to reflect reprojection
                        hdu[0].header.update(ref_header)
                        hdu.writeto(output_path, overwrite=True)
                else:
                    if self.verbose:
                        print(
                            f"Reprojected file {output_path} already exists, skipping.")
        hdul = fits.open(reference_img[0])
        path_ref = os.path.join(self.output_dir, os.path.basename(
            reference_img[0]).replace('.fits', '_reproj.fits'))
        hdul.writeto(path_ref, overwrite=True)

    def coadd_imgs_per_channel(self, files_per_channel):
        hduls_list = {'B': [], 'G': [], 'R': []}
        all_files = [f for files in files_per_channel.values() for f in files]
        reference_img = [
            f for f in all_files if 'R-Bessel' in os.path.basename(f)]
        ref_header = fits.getheader(reference_img[0])
        for band, files in files_per_channel.items():
            if not files:
                print(f"No files for band {band}, skipping coaddition.")
                raise ValueError(f"No files for band {band}.")
            if len(files) == 1:
                print('No coadd needed, only one image found for band', band)
                reproj_path = os.path.join(self.output_dir, os.path.basename(
                    files[0]).replace('.fits', '_reproj.fits'))
                if os.path.isfile(reproj_path):
                    hduls_list[band].append(fits.open(reproj_path)[0])
                else:
                    with fits.open(files[0]) as hdul:
                        data = hdul[0].data
                        wcs = WCS(hdul[0].header)
                        data, _ = reproject_interp(
                            (data, wcs), WCS(ref_header), shape_out=data.shape)
                        hdu = fits.PrimaryHDU(data, header=ref_header)
                        hdu.writeto(reproj_path, overwrite=True)

                    hduls_list[band].append(fits.open(reproj_path)[0])
            else:
                wcs_ref = WCS(ref_header)
                data_stack = []
                for f in files:
                    if '_reproj' not in os.path.basename(f):
                        with fits.open(f) as hdul:
                            data = hdul[0].data
                            wcs = WCS(hdul[0].header)
                            data, _ = reproject_interp(
                                (data, wcs), wcs_ref, shape_out=data.shape)
                            data_stack.append(data)
                    else:
                        with fits.open(f) as hdul:
                            data = hdul[0].data
                            data_stack.append(data)
                data_stack = np.array(data_stack)
                weighted_data = np.nanmedian(data_stack, axis=0)
                hdu = fits.PrimaryHDU(weighted_data, header=ref_header)
                hduls_list[band].append(hdu)

        return hduls_list

    def ajust_image(self, hduls_per_channel):
        b_data = hduls_per_channel['B'][0].data
        g_data = hduls_per_channel['G'][0].data
        r_data = hduls_per_channel['R'][0].data

        # def enhance_contrast(data, stretch=0.1, clip=0.01):
        #     low, high = np.percentile(data, (clip, 100 - clip))
        #     data = np.clip(data, low, high)
        #     data = (data - low) / (high - low)
        #     data = np.power(data, stretch)
        #     return data
        #
        # b_data = enhance_contrast(b_data, self.stretch, self.clip)
        # g_data = enhance_contrast(g_data, self.stretch, self.clip)
        # r_data = enhance_contrast(r_data, self.stretch, self.clip)

        # degrade all images to the worst PSF (assumed to be the one with the highest FWHM)
        if self.degrade_psf:
            fwhm_values = {'B': 0, 'G': 0, 'R': 0}
            img_shape = b_data.shape
            if img_shape[0] % 2 == 0:
                img_shape = (img_shape[0] - 1, img_shape[1])
            if img_shape[1] % 2 == 0:
                img_shape = (img_shape[0], img_shape[1] - 1)

            for band, hdul in hduls_per_channel.items():
                finder = DAOStarFinder(
                    fwhm=2.0, threshold=5.*np.std(hdul[0].data))
                sources = finder(hdul[0].data - np.median(hdul[0].data))
                if sources is not None and len(sources) > 0:
                    xypos = list(
                        zip(sources['xcentroid'], sources['ycentroid']))
                    fwhm = fit_fwhm(hdul[0].data, xypos=xypos,
                                    fwhm=2.0, fit_shape=img_shape)
                    fwhm_values[band] = np.nanmedian(fwhm)
                else:
                    fwhm_values[band] = np.nan

        return b_data, g_data, r_data

    def plot_and_save(self, b_data, g_data, r_data, hdul):
        """Plot and save the coloured image for the correspondent colour at each channel."""
        lupton_path = os.path.join(
            self.output_dir, self.object_name.upper() + '_coloured.png')
        print("Saving coloured image to", lupton_path)
        b_data = b_data / np.nanmedian(b_data)
        g_data = g_data / np.nanmedian(g_data)
        r_data = r_data / np.nanmedian(r_data)
        low_val, up_val = np.nanpercentile(
            np.concatenate([r_data, g_data, b_data]), [self.low_perc, self.up_perc])
        stretch_val = up_val - low_val
        print("Using low, stretch values:", low_val, stretch_val)

        rgb_image = make_lupton_rgb(r_data,
                                    g_data,
                                    b_data,
                                    minimum=low_val,
                                    stretch=stretch_val,
                                    Q=self.smooth_factor,
                                    filename=lupton_path if self.save_output else None)

        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection=WCS(hdul[0].header))
        ax.imshow(rgb_image, origin='lower')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        plt.grid(color='white', ls='dotted')

        plt.tight_layout()

        if self.save_output:
            plt.savefig(lupton_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def process(self):
        b_files = self.select_images_per_channel(self.b_channel_files)
        g_files = self.select_images_per_channel(self.g_channel_files)
        r_files = self.select_images_per_channel(self.r_channel_files)
        save_config = open(os.path.join(
            self.output_dir, f'{self.object_name}_config.txt'), 'w')
        save_config.write(f"Object: {self.object_name}\n")
        save_config.write(f"Blue channel files: {b_files}\n")
        save_config.write(f"Green channel files: {g_files}\n")
        save_config.write(f"Red channel files: {r_files}\n")
        save_config.close()

        files_per_channel = {
            'B': b_files,
            'G': g_files,
            'R': r_files
        }

        self.reproject_into_reference_frame(
            files_per_channel, ref_band='R-Bessel')
        hduls_per_channel = self.coadd_imgs_per_channel(files_per_channel)
        b_data, g_data, r_data = self.ajust_image(hduls_per_channel)
        path_reproj_ref = os.path.join(self.output_dir, os.path.basename(
            r_files[0]).replace('.fits', '_reproj.fits'))
        hdul = fits.open(path_reproj_ref)
        self.plot_and_save(b_data, g_data, r_data, hdul)

        if self.verbose:
            print(f"Coloured image saved to {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    maker = ColouredImageMaker(args)
    maker.process()
