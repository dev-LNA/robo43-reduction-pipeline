#!/bin/python3

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from reproject import reproject_interp
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian2DKernel
import glob
import argparse
import gc


def parse_args():
    parser = argparse.ArgumentParser(description="Coadd astronomical frames.")
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing input FITS files.")
    parser.add_argument("--object_name", type=str, required=True,
                        help="Name of the OBJECT to select frames do coadd.")
    parser.add_argument("--cutout", nargs=4, type=float, metavar=('RA', 'DEC', 'SIZE_X', 'SIZE_Y'),
                        help="Cutout region specified by RA, DEC (degrees) and size in pixels (SIZE_X, SIZE_Y).")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian smoothing sigma.")
    parser.add_argument("--save_plot", action='store_true',
                        help="Plot and save the final coadded image.")
    parser.add_argument("--show", action='store_true',
                        help="Show the plot interactively.")
    parser.add_argument("--clobber", action='store_true',
                        help="Overwrite existing files.")
    parser.add_argument("--runtest", action='store_true',
                        help="Run in test mode with predefined parameters.")
    return parser.parse_args()


class CoaddFrames:
    def __init__(self, args):
        self.workdir = args.input_dir
        self.cutout_params = args.cutout
        self.smooth_sigma = args.smooth
        self.object_name = args.object_name
        self.save_plot = args.save_plot
        self.show_plot = args.show
        self.clobber = args.clobber
        self.runtest = args.runtest
        self.frames = []
        self.wcs_list = []
        self.data_list = []

    def select_frames(self):
        selected_files = []
        files_list = glob.glob(os.path.join(self.workdir, "*_proc.fits"))
        for f in files_list:
            if os.path.isfile(f.replace(".fits", "_astro.png")):
                if self.object_name == fits.getheader(f).get('OBJECT', None).lower():
                    selected_files.append(f)

        if not selected_files:
            print(
                f"No frames found for object '{self.object_name}' in directory '{self.workdir}'.")
            sys.exit(1)
        self.frames = selected_files

    def gatter_frames_per_filter(self):
        filter_dict = {}
        for f in self.frames:
            hdr = fits.getheader(f)
            filt = hdr.get('FILTER', 'UNKNOWN')
            if filt not in filter_dict:
                filter_dict[filt] = []
            filter_dict[filt].append(f)
        return filter_dict

    def coadd_images(self, files):
        header0 = fits.getheader(files[0])
        wcs0 = WCS(header0)
        data_stack = [fits.getdata(files[0])]
        exptimes = []
        # files = files[:5]
        i = 1
        for f in files[1:]:
            with fits.open(f) as hdul:
                data = hdul[0].data
                wcs = WCS(hdul[0].header)
                exptime = hdul[0].header.get('EXPTIME', 1.0)
                exptimes.append(exptime)
                data, _ = reproject_interp(
                    (data, wcs), wcs0, shape_out=data_stack[0].shape)
                print(f"Reprojected {f} %i of %i frames" % (i, len(files) - 1))
                data_stack.append(data)
                i += 1

        # as there are different exposure times, we do weighted average
        weights = [1/np.nanstd(d)**2 if np.nanstd(d) >
                   0 else 0 for d in data_stack]
        weights = np.array(weights)
        weights /= np.nanmax(weights)
        coadded_data = np.nanmedian(np.array(data_stack).T *
                                    weights, axis=2).T

        if self.smooth_sigma > 0:
            coadded_data = gaussian_filter(
                coadded_data, sigma=self.smooth_sigma)

        header0['NCOMBINE'] = (len(files), 'Number of combined frames')
        header0['TEXPTIME'] = (np.sum(exptimes), 'Total exposure time')
        header0['COMMENT'] = "Coadded using coadd_frames.py"
        hdul = fits.PrimaryHDU(
            coadded_data, header=header0)

        return hdul

    def plot_image(self, hdul):
        wcs = WCS(hdul[0].header)
        data = hdul[0].data
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(projection=wcs)
        ax.imshow(data, origin='lower', cmap='gray', vmin=np.nanpercentile(
            data, 5), vmax=np.nanpercentile(data, 95))
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        title = f"Coadded of {self.object_name}"
        title += f"\n(N = {hdul[0].header.get('NCOMBINE', 'N/A')},"
        title += F"TEXPTIME={hdul[0].header.get('TEXPTIME', 'N/A')}s)"
        if self.save_plot:
            plot_filename = os.path.join(
                self.workdir,
                f"{self.object_name.upper()}_" +
                f"{hdul[0].header.get('FILTER', 'UNKNOWN')}_coadded.png")
            plt.savefig(plot_filename)
            print(f"Saved plot to '{plot_filename}'")
            if self.show_plot:
                plt.show()
            else:
                plt.close()
        else:
            plt.show()

    def process(self):
        self.select_frames()
        filter_dict = self.gatter_frames_per_filter()
        for filt, files in filter_dict.items():
            if self.runtest:
                files = files[:3]
            print(f"Coadding {len(files)} frames for filter '{filt}'...")
            output_filename = os.path.join(
                self.workdir, f"{self.object_name.upper()}_{filt}_coadded.fits")
            if os.path.isfile(output_filename) and not args.clobber:
                print(
                    f"File '{output_filename}' already exists. Use --clobber to overwrite.")
            else:
                coadded_hdu = self.coadd_images(files)
                coadded_hdu.writeto(output_filename, overwrite=True)
                print(f"Saved coadded image to '{output_filename}'")
                gc.collect()
            if self.save_plot or self.show_plot:
                coadded_hdu = fits.open(output_filename)
                self.plot_image(coadded_hdu)


if __name__ == "__main__":
    args = parse_args()
    coadder = CoaddFrames(args)
    coadder.process()
