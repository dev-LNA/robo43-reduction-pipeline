#!/bin/python3

import os
import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.astrometry_net import AstrometryNet
from photutils.detection import DAOStarFinder
from multiprocessing import Pool
import matplotlib.pyplot as plt
import glob
import argparse
import logging


def setup_logging(verbose=False, logfile=None, loglevel=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(loglevel if verbose else logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
        description='Process Robo-43 frames to create a master flat field.')
    parser.add_argument('input_dir', type=str,
                        help='Directory containing input FITS files.')
    parser.add_argument('--output_file', type=str,
                        help='Output FITS file for the master flat field.')
    parser.add_argument('--masterbias', type=str, default='master_bias.fits',
                        help='Path to the master bias FITS file.')
    parser.add_argument('--masterdark', type=str, default='master_dark.fits',
                        help='Path to the master dark FITS file.')
    parser.add_argument('--masterflat', type=str, default='master_flat.fits',
                        help='Path to the master flat FITS file.')
    parser.add_argument('--save_processed', action='store_true',
                        help='Save processed frames to disk.')
    parser.add_argument('--np', type=int, default=4,
                        help='Number of processes for parallel processing.')
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite existing output file if it exists.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')
    parser.add_argument('--logfile', type=str,
                        help='File to log output messages.')

    return parser.parse_args()


class ProcessFrame:
    def __init__(self, args):
        self.workdir = args.input_dir
        self.output_dir = args.output_file if args.output_file else self.workdir
        self.masterbias = args.masterbias
        self.masterdark = args.masterdark
        self.masterflat = args.masterflat
        self.save_processed = args.save_processed
        self.np = args.np
        self.clobber = args.clobber
        self.verbose = args.verbose
        self.logfile = args.logfile
        self.logger = setup_logging(self.verbose, self.logfile)
        self.logger.info(
            'Initialized ProcessFrame with workdir: %s', self.workdir)

        self.object_names_to_guess = {
            'eta car': ['etacar', 'eta carinae', 'etaCarNebula'],
        }

    def get_fits_files(self):
        fits_files = glob.glob(os.path.join(self.workdir, '*.fits'))
        self.logger.info('Found %d FITS files in %s',
                         len(fits_files), self.workdir)
        exclusion_indexes = []
        for index, f in enumerate(fits_files):
            self.logger.debug('FITS file: %s', f)
            proc_file_name = os.path.join(
                self.output_dir, os.path.basename(f).replace('.fits', '_proc.fits'))
            if os.path.exists(proc_file_name) and not self.clobber:
                self.logger.info(
                    'Some processed files already exist. Use --clobber to overwrite.')
                sys.exit(1)
            # selecting only science frames
            header = fits.getheader(f)
            if 'flat' in header.get('OBJECT', '').lower():
                self.logger.debug('Selected science frame: %s', f)
                exclusion_indexes.append(index)
            elif 'dark' in header.get('OBJECT', '').lower():
                self.logger.debug('Excluded dark frame: %s', f)
                exclusion_indexes.append(index)
            elif 'bias' in header.get('OBJECT', '').lower():
                self.logger.debug('Excluded bias frame: %s', f)
                exclusion_indexes.append(index)

        fits_files = sorted([f for i, f in enumerate(fits_files)
                             if i not in exclusion_indexes])

        return fits_files

    def bias_subtraction(self, hdul):
        """Subtract master bias from science frame."""
        path_to_bias = os.path.join(self.workdir, self.masterbias)
        if os.path.exists(path_to_bias):
            master_bias = fits.getdata(path_to_bias)
            self.logger.debug('Master bias shape: %s', master_bias.shape)
            bias_subtracted = hdul[0].data - master_bias
            hdul[0].header['HISTORY'] = 'Bias subtracted'
            hdul[0].data = bias_subtracted
            return hdul
        else:
            self.logger.critical(
                'No master bias file found at %s', path_to_bias)
            raise FileNotFoundError(
                f'Master bias file not found: {path_to_bias}')

    def flat_correction(self, hdul):
        """Apply flat field correction to science frame."""
        # get image filter
        image_filter = hdul[0].header.get('FILTER', 'unknown')
        path_to_flat = os.path.join(self.workdir, self.masterflat)
        if not os.path.exists(path_to_flat):
            self.logger.warning(
                'Default master flat not found. Trying to guess...')
            path_to_flat = os.path.join(
                self.workdir, f'master_flat_{image_filter}.fits')
            if not os.path.exists(path_to_flat):
                self.logger.critical(
                    'No master flat file found at %s', path_to_flat)
                raise FileNotFoundError(
                    f'Master flat file not found: {path_to_flat}')
        else:
            self.logger.info('Using default master flat: %s', path_to_flat)

        master_flat = fits.getdata(path_to_flat)
        self.logger.debug('Master flat shape: %s', master_flat.shape)

        flat_corrected_data = hdul[0].data / master_flat
        hdul[0].header['HISTORY'] = 'Flat field corrected'
        hdul[0].data = flat_corrected_data
        return hdul

    def guess_ra_dec(self, proc_path):
        """Guess to get RA and DEC if not present using OBJECT."""
        hdul = fits.open(proc_path, mode='update')
        if 'RA' in hdul[0].header and 'DEC' in hdul[0].header:
            self.logger.debug('RA and DEC already present in header.')
            return hdul

        object_name = hdul[0].header.get('OBJECT', '').strip()
        for key, aliases in self.object_names_to_guess.items():
            if object_name.lower() in aliases:
                object_name = key
                break

        if object_name:
            self.logger.info(
                'Attempting to resolve OBJECT name: %s', object_name)
            try:
                result = Simbad.query_object(object_name)
                # update header with RA and DEC
                hdul[0].header['RA'] = (
                    result['ra'][0], 'Right Ascension from SIMBAD')
                hdul[0].header['DEC'] = (
                    result['dec'][0], 'Declination from SIMBAD')
                self.logger.info('Resolved RA: %s, DEC: %s', ra, dec)
            except Exception as e:
                self.logger.error(
                    'Error querying SIMBAD: %s. Falling back to astrometry.net.', str(e))
        else:
            self.logger.warning('No OBJECT name found in header.')

        hdul.flush()
        return hdul

    def solve_astrometry(self, path_to_fits):
        self.logger.info(
            'RA and DEC not found in header. Querying astrometry.net...')
        # detect photometric sources
        hdul = fits.open(path_to_fits, mode='update')
        daofind = DAOStarFinder(fwhm=3.0, threshold=5.*np.std(hdul[0].data))
        sources = daofind(hdul[0].data - np.median(hdul[0].data))
        sorted_sources = sources[np.argsort(sources['flux'])[::-1]]

        ast = AstrometryNet()
        img_width = hdul[0].header.get('NAXIS1', hdul[0].data.shape[1])
        img_height = hdul[0].header.get('NAXIS2', hdul[0].data.shape[0])
        wcs_header = ast.solve_from_source_list(
            sorted_sources['xcentroid'], sorted_sources['ycentroid'],
            img_width, img_height, solve_timeout=120)

        # NOTE: Alternative method using image upload does not work.
        # Connection errors occur frequently and all attempts failed.
        # wcs_header = ast.solve_from_image(path_to_fits,
        #                                   # force_image_upload=True,
        #                                   ra_key='RA',
        #                                   dec_key='DEC',
        #                                   ra_dec_units=('deg', 'deg'),
        #                                   fwhm=3.0,
        # detect_threshold = 2)
        if wcs_header:
            wcs = WCS(wcs_header)
            hdul[0].header.update(wcs.to_header())
            self.logger.info('Astrometry solved and WCS updated in header.')
        else:
            self.logger.error('Astrometry solving failed.')
        hdul.flush()
        return hdul, sources

    def plot_frame(self, image, file_name):
        """Plot a single frame for visual inspection."""
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray', origin='lower', vmin=np.percentile(image, 5),
                   vmax=np.percentile(image, 95))
        plt.title(os.path.basename(file_name).replace('.fits', ''))
        plt.colorbar()

        if self.save_processed:
            png_file_name = file_name.replace('.fits', '_proc.png')
            plt.savefig(png_file_name)
            self.logger.info('Saved plot to: %s', png_file_name)
        else:
            plt.show()

    def process_frame(self, fits_file):
        """Process a single FITS frame."""
        self.logger.info('Processing frame: %s', fits_file)
        proc_path = os.path.join(
            self.output_dir, os.path.basename(fits_file).replace('.fits', '_proc.fits'))
        if os.path.exists(proc_path) and not self.clobber:
            self.logger.info(
                'Processed file already exists: %s. Skipping.', proc_path)
            return

        with fits.open(fits_file) as hdul:
            # bias subtraction
            processed_data = self.bias_subtraction(hdul)
            self.plot_frame(processed_data[0].data, fits_file)
            # flat correction
            processed_data = self.flat_correction(processed_data)
            self.plot_frame(processed_data[0].data, fits_file)

        if self.save_processed:
            proc_file_name = os.path.join(
                self.output_dir, os.path.basename(fits_file).replace('.fits', '_proc.fits'))
            processed_data.writeto(proc_file_name, overwrite=self.clobber)
            self.logger.info('Saved processed frame to: %s', proc_file_name)

        if 'RA' not in processed_data[0].header or 'DEC' not in processed_data[0].header:
            self.logger.info('RA/DEC missing, attempting to guess.')
            processed_data = self.guess_ra_dec(proc_path)

        if 'RA' in processed_data[0].header and 'DEC' in processed_data[0].header:
            self.logger.info(
                'RA and DEC found in header: RA=%s, DEC=%s',
                processed_data[0].header['RA'], processed_data[0].header['DEC'])
            processed_data = self.solve_astrometry(proc_path)

        import pdb
        pdb.set_trace()

    def main(self):
        self.logger.info('Starting processing of frames.')
        fits_files = self.get_fits_files()
        # Process only the first frame for now
        self.process_frame(fits_files[0])
        import pdb
        pdb.set_trace()


if __name__ == '__main__':
    args = parse_arguments()
    processor = ProcessFrame(args)
    processor.main()
