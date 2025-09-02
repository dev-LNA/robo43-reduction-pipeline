#!/bin/python3

import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.simbad import Simbad
from astroquery.astrometry_net import AstrometryNet
from photutils.detection import DAOStarFinder
import astrometry
from multiprocessing import Pool
import matplotlib.pyplot as plt
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
    parser.add_argument('--np', type=int, default=1,
                        help='Number of processes for parallel processing.')
    parser.add_argument('--solve_astrometry', action='store_true',
                        help='Attempt to solve astrometry for processed frames.')
    parser.add_argument('--sigma_clip', type=float, default=3.0,
                        help='Sigma clipping value for source detection.')
    parser.add_argument('--object', type=str, default=None,
                        help='Name of the target object to select images to process.')
    parser.add_argument('--runtest', action='store_true',
                        help='Run in test mode with limited files.')
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
        self.solve_astrometry = args.solve_astrometry
        self.sigma_clip = args.sigma_clip
        self.object_name = args.object
        self.runtest = args.runtest
        self.clobber = args.clobber
        self.verbose = args.verbose
        self.logfile = args.logfile
        self.logger = setup_logging(self.verbose, self.logfile)
        self.logger.info(
            'Initialized ProcessFrame with workdir: %s', self.workdir)

        self.object_names_to_guess = {
            'eta car': ['etacar', 'eta carinae', 'etaCarNebula', 'etacarnebula'],
        }
        self.proc_status = {}

    def get_fits_files(self):
        fits_files = glob.glob(os.path.join(self.workdir, '*.fits'))
        self.logger.info('Found %d FITS files in %s',
                         len(fits_files), self.workdir)
        exclusion_indexes = []
        for index, f in enumerate(fits_files):
            self.logger.debug('FITS file: %s', f)
            proc_file_name = os.path.join(
                self.output_dir, os.path.basename(f).replace('.fits', '_proc.fits'))
            if os.path.exists(proc_file_name):
                if self.clobber:
                    if '_proc' in f:
                        exclusion_indexes.append(index)
                        self.logger.debug(
                            'Will overwrite existing processed file: %s', proc_file_name)
                else:
                    if self.solve_astrometry:
                        self.logger.info(
                            'Processed file exists, but will attempt astrometry solving: %s', proc_file_name)
                    else:
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
                             if i not in exclusion_indexes and '_proc' not in f])
        if self.object_name:
            fits_files = [f for f in fits_files if self.object_name.lower(
            ) == fits.getheader(f).get('OBJECT', '').lower()]
            self.logger.info(
                'Filtered files for object "%s", %d files remain.', self.object_name, len(fits_files))

        self.logger.info(
            'Selected %d science FITS files for processing.', len(fits_files))

        return fits_files

    def bias_subtraction(self, hdul, raw_file):
        """Subtract master bias from science frame."""
        path_to_bias = os.path.join(self.workdir, self.masterbias)
        raw_name = os.path.basename(raw_file)
        if os.path.exists(path_to_bias):
            try:
                master_bias = fits.getdata(path_to_bias)
                self.logger.debug('Master bias shape: %s', master_bias.shape)
                bias_subtracted = hdul[0].data - master_bias
                hdul[0].header['HISTORY'] = 'Bias subtracted'
                hdul[0].data = bias_subtracted
                self.proc_status[raw_name]['proc_status'] = 'Bias subtraction successful'
                self.proc_status[raw_name]['proc_code'] = 1
                return hdul
            except Exception as e:
                self.logger.error(
                    'Error reading master bias file: %s', str(e))
                self.proc_status[raw_name]['proc_status'] = 'Bias subtraction failed'
                raise e from e
        else:
            self.logger.critical(
                'No master bias file found at %s', path_to_bias)
            self.proc_status[raw_name]['proc_status'] = 'Bias subtraction failed'
            raise FileNotFoundError(
                f'Master bias file not found: {path_to_bias}')

    def flat_correction(self, hdul, raw_file):
        """Apply flat field correction to science frame."""
        # get image filter
        image_filter = hdul[0].header.get('FILTER', 'unknown')
        path_to_flat = os.path.join(self.workdir, self.masterflat)
        raw_name = os.path.basename(raw_file)
        if not os.path.exists(path_to_flat):
            self.logger.warning(
                'Default master flat not found. Trying to guess...')
            path_to_flat = os.path.join(
                self.workdir, f'master_flat_{image_filter}.fits')
            if not os.path.exists(path_to_flat):
                self.logger.critical(
                    'No master flat file found at %s', path_to_flat)
                self.proc_status[raw_name]['proc_status'] = 'Flat correction failed'
                raise FileNotFoundError(
                    f'Master flat file not found: {path_to_flat}')
            else:
                self.logger.info(
                    'Using guessed master flat for filter %s: %s', image_filter, path_to_flat)
        else:
            self.logger.info('Using default master flat: %s', path_to_flat)

        try:
            master_flat = fits.getdata(path_to_flat)
            self.logger.debug('Master flat shape: %s', master_flat.shape)
        except Exception as e:
            self.logger.error(
                'Error reading master flat file: %s', str(e))
            self.proc_status[raw_name]['proc_status'] = 'Flat correction failed'
            raise e from e

        try:
            flat_corrected_data = hdul[0].data / master_flat
            hdul[0].header['HISTORY'] = 'Flat field corrected'
            hdul[0].data = flat_corrected_data
            self.proc_status[raw_name]['proc_status'] = 'Flat correction successful'
            self.proc_status[raw_name]['proc_code'] = 3
            return hdul
        except Exception as e:
            self.logger.error('Error during flat correction: %s', str(e))
            self.proc_status[raw_name]['proc_status'] = 'Flat correction failed'
            return hdul

    def guess_ra_dec(self, proc_path, raw_path):
        """Guess to get RA and DEC if not present using OBJECT."""
        if self.proc_status[os.path.basename(raw_path)]['proc_code'] != 3:
            self.logger.warning(
                'Previous processing steps failed. Skipping RA/DEC guessing.')
            return
        raw_name = os.path.basename(raw_path)

        hdul = fits.open(proc_path, mode='update')
        if 'RA' in hdul[0].header and 'DEC' in hdul[0].header:
            self.logger.debug('RA and DEC already present in header.')
            self.proc_status[raw_name]['RA'] = hdul[0].header['RA']
            self.proc_status[raw_name]['DEC'] = hdul[0].header['DEC']
            self.proc_status[raw_name]['proc_status'] = 'RA/DEC already present'
            self.proc_status[raw_name]['proc_code'] = 7
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
                self.logger.info('Resolved RA: %s, DEC: %s',
                                 result['ra'][0], result['dec'][0])
                self.proc_status[raw_name]['RA'] = hdul[0].header['RA']
                self.proc_status[raw_name]['DEC'] = hdul[0].header['DEC']
                self.proc_status[raw_name]['proc_status'] = 'RA/DEC guessed from SIMBAD'
                self.proc_status[raw_name]['proc_code'] = 7
            except Exception as e:
                self.logger.error(
                    'Error querying SIMBAD: %s.', str(e))
                self.proc_status[raw_name]['proc_status'] = 'RA/DEC guessing failed'
                self.proc_status[raw_name]['proc_code'] = 11
        else:
            self.logger.warning('No OBJECT name found in header.')
            self.proc_status[raw_name]['proc_status'] = 'RA/DEC guessing failed'
            self.proc_status[raw_name]['proc_code'] = 11

        hdul.flush()
        return hdul

    def run_astrometry_solver1(self, sorted_sources, hdul, raw_name):
        with astrometry.Solver(
            astrometry.series_5200.index_files(
                cache_directory="/home/herpich/Documents/.astrometry", scales={6})
        ) as solver:
            stars = [(st['xcentroid'], st['ycentroid'])
                     for st in sorted_sources]
            solution = solver.solve(stars=stars,
                                    size_hint=astrometry.SizeHint(
                                        lower_arcsec_per_pixel=0.2,
                                        upper_arcsec_per_pixel=1.0),
                                    position_hint=astrometry.PositionHint(
                                        ra_deg=hdul[0].header.get('RA', None),
                                        dec_deg=hdul[0].header.get(
                                            'DEC', None),
                                        radius_deg=1.0),
                                    solution_parameters=astrometry.SolutionParameters())
            if solution.has_match():
                wcs_header = solution.best_match().astropy_wcs()
                stars_used = {'ra': [obj.ra_deg for obj in solution.best_match().stars],
                              'dec': [obj.dec_deg for obj in solution.best_match().stars]}
                self.logger.info(
                    'Astrometry solved using local index files.')
                self.proc_status[raw_name]['proc_status'] = 'Astrometry solved'
                self.proc_status[raw_name]['proc_code'] = 59
                return wcs_header, stars_used

    def run_astrometry_solver2(self, sorted_sources, hdul, raw_name):
        ast = AstrometryNet()
        img_width = hdul[0].header.get('NAXIS1', hdul[0].data.shape[1])
        img_height = hdul[0].header.get('NAXIS2', hdul[0].data.shape[0])
        if self.proc_status[raw_name]['proc_code'] == 7:
            ast.ra = hdul[0].header['RA']
            ast.dec = hdul[0].header['DEC']
            ast.radius = 1.0

        try:
            wcs_header = ast.solve_from_source_list(
                sorted_sources['xcentroid'], sorted_sources['ycentroid'],
                img_width, img_height, solve_timeout=120)
            self.logger.info('Astrometry solving completed.')
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solved'
            self.proc_status[raw_name]['proc_code'] = 59
        except Exception as e:
            self.logger.error('Astrometry solving failed: %s', str(e))
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
            self.proc_status[raw_name]['proc_code'] = 27
            return None, e
        return wcs_header, sorted_sources

    def run_daofinder(self, hdul):
        try:
            daofind = DAOStarFinder(
                fwhm=3.0, threshold=self.sigma_clip * np.std(hdul[0].data))
            sources = daofind(hdul[0].data - np.median(hdul[0].data))
            sorted_sources = sources[np.argsort(sources['flux'])[::-1]]
            self.logger.info(
                'Detected %d sources in the image.', len(sorted_sources))
            return sorted_sources
        except Exception as e:
            self.logger.error('Error detecting sources: %s', str(e))
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
            self.proc_status[raw_name]['proc_code'] = 27
            return e

    def solver_astrometry(self, path_to_fits, raw_path):
        self.logger.info(
            'Attempting astrometry solving for: %s', path_to_fits)
        if self.proc_status[os.path.basename(raw_path)]['proc_code'] not in [3, 7]:
            self.logger.warning(
                'Previous processing steps failed. Skipping astrometry solving.')
            return
        raw_name = os.path.basename(raw_path)

        # detect photometric sources
        try:
            hdul = fits.open(path_to_fits, mode='update')
        except Exception as e:
            self.logger.error('Error opening FITS file: %s', str(e))
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
            self.proc_status[raw_name]['proc_code'] = 27
            return

        sorted_sources = self.run_daofinder(hdul)
        if sorted_sources is None:
            self.logger.error('No sources detected, cannot solve astrometry.')
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
            self.proc_status[raw_name]['proc_code'] = 27
            return

        solver_used = 0
        astrometry_solved = False
        try:
            wcs_header, stars_used = self.run_astrometry_solver1(
                sorted_sources, hdul, raw_name)
            astrometry_solved = True
            solver_used = 1
            self.logger.info('Astrometry solving completed.')
        except Exception as e:
            self.logger.warning(
                'Local astrometry solving failed: %s. Trying Astrometry.net service.', str(e))

        if not astrometry_solved:
            wcs_header, stars_used = self.run_astrometry_solver2(
                sorted_sources, hdul, raw_name)
            solver_used = 2

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
            try:
                if solver_used == 1:
                    hdul[0].header.update(wcs_header.to_header())
                    self.logger.info(
                        'Astrometry solved and WCS updated in header.')
                    self.proc_status[raw_name]['proc_status'] = 'Astrometry solved and WCS updated'
                    self.proc_status[raw_name]['proc_code'] = 123
                elif solver_used == 2:
                    wcs = WCS(wcs_header)
                    hdul[0].header.update(wcs.to_header())
                    self.logger.info(
                        'Astrometry solved and WCS updated in header.')
                    self.proc_status[raw_name]['proc_status'] = 'Astrometry solved and WCS updated'
                    self.proc_status[raw_name]['proc_code'] = 123
            except Exception as e:
                self.logger.error(
                    'Error updating WCS in header: %s', str(e))
                self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
                self.proc_status[raw_name]['proc_code'] = 59
                return
        else:
            self.logger.error('Astrometry solving failed.')
            self.proc_status[raw_name]['proc_status'] = 'Astrometry solving failed'
            self.proc_status[raw_name]['proc_code'] = 59
            return

        hdul.flush()

        if solver_used == 1:
            sources = wcs_header.all_world2pix(
                stars_used['ra'], stars_used['dec'], 0)
            sources = [{'xcentroid': src[0], 'ycentroid': src[1]}
                       for src in zip(sources[0], sources[1])]
        elif solver_used == 2:
            sources = sorted_sources
        else:
            self.logger.error('No solver was successful.')
            return
        self.plot_frame(hdul[0].data, path_to_fits,
                        sources=sources, show=True)

        return

    def plot_frame(self, image, file_name, sources=None, show=False):
        """Plot a single frame for visual inspection."""
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray', origin='lower', vmin=np.percentile(image, 5),
                   vmax=np.percentile(image, 95))
        title = os.path.basename(file_name).replace('_proc.fits', '')
        plt.colorbar()

        if sources is not None and len(sources) > 0:
            for source in sources:
                circ = plt.Circle((source['xcentroid'], source['ycentroid']),
                                  radius=30, color='green', fill=False, lw=3)
                plt.gca().add_patch(circ)
            png_file_name = file_name.replace('.fits', '_astro.png')
            title += f' {len(sources)} sources used'
        else:
            png_file_name = file_name.replace('.fits', '_proc.png')

        plt.title(title)
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.tight_layout()
        if self.save_processed:
            plt.savefig(png_file_name)
            self.logger.info('Saved plot to: %s', png_file_name)
            if show and self.np == 1:
                plt.show()
            else:
                plt.close()
        else:
            plt.show()

    def fill_proc_status(self,
                         raw_file,
                         Object=None,
                         Filter=None,
                         RA=None,
                         DEC=None,
                         Exptime=None,
                         proc_status='Not processed',
                         proc_code=0,
                         proc_file=None):
        self.proc_status[raw_file] = {'raw_file': raw_file,
                                      'Object': Object,
                                      'Filter': Filter,
                                      'RA': RA,
                                      'DEC': DEC,
                                      'Exptime': Exptime,
                                      'proc_status': proc_status,
                                      'proc_code': proc_code,
                                      'proc_file': proc_file,
                                      }

    def process_frame(self, fits_file):
        """Process a single FITS frame."""
        self.logger.info('Processing frame: %s', fits_file)
        proc_path = os.path.join(
            self.output_dir, os.path.basename(fits_file).replace('.fits', '_proc.fits'))
        if os.path.exists(proc_path) and not self.clobber:
            self.logger.info(
                'Processed file already exists: %s.', proc_path)
            if self.solve_astrometry:
                self.logger.info('Attempting astrometry solving.')
                path_ast_solved = os.path.join(
                    self.output_dir, os.path.basename(proc_path).replace('.fits', '_astro.png'))
                if not os.path.exists(path_ast_solved):
                    proc_header = fits.getheader(proc_path)
                    self.fill_proc_status(
                        raw_file=os.path.basename(fits_file),
                        Object=proc_header.get('OBJECT', 'unknown'),
                        Filter=proc_header.get('FILTER', 'unknown'),
                        Exptime=proc_header.get('EXPTIME', None),
                        RA=proc_header.get('RA', None),
                        DEC=proc_header.get('DEC', None),
                        proc_status='Processed, astrometry not solved',
                        proc_code=3,
                        proc_file=os.path.basename(proc_path),
                    )
                    self.solver_astrometry(proc_path, fits_file)
                else:
                    self.logger.info(
                        'Astrometry already solved, skipping: %s', path_ast_solved)
                    return
            else:
                self.logger.info('Skipping processing as file exists.')
                return
        else:
            self.logger.debug('Will create processed file: %s', proc_path)
            with fits.open(fits_file) as hdul:
                # bias subtraction
                self.fill_proc_status(
                    raw_file=os.path.basename(fits_file),
                    Object=hdul[0].header.get('OBJECT', 'unknown'),
                    Filter=hdul[0].header.get('FILTER', 'unknown'),
                    Exptime=hdul[0].header.get('EXPTIME', None),
                    proc_status='Started processing',
                )
                processed_data = self.bias_subtraction(hdul, fits_file)
                self.plot_frame(processed_data[0].data, fits_file)
                # flat correction
                processed_data = self.flat_correction(
                    processed_data, fits_file)
                self.plot_frame(processed_data[0].data, fits_file)

            if self.save_processed:
                proc_file_name = os.path.join(
                    self.output_dir, os.path.basename(fits_file).replace('.fits', '_proc.fits'))
                processed_data.writeto(proc_file_name, overwrite=self.clobber)
                self.proc_status[os.path.basename(
                    fits_file)]['proc_file'] = os.path.basename(proc_file_name)
                self.logger.info(
                    'Saved processed frame to: %s', proc_file_name)

            if 'RA' not in processed_data[0].header or 'DEC' not in processed_data[0].header:
                self.logger.info('RA/DEC missing, attempting to guess.')
                processed_data = self.guess_ra_dec(proc_path, fits_file)

            if 'RA' in processed_data[0].header and 'DEC' in processed_data[0].header:
                self.logger.info(
                    'RA and DEC found in header: RA=%s, DEC=%s',
                    processed_data[0].header['RA'], processed_data[0].header['DEC'])
                self.solver_astrometry(proc_path, fits_file)

        # save proc_status to CSV for each processed frame
        out_proc_status_file = os.path.join(
            self.output_dir, f'proc_status_{os.path.basename(fits_file).replace(".fits", "")}.csv')
        self.organize_proc_status(output_path_file=out_proc_status_file)
        self.logger.info('Saved processing status to: %s',
                         out_proc_status_file)

    def organize_proc_status(self, output_path_file='processing_status.csv'):
        """Organize processing status into a structured array and save it as a CSV to output_dir."""
        dtype = [('raw_file', 'U100'), ('Object', 'U100'), ('Filter', 'U20'),
                 ('RA', 'U20'), ('DEC', 'U20'), ('Exptime', 'f4'),
                 ('proc_status', 'U200'), ('proc_code', 'i4'), ('proc_file', 'U100')]
        proc_status_array = np.array(
            [tuple(v.values()) for v in self.proc_status.values()], dtype=dtype)
        csv_file = os.path.join(self.output_dir, output_path_file)
        df = pd.DataFrame(proc_status_array)
        df.to_csv(csv_file, index=False)
        self.logger.info('Saved processing status to: %s', csv_file)

    def main(self):
        self.logger.info('Starting processing of frames.')
        fits_files = self.get_fits_files()
        list(fits_files)

        if self.runtest:
            self.process_frame(fits_files[0])
        else:
            if self.np > 1:
                with Pool(processes=self.np) as pool:
                    pool.map(self.process_frame, fits_files)
            else:
                for fits_file in fits_files[:2]:
                    self.process_frame(fits_file)
        self.organize_proc_status()
        self.logger.info('Processing completed.')


if __name__ == '__main__':
    args = parse_arguments()
    processor = ProcessFrame(args)
    processor.main()
