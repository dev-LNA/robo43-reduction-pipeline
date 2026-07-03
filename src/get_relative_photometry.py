#!/bin/python

"""
Extract the photometry of astronomical images based on specific positions
of stars for relative photometry calculations.

Initial development for EtaCar photometry with ROBO43 data.
"""

import os
# from photutils.aperture import ApertureStats
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract photometry of two stars for relative photometry.")
    parser.add_argument('--images_list', type=str, required=True,
                        help='Text file with list of FITS images to process')
    parser.add_argument('--workdir', default='.',
                        help='Working directory for output files (default: current directory)')
    parser.add_argument('--refstar', nargs=2, type=str, required=True,
                        help='RA and Dec of the reference star (e.g. "123.45678 -12.34567")')
    parser.add_argument('--targetstar', nargs=2, type=str, required=True,
                        help='RA and Dec of the target star (e.g. "123.45789 -12.34678")')
    parser.add_argument('--compstar', nargs=2, type=str, required=True,
                        help='RA and Dec of the comparison star (e.g. "123.45890 -12.34789")')
    parser.add_argument('--aperture', type=float, default=15.0,
                        help='Aperture radius in arcsecs (default: 15.0)')
    parser.add_argument('--annulus', nargs=2, type=float, default=[
                        20.0, 30.0], help='Sky annulus inner and outer radius in arcsecs (default: 20.0 30.0)')
    parser.add_argument('--output_csv', default='photometry_results.csv',
                        help='Name of the output CSV file (default: photometry_results.csv)')
    parser.add_argument('--savefig', action='store_true',
                        help='Save the image with apertures and annuli as a PNG file')
    parser.add_argument('--clobber', action='store_true',
                        help='Overwrite existing output files if they exist')
    return parser.parse_args()


def process_image(image_file,
                  refstar,
                  targetstar,
                  compstar,
                  aperture_radius=15.0,
                  annulus_r_in=20.0,
                  annulus_r_out=30.0,
                  savefig=False
                  ):

    refstar_coords = SkyCoord(
        ra=refstar[0], dec=refstar[1], unit=('deg', 'deg'))
    targetstar_coords = SkyCoord(
        ra=targetstar[0], dec=targetstar[1], unit=('deg', 'deg'))
    compstar_coords = SkyCoord(
        ra=compstar[0], dec=compstar[1], unit=('deg', 'deg'))

    # Load image and WCS
    hdul = fits.open(image_file)
    data = hdul[0].data.astype(float)
    header = hdul[0].header
    wcs = WCS(header)
    if wcs.is_celestial:
        print(f"Loaded WCS for {image_file}")
    else:
        print(f"Warning: No valid WCS found in {
              image_file}, using pixel coordinates directly")
        # If no WCS, we would need to define pixel positions manually or skip
        return

    # Convert RA/Dec → pixel coordinates using your (weak) WCS
    stars = SkyCoord(ra=[refstar_coords.ra, targetstar_coords.ra, compstar_coords.ra]*u.deg,
                     dec=[refstar_coords.dec, targetstar_coords.dec,
                          compstar_coords.dec]*u.deg,
                     frame='icrs')
    x, y = wcs.world_to_pixel(stars)

    print(f"Pixel positions: Star1 ({x[0]:.2f}, {
        y[0]:.2f}), Star2 ({x[1]:.2f}, {y[1]:.2f})")

    # Optional: global background subtraction (helps with large apertures)
    bkg = Background2D(data, box_size=(64, 64), filter_size=(3, 3),
                       bkg_estimator=SExtractorBackground())
    data_sub = data - bkg.background

    # Define large circular apertures
    positions = np.transpose((x, y))
    # aperture_radius is in arcsec, convert to pixels using WCS pixel scale
    pixel_scale = np.mean(
        np.abs(wcs.pixel_scale_matrix.diagonal())) * 3600  # arcsec/pixel
    aperture_radius_pix = aperture_radius / pixel_scale
    apertures = CircularAperture(positions, r=aperture_radius_pix)

    # Define sky annuli for local background per star
    # annulus_r_in and annulus_r_out are in arcsec, convert to pixels
    annulus_r_in_pix = annulus_r_in / pixel_scale
    annulus_r_out_pix = annulus_r_out / pixel_scale
    annuli = CircularAnnulus(
        positions, r_in=annulus_r_in_pix, r_out=annulus_r_out_pix)

    # plot the image with the apertures and annuli for visual check
    plt.figure(figsize=(10, 8))
    plt.imshow(data_sub, origin='lower', cmap='gray', vmin=np.percentile(
        data_sub, 5), vmax=np.percentile(data_sub, 99))
    apertures.plot(color='red', lw=2)
    annuli.plot(color='blue', lw=2)
    if savefig:
        png_path = f"{image_file.strip('.fits')}_apertures.png"
        print(f"Saving image with apertures to {png_path}")
        plt.savefig(f"{png_path}", dpi=300)
        plt.close()
    else:
        plt.show(block=True)

    # Perform photometry
    phot_table = aperture_photometry(data_sub, apertures, error=np.sqrt(
        np.abs(data)))  # basic Poisson + readnoise approx

    # Better background: use annulus to subtract local sky
    # or use ApertureStats for median/mode
    annulus_stats = aperture_photometry(data_sub, annuli)

    # Simple background subtraction per aperture (mean sky)
    bkg_mean = annulus_stats['aperture_sum'] / annuli.area
    phot_table['bkg_sub_flux'] = phot_table['aperture_sum'] - \
        bkg_mean * apertures.area

    # Uncertainties (rough but usable)
    # photutils already propagates
    phot_table['flux_err'] = phot_table['aperture_sum_err']

    # Relative photometry: flux ratio and its uncertainty
    refstar_flux = phot_table['bkg_sub_flux'][0]
    targetstar_flux = phot_table['bkg_sub_flux'][1]
    compstar_flux = phot_table['bkg_sub_flux'][2]
    ratio = targetstar_flux / compstar_flux
    ratio_err = ratio * \
        np.sqrt((phot_table['flux_err'][1]/targetstar_flux)**2 +
                (phot_table['flux_err'][2]/compstar_flux)**2)

    compstar_mags = {"U": 7.8, "B": 8.27, "V": 8.15}
    filters_names_equivalence = {"U": ["U", "U-Bessel", "U-Bessell"],
                                 "B": ["B", "B-Bessel", "B-Bessell"],
                                 "V": ["V", "V-Bessel", "V-Bessell"]}
    filter_name = header.get('FILTER', 'Unknown')
    filter_mag = [
        mag for mag, names in filters_names_equivalence.items() if filter_name in names]
    if filter_mag:
        compstar_mag = compstar_mags[filter_mag[0]]
        targetstar_mag = compstar_mag - 2.5 * np.log10(ratio)
        targetstar_mag_err = 2.5 / np.log(10) * ratio_err / ratio
    else:
        print(
            f"Warning: Filter '{filter_name}' not recognized, cannot determine magnitudes.")
        return

    # Get JD of the observation from DATE-OBS
    jd = header.get('JD', None)
    if jd is None:
        date_obs = header.get('DATE-OBS', None)
        if date_obs is not None:
            from astropy.time import Time
            t = Time(date_obs, format='isot', scale='utc')
            jd = t.jd
        else:
            print(
                "Warning: No JD or DATE-OBS found in header, cannot determine observation time.")
            jd = np.nan

    print(f"Image: {image_file}")
    print(f"JD: {jd} {filter_mag[0]}: {
          targetstar_mag:.3f} ± {targetstar_mag_err:.3f}")

    data = {"image_file": image_file,
            "jd": jd,
            "filter": filter_mag[0],
            "targetstar_mag": targetstar_mag,
            "targetstar_mag_err": targetstar_mag_err}

    return data


def main():
    args = parse_args()

    workdir = args.workdir
    refstar_coords = args.refstar
    targetstar_coords = args.targetstar
    compstar_coords = args.compstar
    aperture_radius = args.aperture
    annulus_r_in, annulus_r_out = args.annulus

    images_list = []
    path_to_images_list = args.images_list
    with open(path_to_images_list, 'r') as f:
        for line in f:
            image_file = line.strip()
            if image_file:  # skip empty lines
                images_list.append(image_file)

    output_data = {}
    for image_file in images_list:
        path_to_image = f"{workdir}/{image_file}"
        data = process_image(path_to_image,
                             refstar_coords,
                             targetstar_coords,
                             compstar_coords,
                             aperture_radius,
                             annulus_r_in,
                             annulus_r_out,
                             savefig=args.savefig)
        if data:
            # If first iteration, initialize dictionary keys
            if not output_data:
                for key in data.keys():
                    output_data[key] = []
            for key in data.keys():
                output_data[key].append(data[key])
            print(f"Processed {image_file} successfully.")
        else:
            print(f"Failed to process {image_file}.")
            import pdb
            pdb.set_trace()

    df = pd.DataFrame(output_data)
    output_file = os.path.join(workdir, args.output_csv)
    if os.path.exists(output_file) and not args.clobber:
        print(
            f"Error: {output_file} already exists. Use --clobber to overwrite.")
    else:
        df.to_csv(output_file, index=False)
        print(f"Saved photometry results to {output_file}")

    return


if __name__ == "__main__":
    main()
