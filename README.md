# robo43-reduction-pipeline
Pipeline to process ROBO43 data

This repository contains a set o modules to process the ROBO43 photometric data. The modules are written in Python and use various libraries such as NumPy, SciPy, and Astropy.

## Usage

The module ``make_thumb.py'' produces a thumbnail image from a FITS file for a quick look.

The module ``collect_night_info.py'' collects the night information from the FITS headers and saves it to a CSV file.

The module ``make_calibration_frames.py'' creates master calibration frames (bias, dark, flat) from a set of raw calibration frames.

The module ``process_robo43_frames.py'' processes the raw science frames using the master calibration frames and performs astrometric calibration.

Module ``coadd_frames.py'' coadds multiple weighted science frames to improve the signal-to-noise ratio.

Module ``make_coloured_images.py'' creates coloured images from the processed science frames.

## TODO
- Add more documentation and examples
- Add module to gather the photometry
- Add more diagnostic plots
- Improve the astrometric calibration

## Example

The following example image from the Eta Car Nebula was obtained using the ROBO43 telescope and processed using this pipeline:
![Eta Car Nebula](ETACARNEBULA_rgb.png)
