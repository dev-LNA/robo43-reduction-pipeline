#!/bin/python3

from astropy.io import fits
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix FITS header by removing duplicate keywords."
    )
    parser.add_argument("input_dir", type=str,
                        help="Input directory containing FITS files.")

    return parser.parse_args()


class HeaderFixer:
    def __init__(self, args):
        self.workdir = args.input_dir
        self.filternames = {'Slot 0/Slot 1': 'U-Bessell',
                            'Slot 0/Slot 2': 'B-Bessell',
                            'Slot 0/Slot 3': 'V-Bessell',
                            'Slot 0/Slot 4': 'R-Bessell',
                            'Slot 0/Slot 5': 'I-Bessell',
                            'Slot 0/Slot 6': 'Clear',
                            'Slot 1/Slot 0': 'Halpha',
                            'Slot 2/Slot 0': 'Methane'
                            }
        self.exact_imagetypes = {'bias': 'ZERO', 'zero': 'ZERO', 'dark': 'DARK'}

    def get_imagetyp(self, object_str):
        """Compute IMAGETYP from OBJECT keyword value."""
        if not object_str:
            return None

        obj = object_str.strip().lower()

        if obj in self.exact_imagetypes:
            return self.exact_imagetypes[obj]
        elif 'flat' in obj:
            return 'FLAT'
        elif 'focus' in obj:
            return 'FOCUS'
        else:
            return 'OBJECT'

    def fix_header(self, imagename):
        with fits.open(imagename, mode="update") as hdul:
            header = hdul[0].header
            if 'FILTER' in header and header['FILTER'] in self.filternames:
                print(
                    f'Updating FILTER from {header["FILTER"]}',
                    f'to {self.filternames[header["FILTER"]]}',
                    f'for image {imagename}')
                header['FILTER'] = self.filternames[header['FILTER']]

            if 'OBJECT' in header and header['OBJECT']:
                imagetyp = self.get_imagetyp(header['OBJECT'])
                if imagetyp is not None:
                    current_imagetyp = header.get('IMAGETYP', 'LIGHT')
                    if current_imagetyp != imagetyp:
                        print(
                            f'Updating IMAGETYP from {current_imagetyp}',
                            f'to {imagetyp}',
                            f'for image {imagename}')
                        header['IMAGETYP'] = imagetyp
            else:
                print(f'WARNING: OBJECT keyword missing or empty for image {imagename}')

            hdul[0].header = header

    def main(self):
        filelist = glob.glob(f'{self.workdir}/*.fits')
        for filename in filelist:
            print(f'Processing {filename}')
            self.fix_header(filename)
        print('Done!')


if __name__ == "__main__":
    args = parse_args()
    fixer = HeaderFixer(args)
    fixer.main()
