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

    def fix_header(self, imagename):
        with fits.open(self.filename, mode="update") as hdul:
            header = hdul[0].header
            if header['FILTER'] in self.filternames:
                print(
                    f'Updating FILTER from {header["FILTER"]}',
                    f'to {self.filternames[header["FILTER"]]}',
                    f'for image {imagename}')
                header['FILTER'] = self.filternames[header['FILTER']]

            hdul[0].header = header

    def main(self):
        filelist = glob.glob(f'{self.workdir}/*.fits')
        for self.filename in filelist:
            print(f'Processing {self.filename}')
            self.fix_header(self.filename)
        print('Done!')


if __name__ == "__main__":
    args = parse_args()
    fixer = HeaderFixer(args)
    fixer.main()
